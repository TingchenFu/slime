from dataclasses import dataclass
from typing import Any
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class SandboxHandle:
    """Wraps a DojoARQSandboxEnvironment so tools can ``await handle.exec(...)``.

    Created by ``boot_sandbox`` once per rollout; passed into the tool
    closures via ``build_default_tools(handle)``.

    ``eq=False`` keeps the default identity-based ``__eq__``/``__hash__`` so
    instances stay hashable — they're tracked in the ``_active_sandboxes`` set
    as an identity registry (one handle per rollout). A dataclass-generated
    ``__eq__`` would set ``__hash__ = None`` and make ``set.add`` raise
    ``TypeError: unhashable type``.
    """

    env: Any  # DojoSandboxEnvironment instance

    async def exec(
        self,
        cmd: list[str],
        *,
        input: str | bytes | None = None,
        timeout: int | None = None,
        user: str | None = None,
        env: dict[str, str] | None = None,
    ) -> Any:
        return await self.env.exec(
            cmd=cmd,
            input=input,
            timeout=timeout,
            user=user,
            env=env or {},
        )


async def boot_sandbox(
    *,
    task_name: str,
    sandbox_config: Any,
    metadata: dict[str, Any] | None = None,
    sandbox_cls: Any | None = None,
) -> SandboxHandle:
    """Initialize a sandbox via the upstream class methods.

    ``sandbox_cls`` defaults to ``DojoSandboxEnvironment``; pass a subclass
    (e.g. our ``DojoARQSandboxEnvironment``) to get ARQ-specific setup like
    the ``/workspace/data → /root/data`` symlink + pip-install of
    container_python_requirements.
    """
    if sandbox_cls is None:
        from atlas.utils.dojo_sandbox import DojoSandboxEnvironment as sandbox_cls

    envs = await sandbox_cls.sample_init(
        task_name=task_name,
        config=sandbox_config,
        metadata=metadata or {},
    )
    handle = SandboxHandle(env=envs["default"])
    # Register so signal/atexit handlers can return the leased AgentBox
    # machine to the pool if the process is killed before teardown_sandbox
    # runs (e.g. SLURM `scancel` SIGTERM). See free_active_sandboxes.
    with _active_sandboxes_lock:
        _active_sandboxes.add(handle)
    return handle


async def teardown_sandbox(
    handle: SandboxHandle,
    *,
    task_name: str,
    sandbox_config: Any,
    interrupted: bool = False,
    sandbox_cls: Any | None = None,
) -> None:
    # Claim ownership before freeing: if a signal/atexit handler already
    # released this sandbox (free_active_sandboxes), skip — don't double-free.
    with _active_sandboxes_lock:
        claimed = handle in _active_sandboxes
        _active_sandboxes.discard(handle)
    if not claimed:
        return

    if sandbox_cls is None:
        from atlas.utils.dojo_sandbox import DojoSandboxEnvironment as sandbox_cls

    await sandbox_cls.sample_cleanup(
        task_name=task_name,
        config=sandbox_config,
        environments={"default": handle.env},
        interrupted=interrupted,
    )


# ---------------------------------------------------------------------------
# Emergency machine release for signal/atexit handlers.
#
# The AgentBox machine is leased inside boot_sandbox (sample_init, which
# calls manager.start_machine when _own_machine=True) and normally returned
# by teardown_sandbox's finally. But the manager has NO server-side lease
# TTL / dead-client reclamation: a machine returns to the pool ONLY via an
# explicit free_machine. If the process is killed (SLURM scancel SIGTERM,
# Ctrl-C) the async teardown may never run, leaking the machine. This sync
# path frees still-leased sandboxes from a signal handler (no event loop).
# ---------------------------------------------------------------------------
_active_sandboxes: set[SandboxHandle] = set()
_active_sandboxes_lock = threading.Lock()

# Bound the emergency gRPC calls so a hung free can't outlast SLURM's grace
# window (KillWait, default 30s) before the SIGKILL lands.
_EMERGENCY_FREE_TIMEOUT_S = 8.0


def _free_sandbox_sync(handle: SandboxHandle, *, timeout: float = _EMERGENCY_FREE_TIMEOUT_S) -> None:
    """Synchronous mirror of DojoSandboxEnvironment.sample_cleanup's body.

    Frees the container and (if we own it) returns the machine to the pool.
    Best-effort: every step is guarded so one failure can't strand the
    machine lease.
    """
    env = handle.env
    try:
        notebook = getattr(env, "notebook", None)
        if notebook is not None:
            notebook.__exit__(None, None, None)
            env.notebook = None
    except Exception as exc:
        logger.warning("emergency sandbox free: notebook exit failed: %s", exc)
    try:
        env.machine.free_container(env.container, timeout=timeout)
    except Exception as exc:
        logger.warning("emergency sandbox free: free_container failed: %s", exc)
    finally:
        try:
            if getattr(env, "_own_machine", False):
                env.manager.free_machine(env.machine, timeout=timeout)
        except Exception as exc:
            logger.warning("emergency sandbox free: free_machine failed: %s", exc)


def free_active_sandboxes(*, timeout: float = _EMERGENCY_FREE_TIMEOUT_S) -> None:
    """Return any still-leased AgentBox machines to the pool.

    Safe to call from a signal or atexit handler: fully synchronous, needs
    no event loop. Each handle is claimed under the lock, so this never
    races teardown_sandbox into a double-free.
    """
    while True:
        with _active_sandboxes_lock:
            if not _active_sandboxes:
                return
            handle = _active_sandboxes.pop()
        _free_sandbox_sync(handle, timeout=timeout)
