import dataclasses
import json
import logging
import os
import queue
import threading
import time
from typing import override
from collections.abc import Callable
import asyncio

from atlas.agents.llm_agents.react_agent import (
    ON_CONTINUE_STR,
    TOOL_FACTORY,
    ReactInspectAIAgent,
)
from atlas.core.agent import Result
from atlas.utils.dojo_sandbox import DojoSandboxEnvironment, DojoSandboxEnvironmentConfig
from inspect_ai import Task as InspectTask
from inspect_ai import eval as inspect_ai_eval
from inspect_ai import task as inspect_ai_task
from inspect_ai._util.content import ContentText
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.model._chat_message import ChatMessageAssistant, ChatMessageTool
from inspect_ai.scorer import match
from inspect_ai.tool import Tool, tool
from inspect_ai.util import SandboxEnvironmentSpec, sandboxenv


logger = logging.getLogger(__name__)


@sandboxenv(name="dojo_arq")
class DojoARQSandboxEnvironment(DojoSandboxEnvironment):
    @override
    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: DojoSandboxEnvironmentConfig,
        metadata: dict[str, str],
    ) -> dict[str, "DojoARQSandboxEnvironment"]:
        """
        Creates and initializes a machine and container for a new sample.
        """
        manager = cls._manager
        if manager is None:
            raise RuntimeError("DojoRPCManager has not been set. Please call DojoSandboxEnvironment.set_manager(your_manager) before running the evaluation.")

        _own_machine = config.machine is None

        def blocking_init():
            """Synchronous setup logic to be run in a thread."""

            if config.machine is not None:
                machine = config.machine
            else:
                machine = manager.start_machine()

            # Start the container on the allocated machine
            container = machine.start_container(config=config.container_config)

            with container.shell(work_dir=config.container_config.working_dir) as shell:
                # Create symlink so that ./data (relative to /workspace) points to /root/data
                # This is needed because generated code often uses ./data/train etc.,
                # but bind_inputs_dir mounts data at /root/data
                symlink_cmd = f"mkdir -p /root/data && rm -rf {config.container_config.working_dir}/data && ln -sfn /root/data {config.container_config.working_dir}/data"
                for block in shell.execute(symlink_cmd, timeout=10):
                    if block.output.strip():
                        logger.info(f"[symlink] {block.output.strip()}")
                for debug_cmd in [
                    "ls -la /root/data/ 2>&1 || echo 'MOUNT_FAILED: /root/data does not exist'",
                    f"readlink -f {config.container_config.working_dir}/data 2>&1",
                ]:
                    for block in shell.execute(debug_cmd, timeout=10):
                        if block.output.strip():
                            logger.info(f"[symlink debug] {block.output.strip()}")

                reqs = [f'"{r}"' for r in metadata.get("container_python_requirements", [])]
                install_cmd = "pip install " + " ".join(reqs)
                for block in shell.execute(install_cmd, timeout=1800):
                    if block.output.strip():
                        logger.info(f"[deps] {block.output.strip()}")

            logger.info(f"agent id {task_name} finish prepare the container")

            # Start a notebook
            notebook = container.notebook().__enter__()
            return machine, container, notebook

        machine, container, notebook = await asyncio.to_thread(blocking_init)

        sandbox_env = cls(
            config=config,
            manager=manager,
            machine=machine,
            container=container,
            notebook=notebook,
            _own_machine=_own_machine,
        )
        return {"default": sandbox_env}
