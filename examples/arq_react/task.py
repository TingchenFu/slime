from dataclasses import dataclass, field
from typing import Any
import os
import yaml


@dataclass
class TaskMetadata:
    """Parsed metadata for a single AirsBench task."""

    task_name: str
    folder_path: str
    metric_lower_is_better: bool
    file_export_globs: list[str]
    container_python_requirements: list[str]
    evaluate_container_python_requirements: list[str]
    logging_info: dict[str, Any]
    prepare_code_python_requirements: list[str] | None
    prepare_script: str
    evaluate_prepare_script: str
    evaluate_script: str
    supporting_scripts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "folder_path": self.folder_path,
            "metric_lower_is_better": self.metric_lower_is_better,
            "file_export_globs": self.file_export_globs,
            "container_python_requirements": self.container_python_requirements,
            "evaluate_container_python_requirements": self.evaluate_container_python_requirements,
            "logging_info": self.logging_info,
            "prepare_code_python_requirements": self.prepare_code_python_requirements,
            "prepare_script": self.prepare_script,
            "evaluate_prepare_script": self.evaluate_prepare_script,
            "evaluate_script": self.evaluate_script,
            "supporting_scripts": self.supporting_scripts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskMetadata":
        return cls(
            task_name=data["task_name"],
            folder_path=data["folder_path"],
            metric_lower_is_better=data["metric_lower_is_better"],
            file_export_globs=data["file_export_globs"],
            container_python_requirements=data["container_python_requirements"],
            evaluate_container_python_requirements=data["evaluate_container_python_requirements"],
            logging_info=data["logging_info"],
            prepare_code_python_requirements=data.get("prepare_code_python_requirements"),
            prepare_script=data["prepare_script"],
            evaluate_prepare_script=data["evaluate_prepare_script"],
            evaluate_script=data["evaluate_script"],
            supporting_scripts=data.get("supporting_scripts", []),
        )


def load_task_metadata(airsbench_tasks_root: str, task_name: str) -> TaskMetadata:
    """Load metadata.yaml and locate scripts for a task."""
    folder_path = os.path.join(airsbench_tasks_root, task_name)
    meta_path = os.path.join(folder_path, "metadata.yaml")

    with open(meta_path) as f:
        meta = yaml.safe_load(f)

    # Locate supporting scripts
    supporting_scripts = []
    for fname in os.listdir(folder_path):
        if fname.startswith("gold_submission"):
            continue
        if fname.endswith(".py") and fname not in [
            "prepare.py",
            "evaluate.py",
            "evaluate_prepare.py",
            "custom_labels.py",
            "download.py",
            "__init__.py",
        ]:
            supporting_scripts.append(os.path.join(folder_path, fname))

    return TaskMetadata(
        task_name=task_name,
        folder_path=folder_path,
        metric_lower_is_better=meta["metric_lower_is_better"],
        file_export_globs=meta["file_export_globs"],
        container_python_requirements=meta.get("container_python_requirements", []),
        evaluate_container_python_requirements=meta.get("evaluate_container_python_requirements", []),
        logging_info=meta["logging_info"],
        prepare_code_python_requirements=meta.get("prepare_code_python_requirements"),
        prepare_script=os.path.join(folder_path, "prepare.py"),
        evaluate_prepare_script=os.path.join(folder_path, "evaluate_prepare.py"),
        evaluate_script=os.path.join(folder_path, "evaluate.py"),
        supporting_scripts=supporting_scripts,
    )
