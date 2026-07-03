"""Pure helpers for submitting and synchronizing gate runs on EPFL RCP."""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class RCPConfig:
    gaspar: str = "lmassai"
    harbor_project: str = "context_leo"
    image: str = "context"
    tag: str = "v1.1"
    runai_project: str = "sci-sti-gft-lmassai"
    gpu: str = "0.1"
    cpu: str = "4"
    cpu_memory: str = "16G"
    code_dir: str = "/home/lmassai/Performance_Boosting"
    home_claim: str = "home"
    scratch_claim: str = "sci-sti-gft-scratch"
    job_name: str = ""
    jumphost: str = "lmassai@jumphost.rcp.epfl.ch"
    runai_bin: str = "~/.local/bin/runai"

    @property
    def image_uri(self) -> str:
        return f"registry.rcp.epfl.ch/{self.harbor_project}/{self.image}:{self.tag}"


def gpu_request_args(gpu: str | float | int) -> list[str]:
    """Translate a GPU amount into current Run:AI fractional/full-device flags."""
    value = str(gpu).strip()
    try:
        amount = float(value)
    except ValueError as exc:
        raise ValueError("GPU request must be 0, a fraction such as 0.1, or a whole number") from exc
    if amount < 0:
        raise ValueError("GPU request cannot be negative")
    if amount == 0:
        return []
    if not amount.is_integer():
        if not 0 < amount < 1:
            raise ValueError("Fractional GPU requests must be between 0 and 1")
        return ["--gpu-portion-request", value]
    if int(amount) < 1:
        raise ValueError("Whole-device GPU request must be at least 1")
    return ["--gpu-devices-request", str(int(amount))]


def remote_run_dir(config: RCPConfig, run_id: str) -> PurePosixPath:
    return PurePosixPath(config.code_dir) / "experiments/contextual_pb_gate_ssm/runs" / run_id


def cpu_thread_env_prefix(config: RCPConfig) -> str:
    """Shell prefix pinning BLAS/OpenMP thread pools to the requested CPU cores.

    Without it torch/numpy size their pools to the NODE's core count (64+),
    while the cgroup only grants ``config.cpu`` cores — oversubscription that
    CFS throttling turns into a slowdown, worst on CPU-only (gpu=0) jobs.
    """
    try:
        cores = max(1, int(float(config.cpu)))
    except (TypeError, ValueError):
        return ""
    return f"export OMP_NUM_THREADS={cores} MKL_NUM_THREADS={cores} && "


def build_remote_command(config: RCPConfig, argv: list[str]) -> str:
    """Build the container shell command, preserving launcher argv exactly."""
    script = "experiments/contextual_pb_gate_ssm/Moving_gate_exp.py"
    python_cmd = shlex.join(["python3", script, *argv])
    return f"{cpu_thread_env_prefix(config)}cd {shlex.quote(config.code_dir)} && {python_cmd}"


def build_submit_command(config: RCPConfig, argv: list[str]) -> list[str]:
    job_name = config.job_name.strip()
    if not job_name:
        raise ValueError("Job name is required")
    cmd = [
        str(Path(config.runai_bin).expanduser()), "training", "standard", "submit", job_name,
        "--project", config.runai_project,
        "--image", config.image_uri,
        *gpu_request_args(config.gpu),
        *(["--cpu-core-request", config.cpu] if config.cpu.strip() else []),
        *(["--cpu-memory-request", config.cpu_memory] if config.cpu_memory.strip() else []),
        "--existing-pvc", f"claimname={config.scratch_claim},path=/scratch",
        "--existing-pvc", f"claimname={config.home_claim},path=/home/{config.gaspar}",
        "--command", "--", "bash", "-lc", build_remote_command(config, argv),
    ]
    return cmd


def build_describe_command(config: RCPConfig) -> list[str]:
    return [str(Path(config.runai_bin).expanduser()), "training", "standard", "describe",
            config.job_name, "-p", config.runai_project]


def build_logs_command(config: RCPConfig, follow: bool = False) -> list[str]:
    cmd = [str(Path(config.runai_bin).expanduser()), "training", "standard", "logs",
           config.job_name, "-p", config.runai_project]
    if follow:
        cmd.append("--follow")
    return cmd


def build_delete_command(config: RCPConfig) -> list[str]:
    return [str(Path(config.runai_bin).expanduser()), "training", "standard", "delete",
            config.job_name, "-p", config.runai_project]


# The launcher runs scp without a TTY, so it can never answer a password
# prompt: fail fast (BatchMode) and reuse a shared master session when the
# user has opened one with the matching ControlPath (see ssh_master_command).
SSH_SHARED_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=%d/.ssh/sockets/%r@%h-%p",
]


def ssh_master_command(config: RCPConfig) -> str:
    """One-liner the user runs in a terminal to open a shared SSH session.

    Prompts for the password once; the connection then stays available for
    4 hours so the launcher's non-interactive scp can piggyback on it.
    """
    return ("ssh -o ControlMaster=auto -o ControlPath='%d/.ssh/sockets/%r@%h-%p' "
            f"-o ControlPersist=4h {config.jumphost} true")


def build_scp_command(config: RCPConfig, run_id: str, staging_dir: Path) -> list[str]:
    source = f"{config.jumphost}:{shlex.quote(str(remote_run_dir(config, run_id)))}"
    return ["scp", *SSH_SHARED_OPTS, "-r", source, str(staging_dir)]


def infer_job_state(describe_output: str) -> str:
    """Extract a useful state from human-readable Run:AI describe output."""
    patterns = (
        r"(?im)^\s*(?:status|state|phase)\s*:\s*([\w -]+?)\s*$",
        r"(?im)^\s*workload status\s*:\s*([\w -]+?)\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, describe_output)
        if match:
            return match.group(1).strip()
    lower = describe_output.lower()
    for state in ("completed", "succeeded", "failed", "deleted", "running", "pending", "creating"):
        if re.search(rf"\b{state}\b", lower):
            return state.title()
    return "Unknown"


def is_terminal_success(state: str) -> bool:
    return state.strip().lower() in {"completed", "succeeded", "success"}
