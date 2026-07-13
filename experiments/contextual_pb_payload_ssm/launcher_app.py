"""Streamlit launcher and artifact browser for the payload-regime PB experiment."""
from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

EXP_DIR = Path(__file__).resolve().parent
ROOT, SCRIPT, RUNS, LOGS = EXP_DIR.parents[1], EXP_DIR / "Moving_payload_exp.py", EXP_DIR / "runs", EXP_DIR / ".launch_logs"
for path in (RUNS, LOGS): path.mkdir(parents=True, exist_ok=True)
for path in (ROOT, EXP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
# The experiment-agnostic RCP backend is maintained in the gate experiment.
_GATE_DIR = ROOT / "experiments" / "contextual_pb_gate_ssm"
if str(_GATE_DIR) not in sys.path: sys.path.insert(0, str(_GATE_DIR))

from payload_core import CONTEXT_FEATURE_META, CONTEXT_FEATURE_ORDER, FAIR_CONTEXT_DEFAULT
from rcp_backend import (RCPConfig, build_delete_command, build_describe_command, build_logs_command,
                         build_scp_command, build_submit_command, infer_job_state, is_terminal_success,
                         remote_run_dir, ssh_master_command)


SKIP = {"help", "plot_only", "no_show_plots", "run_id", "variants", "context_features"}
GROUPS = [
    ("Run & training", lambda d: d in {"task", "seed", "device", "train_batch", "val_batch", "test_batch", "epochs", "disturbance_only_epochs", "eval_every", "lr", "lr_min", "grad_clip"}),
    ("Slalom course & gates", lambda d: d.startswith("slalom_gate_") or d in {"horizon", "dt", "start_x_min", "start_x_max", "start_y_max", "corridor_limit", "goal_tol", "terminal_speed_tol", "control_limit"}),
    ("Nonlinear matched carrier", lambda d: d in {"pre_kp", "pre_kd", "slalom_carrier_cubic_stiffness", "slalom_carrier_quadratic_drag", "slalom_carrier_gyro_gain", "slalom_carrier_gyro_position_scale", "slalom_carrier_actuator_scale", "slalom_carrier_speed_loss"}),
    ("Hidden cargo & tether physics", lambda d: d.startswith("slalom_payload_") or d.startswith("slalom_tether_") or d.startswith("slalom_sway_") or d in {"slalom_physics_substeps", "slalom_test_mass_min", "slalom_test_mass_max", "slalom_test_tether_length_min", "slalom_test_tether_length_max", "slalom_tension_softness", "slalom_tension_scale", "slalom_max_tension", "slalom_max_extension"}),
    ("Collision, settling & curriculum", lambda d: d.startswith("slalom_")),
    ("Causal payload sensing", lambda d: d in {"payload_context_delay", "payload_obs_noise_sigma", "payload_context_dropout_p", "intervention_delay_steps", "intervention_eval"}),
    ("Legacy docking regime", lambda d: d.startswith("payload_") or d.startswith("test_payload") or d in {"regime_protocol", "test_switch_min", "test_switch_max"}),
    ("Disturbance process", lambda d: d.startswith("noise_") or d.startswith("gust_")),
    ("PB architecture", lambda d: d in {"feat_dim", "mb_hidden", "mb_layers", "mb_bound", "z_scale", "z_residual_gain", "use_w_augment", "w_augment_decay", "use_w0_clip", "w0_clip"}),
    ("Contextual SSM (ContextualDeepSSM)", lambda d: d.startswith("ctx_")),
    ("SSM & context lift", lambda d: d.startswith("ssm_") or d.startswith("mp_context")),
    ("Loss & output", lambda d: d.endswith("_weight") or d in {"post_switch_window", "recovery_radius", "sample_traj_count", "skip_plots"}),
]
VARIANTS = {"nominal": "Nominal pre-stabiliser", "disturbance_only": "PB+SSM: no context", "route_context": "PB+SSM: route context only", "context": "PB+SSM: route + payload context", "mad_context": "PB+SSM: route + payload MAD (s=1)", "contextual_ssm": "🆕 PB+SSM: ContextualDeepSSM"}


def specs():
    from Moving_payload_exp import build_parser
    bools, values, order = {}, {}, []
    for action in build_parser()._actions:
        if action.dest in SKIP: continue
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            entry = bools.setdefault(action.dest, {"true": None, "false": None, "default": build_parser().get_default(action.dest), "help": action.help or ""})
            entry["true" if isinstance(action, argparse._StoreTrueAction) else "false"] = action.option_strings[0]
        else:
            values[action.dest] = {"opt": action.option_strings[0], "default": build_parser().get_default(action.dest), "type": action.type, "choices": list(action.choices) if action.choices else None, "help": action.help or ""}
        if action.dest not in order: order.append(action.dest)
    return order, bools, values


def widget(dest, bools, values):
    key = f"p_{dest}"
    if dest in bools:
        e = bools[dest]; kwargs = {"help": e["help"], "key": key}
        if key not in st.session_state: kwargs["value"] = bool(e["default"])
        return st.checkbox(dest, **kwargs)
    e, default = values[dest], values[dest]["default"]
    kwargs = {"help": e["help"], "key": key}
    if e["choices"]:
        if key not in st.session_state: kwargs["index"] = e["choices"].index(default)
        return st.selectbox(dest, e["choices"], **kwargs)
    if e["type"] is int:
        if key not in st.session_state: kwargs["value"] = int(default or 0)
        return st.number_input(dest, step=1, **kwargs)
    if key not in st.session_state: kwargs["value"] = "" if default is None else str(default)
    return st.text_input(dest, **kwargs)


def argv_from_widgets(order, bools, values, run_id: str) -> tuple[list[str], list[str]]:
    argv, warnings = ["--no_show_plots", "--run_id", run_id], []
    for dest in order:
        value = st.session_state.get(f"p_{dest}")
        if dest in bools:
            e = bools[dest]
            if bool(value) != bool(e["default"]): argv.append(e["true"] if value else e["false"])
            continue
        e, default = values[dest], e_default(values, dest)
        if e["choices"]:
            if value != default: argv += [e["opt"], str(value)]
        elif e["type"] is int:
            if int(value) != int(default or 0): argv += [e["opt"], str(int(value))]
        elif e["type"] is float or isinstance(default, float):
            text = "" if value is None else str(value).strip()
            if not text: continue  # blank = leave unset (script default applies)
            try: parsed = float(text)
            except ValueError: warnings.append(f"{dest} is not a valid float and was skipped."); continue
            if default is None or parsed != float(default): argv += [e["opt"], str(parsed)]
        elif str(value).strip() != str(default or ""): argv += [e["opt"], str(value).strip()]
    return argv, warnings


def e_default(values, dest): return values[dest]["default"]


def launch(argv: list[str], run_id: str) -> None:
    log = LOGS / f"{run_id}.log"; handle = open(log, "w", buffering=1)
    proc = subprocess.Popen([sys.executable, "-u", str(SCRIPT), *argv], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    st.session_state.update(payload_proc=proc, payload_log=str(log), payload_run=run_id, payload_cmd=shlex.join([sys.executable, "-u", str(SCRIPT), *argv]))


def tail(path: str) -> str:
    try: return "\n".join(Path(path).read_text(errors="replace").splitlines()[-300:])
    except FileNotFoundError: return "(waiting for process output…)"


# ── EPFL RCP / Run:AI submission (shared backend, payload entry point) ──────
PAYLOAD_SCRIPT_REL = "experiments/contextual_pb_payload_ssm/Moving_payload_exp.py"
RCP_DEFAULTS = {
    "gaspar": "lmassai", "harbor_project": "context_leo", "image": "context", "tag": "v1.1",
    "runai_project": "sci-sti-gft-lmassai", "gpu": "0.1", "cpu": "4", "cpu_memory": "16G",
    "code_dir": "/home/lmassai/Performance_Boosting", "home_claim": "home",
    "scratch_claim": "sci-sti-gft-scratch", "job_name": "",
    "jumphost": "lmassai@jumphost.rcp.epfl.ch", "runai_bin": "~/.local/bin/runai",
}


def rcp_config(job_name: str | None = None) -> RCPConfig:
    values = {key: str(st.session_state.get(f"prcp_{key}", default)).strip()
              for key, default in RCP_DEFAULTS.items()}
    if job_name is not None:
        values["job_name"] = job_name
    return RCPConfig(script=PAYLOAD_SCRIPT_REL, **values)


def start_rcp_submission(argv: list[str], run_id: str, config: RCPConfig) -> None:
    logfile = LOGS / f"rcp_submit_{config.job_name}.log"
    cmd = build_submit_command(config, argv)
    handle = open(logfile, "w", buffering=1)
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=handle,
                            stderr=subprocess.STDOUT, cwd=str(ROOT), text=True)
    job = {"job_name": config.job_name, "run_id": run_id, "config": config,
           "submit_proc": proc, "submit_logf": handle, "submit_log": str(logfile),
           "submit_cmd": shlex.join(cmd), "submit_ret": None,
           "state": "Submitting", "status_output": "", "logs_output": ""}
    jobs = [j for j in st.session_state.get("prcp_jobs", []) if j["job_name"] != config.job_name]
    jobs.append(job)
    st.session_state["prcp_jobs"] = jobs


def run_rcp_cli(cmd: list[str], purpose: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=30, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{purpose} failed: Run:AI CLI not found at `{cmd[0]}`. "
                           "Install it or fix the CLI path in RCP settings.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{purpose} timed out after 30 seconds; check network/VPN access.") from exc


def cli_error(purpose: str, result: subprocess.CompletedProcess) -> str:
    detail = (result.stderr or result.stdout or "No CLI output").strip()
    return (f"{purpose} failed (exit {result.returncode}). Check `runai login`, project, "
            f"workload name, image, and PVC claims.\n\n{detail}")


def sync_rcp_results(config: RCPConfig, run_id: str) -> Path:
    """Copy the remote run dir to staging first, merge only after scp succeeds."""
    staging = Path(tempfile.mkdtemp(prefix=f"payload_{run_id}_", dir=str(LOGS)))
    try:
        result = subprocess.run(build_scp_command(config, run_id, staging), cwd=str(ROOT),
                                text=True, capture_output=True, timeout=900, check=False)
        if result.returncode:
            raise RuntimeError(
                "SSH transfer failed. The launcher's scp cannot type a password — open a shared "
                "session once in Terminal (valid 4 h):\n\n"
                f"    {ssh_master_command(config)}\n\n"
                f"then press Sync again. Also confirm `{remote_run_dir(config, run_id)}` exists.\n\n"
                + (result.stderr or result.stdout or "No scp output").strip())
        downloaded = staging / run_id
        if not downloaded.is_dir():
            raise RuntimeError(f"scp finished but `{downloaded}` was not created.")
        destination = RUNS / run_id
        shutil.copytree(downloaded, destination, dirs_exist_ok=True)
        return destination
    except FileNotFoundError as exc:
        raise RuntimeError("`scp` was not found on this Mac.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("SSH transfer timed out after 15 minutes.") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def render_rcp_settings() -> None:
    with st.expander("☁️ EPFL RCP / Run:AI settings", expanded=True):
        st.caption("Uses your existing Run:AI and SSH sessions; no password or token is stored. "
                   f"Remote entry point: `{PAYLOAD_SCRIPT_REL}` (must be pushed + pulled on the jumphost).")
        a, b, c = st.columns(3)
        a.text_input("GASPAR", value=RCP_DEFAULTS["gaspar"], key="prcp_gaspar")
        b.text_input("Harbor project", value=RCP_DEFAULTS["harbor_project"], key="prcp_harbor_project")
        c.text_input("Image", value=RCP_DEFAULTS["image"], key="prcp_image")
        a.text_input("Image tag", value=RCP_DEFAULTS["tag"], key="prcp_tag")
        b.text_input("Run:AI project", value=RCP_DEFAULTS["runai_project"], key="prcp_runai_project")
        c.text_input("GPU request", value=RCP_DEFAULTS["gpu"], key="prcp_gpu",
                      help="0.1 = fractional GPU; 1+ = whole device(s); 0 = none.")
        a.text_input("CPU cores", value=RCP_DEFAULTS["cpu"], key="prcp_cpu")
        b.text_input("CPU memory", value=RCP_DEFAULTS["cpu_memory"], key="prcp_cpu_memory")
        c.text_input("Job name (blank = run ID)", value=RCP_DEFAULTS["job_name"], key="prcp_job_name")
        st.text_input("Remote code directory", value=RCP_DEFAULTS["code_dir"], key="prcp_code_dir")
        a2, b2 = st.columns(2)
        a2.text_input("Home PVC claim", value=RCP_DEFAULTS["home_claim"], key="prcp_home_claim")
        b2.text_input("Scratch PVC claim", value=RCP_DEFAULTS["scratch_claim"], key="prcp_scratch_claim")
        st.text_input("Jumphost", value=RCP_DEFAULTS["jumphost"], key="prcp_jumphost")
        st.text_input("Run:AI CLI path", value=RCP_DEFAULTS["runai_bin"], key="prcp_runai_bin")


def submit_to_rcp(argv: list[str], run_id: str) -> None:
    raw_job = st.session_state.get("prcp_job_name", "").strip() or run_id
    job_name = re.sub(r"[^a-z0-9-]+", "-", raw_job.lower()).strip("-")[:63]
    if not job_name:
        st.error("Choose a job name containing letters or numbers.")
        return
    try:
        config = rcp_config(job_name)
        gpu_amount = float(config.gpu)
        device = str(st.session_state.get("p_device", "cpu")).strip()
        if gpu_amount > 0 and device != "cuda":
            st.error(f"GPU `{config.gpu}` requested but experiment device is `{device}`. "
                     "Set device to `cuda` (Run & training) or set GPU request to 0.")
            return
        argv = list(argv)
        if gpu_amount > 0 and "--require_cuda" not in argv:
            argv.append("--require_cuda")  # fail fast on a broken image/driver
        start_rcp_submission(argv, run_id, config)
        st.success(f"Submitting Run:AI workload `{job_name}` for run **{run_id}** — "
                   "track it under ☁️ Run:AI workloads below.")
    except (TypeError, ValueError, OSError) as exc:
        st.error(f"Could not start RCP submission: {exc}")


def render_rcp_jobs() -> None:
    jobs = st.session_state.get("prcp_jobs", [])
    if not jobs:
        return
    st.divider(); st.subheader("☁️ Run:AI workloads")

    any_submitting = False
    for job in jobs:
        sproc = job.get("submit_proc")
        if sproc is None:
            continue
        sret = sproc.poll()
        if sret is None:
            any_submitting = True
            continue
        handle = job.pop("submit_logf", None)
        if handle and not handle.closed:
            handle.close()
        job["submit_proc"], job["submit_ret"] = None, sret
        job["state"] = "Submitted" if sret == 0 and job["state"] == "Submitting" else \
                       ("Submit failed" if sret != 0 else job["state"])

    def _refresh(job: dict) -> None:
        result = run_rcp_cli(build_describe_command(job["config"]), "Status refresh")
        if result.returncode:
            st.error(cli_error(f"Status refresh (`{job['job_name']}`)", result))
        else:
            job["status_output"] = result.stdout or result.stderr
            job["state"] = infer_job_state(job["status_output"])
        logs = run_rcp_cli(build_logs_command(job["config"]), "Log refresh")
        if logs.returncode == 0:  # failing logs are normal while Pending
            job["logs_output"] = logs.stdout or logs.stderr

    def _refreshable(job: dict) -> bool:
        return (job.get("submit_proc") is None and job.get("submit_ret") == 0
                and not is_terminal_success(job["state"])
                and job["state"] not in ("Deleted", "Failed"))

    head = st.columns([2, 1, 1, 1])
    head[0].caption(f"{len(jobs)} workload(s) tracked this session")
    refresh_all = head[1].button("↻ Refresh all", width="stretch", disabled=any_submitting)
    auto = head[2].checkbox("Auto (5s)", value=False, key="prcp_auto_refresh", disabled=any_submitting)
    if head[3].button("🧹 Clear list", width="stretch",
                      help="Forget these UI entries (remote workloads are untouched)."):
        st.session_state["prcp_jobs"] = []
        (getattr(st, "rerun", None) or st.experimental_rerun)()
    if refresh_all:
        try:
            for job in jobs:
                if job.get("submit_proc") is None and job.get("submit_ret") == 0:
                    _refresh(job)
        except RuntimeError as exc:
            st.error(str(exc))

    for job in jobs:
        state = job["state"]
        icon = "✅" if is_terminal_success(state) else ("❌" if state in ("Failed", "Submit failed", "Deleted") else "🔵")
        with st.expander(f"{icon} `{job['job_name']}` — **{state}**  ·  run `{job['run_id']}`",
                         expanded=(len(jobs) == 1 or state in ("Failed", "Submit failed"))):
            if job.get("submit_proc") is not None:
                st.info("Submission request is in progress…")
            elif job.get("submit_ret") != 0:
                st.error(f"Submission failed (exit {job.get('submit_ret')}). Check `runai login`, "
                         f"project, image visibility, code path, and PVC claims.\n\n{tail(job.get('submit_log', ''))}")
            if st.button("🗑 Delete / cancel", key=f"pdel_{job['job_name']}",
                         disabled=job.get("submit_proc") is not None):
                try:
                    result = run_rcp_cli(build_delete_command(job["config"]), "Delete/cancel")
                    if result.returncode:
                        st.error(cli_error("Delete/cancel", result))
                    else:
                        job["state"] = "Deleted"; st.success(result.stdout or "Delete request accepted.")
                except RuntimeError as exc:
                    st.error(str(exc))
            st.code(job.get("submit_cmd", ""), language="bash")
            if job.get("status_output"): st.code(job["status_output"], language="text")
            if job.get("logs_output"):
                st.caption("Latest log snapshot (↻ Refresh all to update)")
                st.code("\n".join(job["logs_output"].splitlines()[-400:]), language="text")

    st.markdown("**Sync results into Browse artifacts**")
    st.caption("Sync piggybacks on a shared SSH session. If none is open yet, run this once in "
               f"Terminal (one password, valid 4 h): `{ssh_master_command(jobs[0]['config'])}`")
    allow_partial = st.checkbox("Allow partial sync (job not complete)", value=False, key="prcp_partial")
    seen: list[str] = []
    for job in jobs:
        if job["run_id"] not in seen:
            seen.append(job["run_id"])
    for rid in seen:
        group = [j for j in jobs if j["run_id"] == rid]
        done = sum(1 for j in group if is_terminal_success(j["state"]))
        row = st.columns([3, 1])
        row[0].markdown(f"`{rid}` — {done}/{len(group)} job(s) completed")
        if row[1].button("⇣ Sync", key=f"psync_{rid}", type="primary", width="stretch",
                         disabled=not ((done == len(group)) or allow_partial)):
            try:
                with st.spinner("Copying remote artifacts over SSH…"):
                    destination = sync_rcp_results(group[0]["config"], rid)
                st.success(f"Synced into `{destination.relative_to(ROOT)}` — open Browse artifacts.")
            except RuntimeError as exc:
                st.error(str(exc))

    pending = [j for j in jobs if _refreshable(j)]
    if auto and not any_submitting and pending:
        try:
            for job in pending:
                _refresh(job)
        except RuntimeError as exc:
            st.error(str(exc))
        time.sleep(5)
        (getattr(st, "rerun", None) or st.experimental_rerun)()
    if any_submitting:
        time.sleep(1)
        (getattr(st, "rerun", None) or st.experimental_rerun)()


def render_launch() -> None:
    order, bools, values = specs(); st.subheader("Tethered Cargo Slalom")
    st.caption("A strongly nonlinear, origin-stable carrier tows a hidden nonlinear cargo body through three alternating precision gates. Nominal and true carrier transitions are matched, so PB reconstructs only tapered process noise; payload context matters because carrier, cargo, and every tether segment must clear the gates and settle.")

    target = st.radio("Execution target", ["Local", "EPFL RCP"], horizontal=True, key="prcp_target")
    # A fresh RCP selection should use its requested GPU; restore the CPU default
    # when returning to Local so local behavior is unchanged.
    previous = st.session_state.get("_prcp_previous_target")
    if target == "EPFL RCP" and previous != "EPFL RCP":
        if str(st.session_state.get("p_device", "cpu")).strip() in ("", "cpu"):
            st.session_state["p_device"] = "cuda"; st.session_state["_prcp_device_auto"] = True
    elif (target == "Local" and previous == "EPFL RCP"
          and st.session_state.get("_prcp_device_auto")
          and str(st.session_state.get("p_device", "")).strip() == "cuda"):
        st.session_state["p_device"] = "cpu"; st.session_state["_prcp_device_auto"] = False
    st.session_state["_prcp_previous_target"] = target
    if target == "EPFL RCP":
        render_rcp_settings()

    with st.form("payload_launch", border=False):
        st.markdown("#### Controllers")
        cols = st.columns(2)
        for i, (key, label) in enumerate(VARIANTS.items()): cols[i % 2].checkbox(label, value=True, key=f"variant_{key}")
        st.markdown("#### Causal context signal")
        st.caption("Route-only and full-context controllers have the same context width; payload slots are zeroed for route-only. No future state or gate outcome is exposed.")
        cols = st.columns(3)
        for i, key in enumerate(CONTEXT_FEATURE_ORDER): cols[i % 3].checkbox(CONTEXT_FEATURE_META[key][1], value=key in FAIR_CONTEXT_DEFAULT, key=f"context_{key}")
        used = set()
        for title, predicate in GROUPS:
            members = [d for d in order if d not in used and predicate(d)]; used.update(members)
            if members:
                with st.expander(title, expanded=title in {"Run & training", "Slalom course & gates", "Hidden cargo & tether physics"}):
                    for dest in members: widget(dest, bools, values)
        remaining = [d for d in order if d not in used]
        if remaining:
            with st.expander("Other parameters"):
                for dest in remaining: widget(dest, bools, values)
        submitted = st.form_submit_button("🚀 Launch cargo slalom", type="primary", width="stretch")
    if submitted:
        variants = [key for key in VARIANTS if st.session_state.get(f"variant_{key}")]; context = [key for key in CONTEXT_FEATURE_ORDER if st.session_state.get(f"context_{key}")]
        if not variants: st.error("Choose at least one controller.")
        elif not context: st.error("Choose at least one causal context feature.")
        else:
            run_id = f"tethered_ui_{datetime.now():%Y%m%d_%H%M%S}"; argv, warnings = argv_from_widgets(order, bools, values, run_id); argv += ["--variants", ",".join(variants), "--context_features", ",".join(context)]
            for warning in warnings: st.warning(warning)
            if st.session_state.get("prcp_target") == "EPFL RCP":
                submit_to_rcp(argv, run_id)
            else:
                launch(argv, run_id); st.success(f"Launched `{run_id}` locally.")
    proc = st.session_state.get("payload_proc")
    if proc is not None:
        status = proc.poll(); st.divider(); st.caption(f"Run `{st.session_state.get('payload_run')}` — {'running' if status is None else f'exit {status}'}")
        st.code(st.session_state.get("payload_cmd", ""), language="bash"); st.code(tail(st.session_state.get("payload_log", "")), language="text")
    render_rcp_jobs()


def render_browse() -> None:
    with st.expander("☁️ Sync RCP run by ID", expanded=False):
        st.caption("Copies a remote payload run directory even when the UI no longer tracks the job.")
        manual_id = st.text_input("Remote run ID", value="", placeholder="payload_ui_20260710_140000",
                                  key="prcp_manual_sync_id")
        st.caption("If sync fails in batch mode, open a shared SSH session once in Terminal: "
                   f"`{ssh_master_command(rcp_config('manual'))}`")
        if st.button("⇣ Sync this run", type="primary", key="prcp_manual_sync_btn"):
            rid = manual_id.strip()
            if not rid:
                st.error("Enter the remote run ID first.")
            else:
                try:
                    with st.spinner("Copying remote artifacts over SSH…"):
                        destination = sync_rcp_results(rcp_config("manual"), rid)
                    st.success(f"Synced into `{destination.relative_to(ROOT)}`.")
                except RuntimeError as exc:
                    st.error(str(exc))

    runs = sorted([p for p in RUNS.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs: st.info("No payload runs yet."); return
    selected = st.selectbox("Payload run", [p.name for p in runs]); run = RUNS / selected; top = st.columns([1, 3])
    if top[0].button("🔁 Re-plot", width="stretch"):
        launch(["--no_show_plots", "--plot_only", selected], f"replot_{selected}"); st.success(f"Regenerating artifacts for `{selected}`.")
    top[1].caption(str(run.relative_to(ROOT)))
    metrics_file = run / "metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text()); rows = [{"variant": key, **{k: round(float(v), 4) for k, v in values.items() if isinstance(v, (int, float))}} for key, values in metrics.items()]
        st.dataframe(rows, hide_index=True, width="stretch")
    for name in ("interpretation.txt", "interventions.json", "config.json"):
        path = run / name
        if path.exists():
            with st.expander(name, expanded=name == "interpretation.txt"):
                st.text(path.read_text() if path.suffix == ".txt" else json.dumps(json.loads(path.read_text()), indent=2))
    gifs, pngs, pdfs = sorted(run.glob("*.gif")), sorted(run.glob("*.png")), sorted(run.glob("*.pdf"))
    if gifs:
        st.subheader("Animations")
        for path in gifs:
            st.image(str(path), caption=path.name)
    if pngs:
        st.subheader("Figures"); columns = st.columns(2)
        for i, path in enumerate(pngs): columns[i % 2].image(str(path), caption=path.name, width="stretch")
    if pdfs:
        st.subheader("Storyboards")
        for path in pdfs: st.download_button(path.name, path.read_bytes(), file_name=path.name)


def main() -> None:
    st.set_page_config(page_title="Tethered Cargo Slalom", layout="wide"); st.title("📦 Contextual PB + SSM — Tethered Cargo Slalom")
    st.caption(f"`{SCRIPT.relative_to(ROOT)}`")
    launch_tab, browse_tab = st.tabs(["🚀 Launch run", "📊 Browse artifacts"])
    with launch_tab: render_launch()
    with browse_tab: render_browse()


def _under_streamlit_run() -> bool:
    """True if executed via ``streamlit run`` (a ScriptRunContext is attached)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        try:
            from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
        except Exception:
            return False
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _under_streamlit_run():
        main()
    else:
        # Launched as a plain script (e.g. an IDE Run button). Streamlit needs its
        # own runtime, so re-exec through the streamlit CLI instead of emitting
        # "missing ScriptRunContext" warnings with nothing rendered.
        print("This is a Streamlit app — relaunching via `streamlit run`…\n")
        raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", __file__, *sys.argv[1:]]))
