import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_experiment_controlled_xy as controlled_xy


def parse_args() -> argparse.Namespace:
    exp_defaults = controlled_xy.parse_args([])
    parser = argparse.ArgumentParser(
        "Diagnostic plot: check that the overall operator M(w,z) decays to zero for decaying inputs"
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--horizon", type=int, default=180)
    parser.add_argument("--w_dim", type=int, default=4)
    parser.add_argument("--z_dim", type=int, default=controlled_xy.context_dim())
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--no_show_plots", action="store_true")

    parser.add_argument("--w_decay_power", type=float, default=1.15)
    parser.add_argument("--z_decay_power", type=float, default=1.05)
    parser.add_argument("--w_scale", type=float, default=0.30)
    parser.add_argument("--z_scale", type=float, default=0.75)
    parser.add_argument("--tail_window", type=int, default=16)
    parser.add_argument("--tail_ratio_threshold", type=float, default=0.08)

    parser.add_argument("--feat_dim", type=int, default=exp_defaults.feat_dim)
    parser.add_argument("--mb_hidden", type=int, default=exp_defaults.mb_hidden)
    parser.add_argument("--mb_layers", type=int, default=exp_defaults.mb_layers)
    parser.add_argument("--z_residual_gain", type=float, default=exp_defaults.z_residual_gain)
    parser.add_argument("--mb_bound", type=float, default=exp_defaults.mb_bound)
    parser.add_argument("--ssm_param", type=str, default=exp_defaults.ssm_param, choices=["lru", "tv"])
    parser.add_argument("--ssm_layers", type=int, default=exp_defaults.ssm_layers)
    parser.add_argument("--ssm_d_model", type=int, default=exp_defaults.ssm_d_model)
    parser.add_argument("--ssm_d_state", type=int, default=exp_defaults.ssm_d_state)
    parser.add_argument("--ssm_ff", type=str, default=exp_defaults.ssm_ff)
    parser.add_argument("--mp_context_lift", dest="mp_context_lift", action="store_true")
    parser.add_argument("--no_mp_context_lift", dest="mp_context_lift", action="store_false")
    parser.set_defaults(mp_context_lift=bool(exp_defaults.mp_context_lift))
    parser.add_argument("--mp_context_lift_type", type=str, default=exp_defaults.mp_context_lift_type)
    parser.add_argument("--mp_context_lift_dim", type=int, default=exp_defaults.mp_context_lift_dim)
    parser.add_argument("--mp_context_hidden_dim", type=int, default=exp_defaults.mp_context_hidden_dim)
    parser.add_argument("--mp_context_decay_law", type=str, default=exp_defaults.mp_context_decay_law)
    parser.add_argument("--mp_context_decay_rate", type=float, default=exp_defaults.mp_context_decay_rate)
    parser.add_argument("--mp_context_decay_power", type=float, default=exp_defaults.mp_context_decay_power)
    parser.add_argument("--mp_context_decay_horizon", type=int, default=exp_defaults.mp_context_decay_horizon)
    parser.add_argument("--mp_context_lp_p", type=float, default=exp_defaults.mp_context_lp_p)
    parser.add_argument("--mp_context_scale", type=float, default=exp_defaults.mp_context_scale)
    return parser.parse_args()


def make_decaying_sequence(
    *,
    horizon: int,
    dim: int,
    scale: float,
    decay_power: float,
    freq0: float,
    device: torch.device,
) -> torch.Tensor:
    t = torch.arange(int(horizon), device=device, dtype=torch.float32).view(1, int(horizon), 1)
    amps = torch.linspace(float(scale), 0.45 * float(scale), int(dim), device=device).view(1, 1, int(dim))
    freqs = torch.linspace(float(freq0), float(freq0) + 0.18, int(dim), device=device).view(1, 1, int(dim))
    phases = torch.linspace(0.0, float(np.pi) / 2.0, int(dim), device=device).view(1, 1, int(dim))
    oscillatory = 0.65 * torch.cos(freqs * t + phases) + 0.35 * torch.sin(0.7 * freqs * t + 0.5 * phases)
    decay = (1.0 + t) ** (-float(decay_power))
    return amps * decay * oscillatory


def plot_diagnostic(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    w_seq: torch.Tensor,
    z_seq: torch.Tensor,
    z_lift: torch.Tensor,
    w_mp: torch.Tensor,
    mp_feat: torch.Tensor,
    m_out: torch.Tensor,
    metrics: dict[str, float],
) -> None:
    show_plots = not args.no_show_plots
    plt = controlled_xy.get_plt(show_plots)
    controlled_xy.setup_plot_style(plt)

    time = np.arange(int(args.horizon))
    w_np = w_seq.squeeze(0).cpu().numpy()
    z_np = z_seq.squeeze(0).cpu().numpy()
    z_lift_np = z_lift.squeeze(0).cpu().numpy()
    w_mp_np = w_mp.squeeze(0).cpu().numpy()
    mp_feat_np = mp_feat.squeeze(0).cpu().numpy()
    m_out_np = m_out.squeeze(0).cpu().numpy()

    w_norm = np.linalg.norm(w_np, axis=-1)
    z_norm = np.linalg.norm(z_np, axis=-1)
    z_lift_norm = np.linalg.norm(z_lift_np, axis=-1) if z_lift_np.shape[-1] > 0 else np.zeros_like(w_norm)
    w_mp_norm = np.linalg.norm(w_mp_np, axis=-1)
    mp_feat_norm = np.linalg.norm(mp_feat_np, axis=-1)
    m_out_norm = np.linalg.norm(m_out_np, axis=-1)

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 13.2), sharex=True)

    axes[0].plot(time, w_norm, color="#1f77b4", lw=2.2, label=r"$\|w_t\|_2$")
    axes[0].plot(time, z_norm, color="#6b7280", lw=2.0, linestyle="--", label=r"$\|z_t\|_2$")
    axes[0].set_title("Decaying source signals", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("source norm")
    axes[0].legend(loc="upper right")

    axes[1].plot(time, z_lift_norm, color="#d97706", lw=2.2, label=r"$\|\zeta_t\|_2$")
    axes[1].plot(time, w_mp_norm, color="#0f766e", lw=2.3, label=r"$\|[w_t,\zeta_t]\|_2$")
    axes[1].set_title("Lifted context and concatenated M_p input", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("lifted / M input norm")
    axes[1].legend(loc="upper right")

    axes[2].plot(time, mp_feat_norm, color="#8b5cf6", lw=2.2, label=r"$\|M_p([w_t,\zeta_t])\|_2$")
    axes[2].plot(time, m_out_norm, color="#dc2626", lw=2.3, label=r"$\|M(w_t,z_t)\|_2$")
    axes[2].set_title("SSM feature response and overall operator output", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("operator norm")
    axes[2].legend(loc="upper right")

    axes[3].plot(time, w_np[:, 0], color="#1f77b4", lw=1.9, label=r"$w_t^{(1)}$")
    if z_lift_np.shape[-1] > 0:
        axes[3].plot(time, z_lift_np[:, 0], color="#d97706", lw=1.9, label=r"$\zeta_t^{(1)}$")
    axes[3].plot(time, mp_feat_np[:, 0], color="#8b5cf6", lw=2.0, label=r"$M_p^{(1)}$")
    axes[3].plot(time, m_out_np[:, 0], color="#dc2626", lw=2.0, label=r"$M^{(1)}$")
    if m_out_np.shape[-1] > 1:
        axes[3].plot(time, m_out_np[:, 1], color="#b91c1c", lw=1.8, linestyle="--", label=r"$M^{(2)}$")
    axes[3].set_title("Representative channels", fontsize=13, fontweight="bold")
    axes[3].set_xlabel("time step")
    axes[3].set_ylabel("channel value")
    axes[3].legend(loc="upper right", ncol=2)

    verdict = "PASS" if metrics["passed"] > 0.5 else "FAIL"
    fig.suptitle(
        "Decaying inputs drive the overall operator output to zero "
        f"| operator tail ratio = {metrics['operator_tail_ratio']:.4f} ({verdict})",
        y=0.995,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(run_dir / "operator_decay_test.png", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def build_operator_args(args: argparse.Namespace) -> argparse.Namespace:
    exp_args = controlled_xy.parse_args([])
    override_names = [
        "feat_dim",
        "mb_hidden",
        "mb_layers",
        "z_residual_gain",
        "mb_bound",
        "ssm_param",
        "ssm_layers",
        "ssm_d_model",
        "ssm_d_state",
        "ssm_ff",
        "mp_context_lift",
        "mp_context_lift_type",
        "mp_context_lift_dim",
        "mp_context_hidden_dim",
        "mp_context_decay_law",
        "mp_context_decay_rate",
        "mp_context_decay_power",
        "mp_context_decay_horizon",
        "mp_context_lp_p",
        "mp_context_scale",
    ]
    for name in override_names:
        setattr(exp_args, name, getattr(args, name))
    return exp_args


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_id or f"mp_input_decay_test_{timestamp}"
    run_dir = Path(__file__).resolve().parent / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    w_seq = make_decaying_sequence(
        horizon=int(args.horizon),
        dim=int(args.w_dim),
        scale=float(args.w_scale),
        decay_power=float(args.w_decay_power),
        freq0=0.20,
        device=device,
    )
    z_seq = make_decaying_sequence(
        horizon=int(args.horizon),
        dim=int(args.z_dim),
        scale=float(args.z_scale),
        decay_power=float(args.z_decay_power),
        freq0=0.11,
        device=device,
    )

    operator_args = build_operator_args(args)
    controller, _ = controlled_xy.build_controller(device, operator_args)
    operator = controller.operator
    with torch.no_grad():
        operator.reset()
        m_out_direct = operator(w_seq, z_seq)

        operator.reset()
        lifter = getattr(operator, "mp_context_lifter", None)
        if lifter is not None:
            z_lift = lifter(z_seq)
            w_mp = torch.cat([w_seq, z_lift], dim=-1)
        else:
            z_lift = torch.zeros(w_seq.shape[0], w_seq.shape[1], 0, device=device, dtype=w_seq.dtype)
            w_mp = w_seq
        mp_feat = operator.mp(w_mp)
        mb_map = operator.mb(w_seq, z_seq)
        m_out = operator.product_fn(mb_map, mp_feat)

    tail = max(4, int(args.tail_window))
    head_slice = slice(0, tail)
    tail_slice = slice(-tail, None)

    w_norm = torch.linalg.vector_norm(w_seq, dim=-1).squeeze(0)
    z_lift_norm = (
        torch.linalg.vector_norm(z_lift, dim=-1).squeeze(0)
        if z_lift.shape[-1] > 0
        else torch.zeros(int(args.horizon), device=device, dtype=w_seq.dtype)
    )
    w_mp_norm = torch.linalg.vector_norm(w_mp, dim=-1).squeeze(0)
    mp_feat_norm = torch.linalg.vector_norm(mp_feat, dim=-1).squeeze(0)
    m_out_norm = torch.linalg.vector_norm(m_out, dim=-1).squeeze(0)

    head_w_mp = float(w_mp_norm[head_slice].mean().item())
    tail_w_mp = float(w_mp_norm[tail_slice].mean().item())
    head_z_lift = float(z_lift_norm[head_slice].mean().item())
    tail_z_lift = float(z_lift_norm[tail_slice].mean().item())
    head_mp_feat = float(mp_feat_norm[head_slice].mean().item())
    tail_mp_feat = float(mp_feat_norm[tail_slice].mean().item())
    head_operator = float(m_out_norm[head_slice].mean().item())
    tail_operator = float(m_out_norm[tail_slice].mean().item())

    w_mp_tail_ratio = tail_w_mp / max(head_w_mp, 1e-8)
    z_lift_tail_ratio = tail_z_lift / max(head_z_lift, 1e-8)
    mp_feat_tail_ratio = tail_mp_feat / max(head_mp_feat, 1e-8)
    operator_tail_ratio = tail_operator / max(head_operator, 1e-8)
    direct_gap = float(torch.max(torch.abs(m_out_direct - m_out)).item())

    metrics = {
        "head_window": int(tail),
        "tail_window": int(tail),
        "w_norm_head_mean": float(w_norm[head_slice].mean().item()),
        "w_norm_tail_mean": float(w_norm[tail_slice].mean().item()),
        "z_lift_norm_head_mean": head_z_lift,
        "z_lift_norm_tail_mean": tail_z_lift,
        "w_mp_norm_head_mean": head_w_mp,
        "w_mp_norm_tail_mean": tail_w_mp,
        "mp_feat_norm_head_mean": head_mp_feat,
        "mp_feat_norm_tail_mean": tail_mp_feat,
        "operator_norm_head_mean": head_operator,
        "operator_norm_tail_mean": tail_operator,
        "z_lift_tail_ratio": z_lift_tail_ratio,
        "w_mp_tail_ratio": w_mp_tail_ratio,
        "mp_feat_tail_ratio": mp_feat_tail_ratio,
        "operator_tail_ratio": operator_tail_ratio,
        "manual_vs_direct_gap": direct_gap,
        "tail_ratio_threshold": float(args.tail_ratio_threshold),
        "passed": float(operator_tail_ratio <= float(args.tail_ratio_threshold)),
    }

    plot_diagnostic(
        args=args,
        run_dir=run_dir,
        w_seq=w_seq,
        z_seq=z_seq,
        z_lift=z_lift,
        w_mp=w_mp,
        mp_feat=mp_feat,
        m_out=m_out,
        metrics=metrics,
    )

    payload = dict(vars(args))
    payload["metrics"] = metrics
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Run directory: {run_dir}")
    print(f"w_mp tail ratio:   {w_mp_tail_ratio:.6f}")
    print(f"mp feat tail ratio:{mp_feat_tail_ratio:.6f}")
    print(f"operator tail ratio:{operator_tail_ratio:.6f}")
    print(f"z_lift tail ratio: {z_lift_tail_ratio:.6f}")
    print(f"manual/direct gap: {direct_gap:.6e}")
    print(f"Threshold:         {float(args.tail_ratio_threshold):.6f}")
    print(f"Pass:              {bool(metrics['passed'])}")

    if not bool(metrics["passed"]):
        raise SystemExit("Operator diagnostic failed: the tail of M(w,z) did not become small enough.")


if __name__ == "__main__":
    main()
