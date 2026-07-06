"""2D contextual gate PB+SSM experiment with controlled x/y motion toward the origin.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

from bounded_mlp_operator import BoundedMLPOperator
from context_lifting import LpContextLifter
from nav_plants import DoubleIntegratorNominal, DoubleIntegratorTrue
from pb_core import PBController, as_bt, rollout_pb, validate_component_compatibility, WIntegralAugmenter
from pb_core.factories import build_factorized_controller
from ssm_operators import MpDeepSSM, MpContextualSSM


@dataclass
class ScenarioBatch:
    start: torch.Tensor
    goal: torch.Tensor
    gate_y: torch.Tensor
    gate_v: torch.Tensor
    gate_ema: torch.Tensor
    gate_slow_ema: torch.Tensor
    switch_age: torch.Tensor
    process_noise: torch.Tensor
    pair_id: torch.Tensor
    is_adversarial: torch.Tensor  # bool (B,): True if episode has a late adversarial switch

    def to(self, device: torch.device) -> "ScenarioBatch":
        non_blocking = device.type == "cuda"
        return ScenarioBatch(
            start=self.start.to(device, non_blocking=non_blocking),
            goal=self.goal.to(device, non_blocking=non_blocking),
            gate_y=self.gate_y.to(device, non_blocking=non_blocking),
            gate_v=self.gate_v.to(device, non_blocking=non_blocking),
            gate_ema=self.gate_ema.to(device, non_blocking=non_blocking),
            gate_slow_ema=self.gate_slow_ema.to(device, non_blocking=non_blocking),
            switch_age=self.switch_age.to(device, non_blocking=non_blocking),
            process_noise=self.process_noise.to(device, non_blocking=non_blocking),
            pair_id=self.pair_id.to(device, non_blocking=non_blocking),
            is_adversarial=self.is_adversarial.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "ScenarioBatch":
        """Pin CPU tensors so CUDA copies can overlap with host work."""
        return ScenarioBatch(**{
            name: getattr(self, name).pin_memory()
            for name in self.__dataclass_fields__
        })


@dataclass
class RolloutArtifacts:
    x_seq: torch.Tensor
    u_seq: torch.Tensor
    w_seq: torch.Tensor


_PLT = None


def get_plt(show_plots: bool):
    global _PLT
    if _PLT is None:
        import matplotlib

        if not show_plots:
            matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        _PLT = plt
    return _PLT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("2D contextual PB gate experiment with PBController + SSM")
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tf32", dest="tf32", action="store_true",
                        help="On CUDA: allow TF32 matmul/cudnn kernels (faster on "
                             "Ampere+ GPUs, negligible precision loss). Default on; "
                             "ignored on CPU.")
    parser.add_argument("--no_tf32", dest="tf32", action="store_false",
                        help="Disable TF32; use full FP32 matmuls on CUDA.")
    parser.set_defaults(tf32=True)
    parser.add_argument("--cuda_amp", dest="cuda_amp", action="store_true",
                        help="Use CUDA BF16 autocast for rollout/loss (default on).")
    parser.add_argument("--no_cuda_amp", dest="cuda_amp", action="store_false",
                        help="Keep CUDA training/evaluation in FP32.")
    parser.set_defaults(cuda_amp=True)
    parser.add_argument("--prefetch_batches", dest="prefetch_batches", action="store_true",
                        help="Generate the next training batch concurrently with training.")
    parser.add_argument("--no_prefetch_batches", dest="prefetch_batches", action="store_false")
    parser.set_defaults(prefetch_batches=True)
    parser.add_argument("--torch_compile", action="store_true",
                        help="EXPERIMENTAL: wrap the PB operator's per-step forward in "
                             "torch.compile(dynamic=True) during training to fuse the many "
                             "tiny kernels of the sequential rollout. Falls back to eager "
                             "on any compilation error. Expect several slow warm-up "
                             "epochs; benefit depends heavily on the torch version "
                             "(cluster image ships 2.1) — A/B the epoch times before "
                             "trusting it.")
    parser.add_argument("--require_cuda", action="store_true",
                        help="Fail fast when --device cuda is requested but CUDA is "
                             "unavailable, instead of silently training on CPU "
                             "(recommended for cluster jobs).")
    parser.add_argument("--skip_plots", action="store_true",
                        help="Skip all figure/GIF generation at the end of the run. "
                             "Metrics and checkpoints are still saved incrementally; "
                             "regenerate figures later with --plot_only <run_id> "
                             "(used by per-variant cluster jobs).")
    parser.add_argument("--fresh", action="store_true",
                        help="Retrain every variant even when its checkpoint and metrics "
                             "already exist in the run directory. Default behavior "
                             "resumes: completed variants are loaded and skipped "
                             "(preemption/restart safety on the cluster).")
    parser.add_argument("--train_batch", type=int, default=512)
    parser.add_argument("--val_batch", type=int, default=512)
    parser.add_argument("--test_batch", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--disturbance_only_epochs", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lr_min", type=float, default=2e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--plot_only", type=str, default="",
                        help="Path to an existing run directory. Loads saved controller weights "
                             "and config, skips training, and re-runs evaluation + all plots.")
    # Custom-rollout playground: replay a trained run from an arbitrary start.
    parser.add_argument("--custom_rollout", type=str, default="",
                        help="Path/ID of a trained run directory: load its config + saved "
                             "controllers, roll them out from (--custom_start_x, "
                             "--custom_start_y) on a fresh scenario (--custom_seed), and "
                             "save a trajectory GIF under <run>/custom/. Skips training.")
    parser.add_argument("--custom_start_x", type=float, default=None,
                        help="custom_rollout: start x position.")
    parser.add_argument("--custom_start_y", type=float, default=None,
                        help="custom_rollout: start y position.")
    parser.add_argument("--custom_seed", type=int, default=123,
                        help="custom_rollout: scenario seed (gate motion + gusts).")
    parser.add_argument("--custom_variants", type=str, default="",
                        help="custom_rollout: comma-separated variants to overlay "
                             "(default: every *_controller.pt in the run, plus nominal).")
    parser.add_argument("--no_show_plots", action="store_true")
    parser.add_argument("--warm_start", dest="warm_start", action="store_true",
                        help="Warm-start the context variant from the disturbance_only checkpoint.")
    parser.add_argument("--no_warm_start", dest="warm_start", action="store_false")
    parser.set_defaults(warm_start=False)

    # Geometry and dynamics.
    parser.add_argument("--horizon", type=int, default=160)
    parser.add_argument("--plot_horizon", type=int, default=270,
                        help="Extended horizon for plots only. After --horizon steps the PB "
                             "correction is set to zero and the nominal plant is simulated forward. "
                             "Defaults to --horizon (no extension).")
    parser.add_argument("--use_plot_horizon", dest="use_plot_horizon", action="store_true")
    parser.add_argument("--no_plot_horizon", dest="use_plot_horizon", action="store_false")
    parser.set_defaults(use_plot_horizon=True)
    parser.add_argument("--lift_comparison", dest="lift_comparison", action="store_true",
                        help="Add context_no_lift variant: factorized M_b x M_p without "
                             "lifting on M_p, paired against the default mp_only_context "
                             "(M_p-only with lift) to isolate the effect of lifting.")
    parser.add_argument("--no_lift_comparison", dest="lift_comparison", action="store_false")
    parser.set_defaults(lift_comparison=False)
    parser.add_argument("--mad_comparison", dest="mad_comparison", action="store_true",
                        help="Add the MAD special case: factorized context operator with s=1, "
                             "so M_p outputs a scalar magnitude and M_b is a bounded 2x1 mixer.")
    parser.add_argument("--no_mad_comparison", dest="mad_comparison", action="store_false")
    parser.set_defaults(mad_comparison=True)
    parser.add_argument("--use_storyboard", dest="use_storyboard", action="store_true")
    parser.add_argument("--no_storyboard", dest="use_storyboard", action="store_false")
    parser.set_defaults(use_storyboard=True)
    parser.add_argument("--use_storyboard_compact", dest="use_storyboard_compact", action="store_true")
    parser.add_argument("--no_storyboard_compact", dest="use_storyboard_compact", action="store_false")
    parser.set_defaults(use_storyboard_compact=True)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--pre_kp", type=float, default=0.32)
    parser.add_argument("--pre_kd", type=float, default=0.80)
    parser.add_argument("--drag_coeff", type=float, default=0.0,
                        help="Quadratic velocity-drag coefficient c in the plant: "
                             "acc -= c*||vel||*vel (added to BOTH nominal and true "
                             "dynamics). 0 = linear double integrator. ~0.5-2 is 'a bit'.")
    parser.add_argument("--start_x_min", type=float, default=1.7)
    parser.add_argument("--start_x_max", type=float, default=2.1)
    parser.add_argument("--start_y_max", type=float, default=0.40)
    parser.add_argument("--start_y_min", type=float, default=None,
                        help="Start y lower bound (blank = -start_y_max, i.e. the legacy "
                             "symmetric band). Set both bounds for an asymmetric band, "
                             "e.g. 0.5/1.0 to start strictly above the gate's home region. "
                             "Keep |y| below --corridor_limit.")
    # Generalization over initial positions: widen the training box via the
    # start_* flags above, and/or evaluate out-of-distribution by giving the
    # TEST batch its own start ranges (validation stays on the training
    # distribution so model selection is unaffected).
    parser.add_argument("--test_start_x_min", type=float, default=None,
                        help="TEST-batch start x lower bound (blank = same as training). "
                             "Use with the other test_start_* flags to measure "
                             "generalization to initial positions never seen in training.")
    parser.add_argument("--test_start_x_max", type=float, default=None,
                        help="TEST-batch start x upper bound (blank = same as training). "
                             "Keep the implied wall-crossing time inside --horizon.")
    parser.add_argument("--test_start_y_max", type=float, default=None,
                        help="TEST-batch start y upper bound (blank = same as training).")
    parser.add_argument("--test_start_y_min", type=float, default=None,
                        help="TEST-batch start y lower bound (blank = same as training, "
                             "which itself defaults to -start_y_max).")
    parser.add_argument("--freeze_per_episode", action="store_true",
                        help="switch only: freeze each episode's gate before that episode's "
                             "OWN predicted wall-crossing time (from its start x) instead of "
                             "one global freeze at the mean start's crossing. Recommended "
                             "whenever the start x range is wide.")
    parser.add_argument("--wall_x", type=float, default=0.55)
    parser.add_argument("--gate_half_width", type=float, default=0.20)
    parser.add_argument("--gate_amplitude", type=float, default=0.95)
    parser.add_argument("--goal_tol", type=float, default=0.18)
    parser.add_argument("--corridor_limit", type=float, default=1.6)
    parser.add_argument("--wall_focus_sigma", type=float, default=0.14)
    parser.add_argument("--gate_settle_steps", type=int, default=2)

    # Gate motion: 'switch' (original piecewise-constant schedule + pre-crossing
    # freeze) or 'continuous' (Ornstein-Uhlenbeck drift that never settles).
    parser.add_argument("--gate_motion", type=str, default="switch",
                        choices=["switch", "continuous"],
                        help="Gate-center dynamics. 'switch' = original schedule with a "
                             "pre-crossing freeze; 'continuous' = OU drift (no freeze, no "
                             "discrete switches) so the controller must infer motion from "
                             "the observation history.")
    parser.add_argument("--gate_corr_time", type=float, default=40.0,
                        help="continuous only: gate-motion smoothness as an OU correlation "
                             "time in steps. LARGER = slower, smoother drift; smaller = "
                             "faster, jerkier (theta = 1 - exp(-1/corr_time)).")
    parser.add_argument("--gate_range", type=float, default=0.50,
                        help="continuous only: within-episode roaming amplitude (OU "
                             "stationary std), same units as the gate center, clamped to "
                             "+/- gate_amplitude. Independent of --gate_corr_time.")
    parser.add_argument("--gate_center_range", type=float, default=0.35,
                        help="continuous only: per-episode random reversion center, as a "
                             "fraction of gate_amplitude. 0 = every episode centered on the "
                             "wall middle (gate hugs center); larger = different episodes "
                             "settle around different heights, so the gate covers more "
                             "vertical space and the controller can't just park at center.")
    parser.add_argument("--gate_margin_train", type=float, default=0.04,
                        help="continuous only: safety margin subtracted from "
                             "--gate_half_width inside the training collision loss. "
                             "Evaluation still uses the real gate width; set 0 to disable.")

    # Gate schedule.
    parser.add_argument("--gate_dwell_min", type=int, default=6)
    parser.add_argument("--gate_dwell_max", type=int, default=16)
    parser.add_argument("--context_ema_alpha", type=float, default=0.35)
    parser.add_argument("--gate_obs_delay", type=int, default=0,
                        help="Steps of delay on gate observations (0=no delay).")
    parser.add_argument("--context_dropout_p", type=float, default=0.0,
                        help="Probability of zeroing gate context features during training.")

    # Disturbance process.
    parser.add_argument("--noise_pos_sigma", type=float, default=3e-4)
    parser.add_argument("--noise_vel_sigma", type=float, default=1.2e-3)
    parser.add_argument("--gust_count_min", type=int, default=2)
    parser.add_argument("--gust_count_max", type=int, default=4)
    parser.add_argument("--gust_duration_min", type=int, default=4)
    parser.add_argument("--gust_duration_max", type=int, default=10)
    parser.add_argument("--gust_vel_y_min", type=float, default=0.010)
    parser.add_argument("--gust_vel_y_max", type=float, default=0.028)
    parser.add_argument("--gust_vel_x_max", type=float, default=0.004)
    parser.add_argument("--gust_clip_y", type=float, default=0.045)

    # Process-noise decay: fade the injected noise (hence the reconstructed
    # disturbance w_t) toward 0 over the horizon, so w is an L2 / transient signal
    # rather than a stationary floor. The IC term w_0 = x_0 is left untouched.
    parser.add_argument("--noise_decay", type=str, default="taper",
                        choices=["none", "taper", "linear", "exponential"],
                        help="Decay window on the process noise over the horizon (w_t -> 0 by "
                             "t=T). 'none' = stationary noise (legacy); 'taper' (default) keeps "
                             "noise flat then cosine-fades the trailing --noise_decay_ramp steps.")
    parser.add_argument("--noise_decay_ramp", type=int, default=0,
                        help="taper only: trailing cosine roll-off length in steps "
                             "(0 = auto = horizon//2).")
    parser.add_argument("--noise_decay_rate", type=float, default=0.98,
                        help="exponential only: per-step decay factor (noise scaled by rate**t).")

    # PB architecture.
    parser.add_argument("--feat_dim", type=int, default=16)
    parser.add_argument("--mb_hidden", type=int, default=64)
    parser.add_argument("--mb_layers", type=int, default=4)
    parser.add_argument("--z_scale", type=float, default=6.0)
    parser.add_argument("--z_residual_gain", type=float, default=10.0)
    parser.add_argument("--mb_bound", type=float, default=8.0)
    parser.add_argument("--ssm_param", type=str, default="lru", choices=["lru", "tv", "tvc"],
                        help="SSM core parametrization: 'lru' (default), 'tv' (time-varying "
                             "selective), or 'tvc' (time-varying selective with a richer "
                             "parametrization: MLP selector, signed transitions + feedthrough). "
                             "The contextual 'select' port conditions the selective dynamics on "
                             "context with EITHER tv or tvc (lru is rejected).")
    parser.add_argument("--ssm_bcd_nonlinearity", type=str, default="tanh",
                        choices=["tanh", "identity"],
                        help="tvc only: nonlinearity bounding b,c,d before normalization. "
                             "'tanh' (default) trains more stably; 'identity' is the legacy "
                             "unbounded behavior. Ignored by lru/tv.")
    parser.add_argument("--ssm_layers", type=int, default=4)
    parser.add_argument("--ssm_d_model", type=int, default=32)
    parser.add_argument("--ssm_d_state", type=int, default=64)
    parser.add_argument("--ssm_ff", type=str, default="GLU")
    # M_p-only variant SSM sizing (v3).
    # When None, d_model / n_layers are auto-matched to the factorized budget.
    parser.add_argument("--mp_only_ssm_d_model", type=int, default=None,
                        help="SSM d_model for the M_p-only variant. "
                             "Defaults to auto-matched value that equalises total params.")
    parser.add_argument("--mp_only_ssm_layers", type=int, default=None,
                        help="SSM n_layers for the M_p-only variant. "
                             "Defaults to --ssm_layers (same as factorized variant).")

    # Contextual SSM variant (neural_ssm.ContextualDeepSSM from Clean_SSM / neural-ssm >= 0.4).
    # A single context-native operator (M_p and M_b fused) that injects context through
    # the mixer (router) / input (driver) / gate ports.  Added as an extra variant
    # alongside the hand-rolled factorized M_b x M_p controllers.
    parser.add_argument("--contextual_comparison", dest="contextual_comparison", action="store_true",
                        help="Include the ContextualDeepSSM variant (default on).")
    parser.add_argument("--no_contextual_comparison", dest="contextual_comparison", action="store_false")
    parser.set_defaults(contextual_comparison=True)
    parser.add_argument("--match_to_contextual", dest="match_to_contextual", action="store_true",
                        help="When contextual_ssm is among the variants, size every OTHER "
                             "trainable variant's SSM d_model so its total parameter count "
                             "matches contextual_ssm as closely as possible (fair comparison). "
                             "Layers stay at --ssm_layers; only d_model is adjusted.")
    parser.add_argument("--no_match_to_contextual", dest="match_to_contextual", action="store_false")
    parser.set_defaults(match_to_contextual=False)
    parser.add_argument("--ctx_modes", type=str, default="mixer,input,gate",
                        help="Comma-separated context ports for ContextualDeepSSM: any of "
                             "mixer,input,gate,select.")
    parser.add_argument("--ctx_filter", type=str, default="taper",
                        choices=["auto", "finite_horizon", "taper", "exponential", "polynomial",
                                 "difference", "none"],
                        help="L2 projection for the 'input' (driver) port. Time-windowed filters "
                             "(taper/finite_horizon/exponential/polynomial) are correct in the "
                             "step-by-step closed loop; difference/none are not recommended there.")
    parser.add_argument("--ctx_filter_ramp", type=int, default=20,
                        help="Cosine roll-off length for the 'taper' filter (1..horizon). "
                             "0 -> defaults to the full horizon.")
    parser.add_argument("--ctx_mixer_bound", type=float, default=4.0,
                        help="Spectral-norm bound on the mixer matrix A_t (router gain cap).")
    parser.add_argument("--ctx_d_features", type=int, default=16,
                        help="Core feature width fed to the mixer (analogous to feat_dim).")
    parser.add_argument("--ctx_gamma", type=float, default=0.0,
                        help="Prescribed L2 gain cap for the ContextualDeepSSM core. "
                             "0 -> no cap (free finite gain, matches the other SSM variants).")
    parser.add_argument("--ctx_gate_per_channel", dest="ctx_gate_per_channel", action="store_true",
                        help="Per-channel (vs scalar) sigmoid gates in the 'gate' port.")
    parser.set_defaults(ctx_gate_per_channel=False)
    parser.add_argument("--ctx_select", dest="ctx_select", action="store_true",
                        help="Enable the contextual 'select' port: inject context into the "
                             "selective SSM matrices (A/B/C[/D]) via the cell param_net. "
                             "Requires --ssm_param tv or tvc; gain-safe (no L2 projection). "
                             "Adds 'select' to --ctx_modes if not already present.")
    parser.set_defaults(ctx_select=False)

    # Explicit variant selection. When non-empty, runs EXACTLY these variants
    # (comma-separated, in order) and overrides the *_comparison toggles below.
    # Example: --variants contextual_ssm   (run only the new ContextualDeepSSM variant)
    parser.add_argument("--variants", type=str, default="",
                        help="Comma-separated subset of variants to run, e.g. "
                             "'contextual_ssm' or 'context,contextual_ssm'. Choices: "
                             "nominal, disturbance_only, context, mad_context, "
                             "mp_only_context, context_no_lift, contextual_ssm. "
                             "Empty = use the *_comparison flags.")

    # Context mode (v3): full (11-D) or minimal (3-D: gate error, approach, switch age).
    parser.add_argument("--simple_comparison", dest="simple_comparison", action="store_true",
                        help="Only run two variants: disturbance-only M_p (no context) "
                             "vs. full factorized M_b⊠M_p with context (+ lifting if enabled). "
                             "Skips nominal-only and matched-param M_p-only+context variants.")
    parser.add_argument("--no_simple_comparison", dest="simple_comparison", action="store_false")
    parser.set_defaults(simple_comparison=True)
    parser.add_argument("--context_mode", type=str, default="minimal",
                        choices=["full", "minimal"],
                        help="Context feature set fed to the PB operator. "
                             "'full' uses all 11 features; 'minimal' uses only "
                             "[gate_error_t, approach_t, switch_age_t] (3-D). "
                             "Superseded by --context_features when that is non-empty.")
    parser.add_argument("--context_features", type=str, default="",
                        help="Explicit comma-separated context features, e.g. "
                             "'gate_obs,gate_error,approach'. Overrides --context_mode when "
                             "non-empty; emitted in a fixed canonical order. Choices: "
                             + ",".join(CONTEXT_FEATURE_ORDER) + ". 'Fair' features "
                             "(observation + own state) vs 'privileged' (gate dynamics / "
                             "schedule = cheating) are documented in CONTEXT_FEATURE_META.")

    # Context ablation: hold ONE architecture fixed and sweep context subsets to
    # find the decisive context component (overrides the normal multi-variant run).
    parser.add_argument("--ablate_context", action="store_true",
                        help="Run a context-feature ablation: train --ablation_variant once "
                             "per config in --ablation_configs (same data/seeds) and compare.")
    parser.add_argument("--ablation_variant", type=str, default="context",
                        choices=["context", "mad_context", "mp_only_context", "contextual_ssm"],
                        help="Single architecture held fixed across the ablation.")
    parser.add_argument("--ablation_configs", type=str, default="",
                        help="Semicolon-separated context configs, each 'label:feat1,feat2' "
                             "(label optional). Empty = leave-one-out over the fair default "
                             "set. Example: 'full:gate_obs,gate_error,approach;"
                             "-gate_obs:gate_error,approach'.")
    parser.add_argument("--ablate_layers", action="store_true",
                        help="Run an SSM-depth comparison: train --ablation_variant with the "
                             "SAME context at each --ablation_layers depth (same data/seeds).")
    parser.add_argument("--ablation_layers", type=str, default="",
                        help="Comma-separated SSM layer counts to compare, e.g. '3,6,9,12'. "
                             "Empty = sweep around --ssm_layers (base//2, base, base*2).")
    parser.add_argument("--mp_context_lift", dest="mp_context_lift", action="store_true")
    parser.add_argument("--no_mp_context_lift", dest="mp_context_lift", action="store_false")
    parser.set_defaults(mp_context_lift=True)
    parser.add_argument("--mp_context_lift_type", type=str, default="linear", choices=["identity", "linear", "mlp"])
    parser.add_argument("--mp_context_lift_dim", type=int, default=6)
    parser.add_argument("--mp_context_hidden_dim", type=int, default=24)
    parser.add_argument("--mp_context_decay_law", type=str, default="finite", choices=["exp", "poly", "finite"])
    parser.add_argument("--mp_context_decay_rate", type=float, default=0.04)
    parser.add_argument("--mp_context_decay_power", type=float, default=0.73)
    parser.add_argument("--mp_context_decay_horizon", type=int, default=140)
    parser.add_argument("--mp_context_lp_p", type=float, default=2.0)
    parser.add_argument("--mp_context_scale", type=float, default=0.25)

    parser.add_argument("--w0_clip", type=float, default=0.15,
                        help="Clip value for w_0=x_0 fed to the operator at t=0.")
    parser.add_argument("--use_w0_clip", dest="use_w0_clip", action="store_true",
                        help="Enable w_0 clipping (default on).")
    parser.add_argument("--no_w0_clip", dest="use_w0_clip", action="store_false",
                        help="Disable w_0 clipping.")
    parser.set_defaults(use_w0_clip=False)
    parser.add_argument("--w_augment_decay", type=float, default=0.97,
                        help="Decay rate γ for the leaky-integral w augmentation. "
                             "Operator input becomes [w_t, γ^t*w̄_0 + ...] (dim doubled).")
    parser.add_argument("--use_w_augment", dest="use_w_augment", action="store_true",
                        help="Augment w with its causal leaky integral before feeding the operator.")
    parser.add_argument("--no_w_augment", dest="use_w_augment", action="store_false")
    parser.set_defaults(use_w_augment=False)

    # Loss.
    parser.add_argument("--goal_stage_weight", type=float, default=5.2)
    parser.add_argument("--goal_terminal_weight", type=float, default=64.0)
    parser.add_argument("--wall_track_weight", type=float, default=24.0)
    parser.add_argument("--wall_collision_weight", type=float, default=120.0)
    parser.add_argument("--control_weight", type=float, default=6.05)
    parser.add_argument("--corridor_weight", type=float, default=10.0)
    parser.add_argument("--collision_sharpness", type=float, default=14.0)
    parser.add_argument("--corridor_sharpness", type=float, default=12.0)
    parser.add_argument("--terminal_vel_weight", type=float, default=4.0,
                        help="Weight on ||v_T||^2 terminal velocity penalty.")
    parser.add_argument("--use_terminal_vel", dest="use_terminal_vel", action="store_true")
    parser.add_argument("--no_terminal_vel", dest="use_terminal_vel", action="store_false")
    parser.set_defaults(use_terminal_vel=True)
    parser.add_argument("--sample_traj_count", type=int, default=4)
    # Overshoot control: deceleration-near-goal penalty (velocity weighted by a
    # Gaussian in distance-to-goal). Raw-mode weight + radius live here with the
    # other raw weights; its normalized-mode priority is alpha_settle_vel below.
    parser.add_argument("--settle_vel_weight", type=float, default=0.0,
                        help="Raw-mode weight on the deceleration-near-goal penalty: speed "
                             "weighted by a Gaussian in distance-to-goal. 0 = off (default). "
                             "Raise to curb overshoot at the target.")
    parser.add_argument("--settle_sigma", type=float, default=0.35,
                        help="Radius (std, in position units) of the near-goal region for the "
                             "deceleration penalty; smaller = only very close to the goal.")

    # Loss normalization: scale each term by its value on the nominal (no-boost)
    # rollout, then weight by O(1) --alpha_* priorities (scale-free, regime-invariant).
    parser.add_argument("--loss_normalize", dest="loss_normalize", action="store_true",
                        help="Normalize each loss term by its nominal-rollout scale, then "
                             "weight by --alpha_* (default 1 = equal). Decouples scale from "
                             "priority and is invariant to the gate regime. Off (default) = "
                             "legacy raw *_weight behavior.")
    parser.add_argument("--no_loss_normalize", dest="loss_normalize", action="store_false")
    parser.set_defaults(loss_normalize=False)
    parser.add_argument("--alpha_goal_stage", type=float, default=1.0,
                        help="--loss_normalize: priority on the goal-tracking stage cost.")
    parser.add_argument("--alpha_goal_term", type=float, default=1.0,
                        help="--loss_normalize: priority on the terminal goal-distance.")
    parser.add_argument("--alpha_terminal_vel", type=float, default=1.0,
                        help="--loss_normalize: priority on the terminal-velocity penalty.")
    parser.add_argument("--alpha_wall_track", type=float, default=1.0,
                        help="--loss_normalize: priority on the gate-tracking-near-wall cost.")
    parser.add_argument("--alpha_wall_collision", type=float, default=1.0,
                        help="--loss_normalize: priority on the wall-collision penalty.")
    parser.add_argument("--alpha_control", type=float, default=1.0,
                        help="--loss_normalize: priority on the control-effort cost.")
    parser.add_argument("--alpha_corridor", type=float, default=1.0,
                        help="--loss_normalize: priority on the corridor-limit penalty.")
    parser.add_argument("--alpha_settle_vel", type=float, default=0.0,
                        help="--loss_normalize: priority on the deceleration-near-goal "
                             "penalty. 0 = off (default).")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_python_float(value) -> float:
    return float(value.item() if torch.is_tensor(value) else value)


def tensor_scalars_to_floats(values: dict[str, torch.Tensor]) -> dict[str, float]:
    """Transfer multiple scalar tensors with a single device synchronization."""
    if not values:
        return {}
    names = list(values)
    packed = torch.stack([values[name].detach().float().reshape(()) for name in names])
    return dict(zip(names, packed.cpu().tolist()))


def maybe_compile_operator(controller: PBController, args: argparse.Namespace) -> None:
    """Opt-in torch.compile of the operator's per-step forward (--torch_compile).

    The closed-loop rollout calls ``controller.operator(w_t, z_t)`` once per
    timestep with tiny tensors, so the run is kernel-launch bound; compiling
    the operator fuses those launches. The forward METHOD is wrapped (not the
    module), so ``state_dict`` keys stay unprefixed and checkpoints remain
    compatible with resume / plot_only / custom rollouts. ``dynamic=True``
    keeps the changing step index / recurrent state from recompiling per step.
    Best-effort: any failure keeps eager mode.
    """
    if not bool(getattr(args, "torch_compile", False)):
        return
    op = controller.operator
    # PB_TORCH_COMPILE_BACKEND overrides the backend (e.g. aot_eager for
    # correctness checks on machines where inductor's C++ build is unusable).
    backend = os.environ.get("PB_TORCH_COMPILE_BACKEND", "inductor")
    try:
        # Step counters live as int attributes (e.g. MpContextualSSM._t fed to
        # the core as time_offset). By default dynamo bakes such ints into
        # guards -> one recompile per timestep until the cache limit, then
        # silent eager fallback. Treat them as dynamic instead.
        if hasattr(torch._dynamo.config, "allow_unspec_int_on_nn_module"):
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
        op.forward = torch.compile(op.forward, dynamic=True, backend=backend)
        print(f"[compile] torch.compile enabled on the PB operator (backend={backend}; "
              "first epochs include compilation warm-up).")
    except Exception as exc:  # pragma: no cover - depends on torch build
        print(f"[compile] torch.compile unavailable ({exc}); continuing in eager mode.")


def cuda_autocast(args: argparse.Namespace, device: torch.device):
    if (device.type == "cuda" and bool(getattr(args, "cuda_amp", True))
            and torch.cuda.is_bf16_supported()):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def iter_training_batches(
    *, args: argparse.Namespace, epochs: int, expected_cross_index: int,
    pin_memory: bool,
):
    """Yield deterministic epoch batches, optionally producing one ahead."""
    def make(epoch: int) -> ScenarioBatch:
        batch = sample_batch(
            args=args,
            batch_size=int(args.train_batch),
            seed=int(args.seed) + 1000 + epoch,
            paired=True,
            shuffle=True,
            expected_cross_index=expected_cross_index,
        )
        return batch.pin_memory() if pin_memory else batch

    # Prefetch also pays off on CPU-only runs: batch generation (numpy loops)
    # overlaps with the torch training step. Pinning stays CUDA-only.
    prefetch = bool(getattr(args, "prefetch_batches", True))
    if not prefetch:
        for epoch in range(1, epochs + 1):
            yield epoch, make(epoch)
        return

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="gate-batch") as pool:
        future = pool.submit(make, 1)
        for epoch in range(1, epochs + 1):
            batch = future.result()
            if epoch < epochs:
                future = pool.submit(make, epoch + 1)
            yield epoch, batch


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def find_matched_ssm_d_model(
    *,
    args: argparse.Namespace,
    target_params: int,
    mp_in_dim: int,
) -> int:
    """Return the smallest d_model (stepping by 4) such that
    MpDeepSSM(mp_in_dim, nu=2, d_model=d_model, ...) has >= target_params.

    Throwaway probe modules stay on CPU — parameter counts are device-independent
    and this avoids churning GPU memory/startup time on the cluster.
    """
    nu = 2
    n_layers = int(args.mp_only_ssm_layers or args.ssm_layers)
    start = int(args.ssm_d_model)
    for d_model in range(start, 1024, 4):
        mp = MpDeepSSM(
            mp_in_dim,
            nu,
            mode="loop",
            param=args.ssm_param,
            n_layers=n_layers,
            d_model=d_model,
            d_state=int(args.ssm_d_state),
            ff=args.ssm_ff,
            bcd_nonlinearity=args.ssm_bcd_nonlinearity,
        )
        if count_params(mp) >= target_params:
            return d_model
    raise RuntimeError(
        f"Cannot match param budget of {target_params} for M_p-only SSM "
        f"within d_model search range [{ start}, 1024)."
    )


def find_matched_d_model(
    *,
    args: argparse.Namespace,
    target_params: int,
    mp_only: bool,
    use_mad: bool,
    d_min: int = 4,
    d_max: int = 1024,
    step: int = 2,
) -> int:
    """SSM d_model whose FULL-controller param count is closest to target_params
    for the given variant shape (factorized / MAD / M_p-only).

    Param count is monotone in d_model, so we sweep and keep the value minimizing
    |count - target|. The whole controller is built each step, so M_b / lifter
    params are included in the match. Probe controllers are built on CPU:
    param counts are device-independent and this keeps the (potentially hundreds
    of) throwaway builds off the GPU.
    """
    probe_device = torch.device("cpu")
    best_d, best_err = d_min, None
    for d in range(d_min, d_max, step):
        try:
            ctrl, _ = build_controller(
                probe_device, args, mp_only=mp_only,
                factor_rank_override=1 if use_mad else None,
                ssm_d_model_override=d,
            )
        except Exception:
            continue
        c = count_params(ctrl)
        del ctrl
        err = abs(c - target_params)
        if best_err is None or err < best_err:
            best_d, best_err = d, err
        if c >= target_params:
            break
    return best_d


def contextual_label(args=None) -> str:
    modes = "mixer,input,gate"
    if args is not None:
        modes = str(getattr(args, "ctx_modes", modes))
        if getattr(args, "ctx_select", False) and "select" not in modes.split(","):
            modes = f"{modes},select"
    return f"PB+SSM: ContextualDeepSSM ({modes})"


# Canonical label for every supported variant (used by --variants selection).
ALL_VARIANTS: dict[str, str] = {
    "nominal": "Nominal only",
    "disturbance_only": "PB+SSM: no context",
    "context": "PB+SSM: factorized M_b x M_p",
    "mad_context": "PB+SSM: MAD s=1",
    "mp_only_context": "PB+SSM: M_p-only + lift (matched params)",
    "context_no_lift": "PB+SSM: factorized M_b x M_p (no lift)",
    "contextual_ssm": "PB+SSM: ContextualDeepSSM",
}


def variant_specs(args=None) -> list[tuple[str, str]]:
    # Explicit selection overrides all *_comparison toggles.
    selection = str(getattr(args, "variants", "") or "").strip() if args is not None else ""
    if selection:
        chosen = [m.strip() for m in selection.split(",") if m.strip()]
        unknown = [m for m in chosen if m not in ALL_VARIANTS]
        if unknown:
            raise ValueError(
                f"Unknown variant(s) {unknown}. Choose from {list(ALL_VARIANTS)}."
            )
        return [
            (m, contextual_label(args) if m == "contextual_ssm" else ALL_VARIANTS[m])
            for m in chosen
        ]

    lift_cmp = args is not None and getattr(args, "lift_comparison", False)
    mad_cmp = args is None or getattr(args, "mad_comparison", True)
    ctx_cmp = args is None or getattr(args, "contextual_comparison", True)
    if args is not None and getattr(args, "simple_comparison", False) and not lift_cmp:
        specs = [
            ("nominal", "Nominal only"),
            ("disturbance_only", "PB+SSM: no context"),
            ("context", "PB+SSM: factorized M_b x M_p"),
        ]
        if mad_cmp:
            specs.append(("mad_context", "PB+SSM: MAD s=1"))
        if ctx_cmp:
            specs.append(("contextual_ssm", contextual_label(args)))
        return specs
    specs = [
        ("nominal", "Nominal only"),
        ("disturbance_only", "PB+SSM: no context"),
        ("context", "PB+SSM: factorized M_b x M_p"),
    ]
    if mad_cmp:
        specs.append(("mad_context", "PB+SSM: MAD s=1"))
    specs.append(("mp_only_context", "PB+SSM: M_p-only + lift (matched params)"))
    if args is not None and getattr(args, "lift_comparison", False):
        specs.append(("context_no_lift", "PB+SSM: factorized M_b x M_p (no lift)"))
    if ctx_cmp:
        specs.append(("contextual_ssm", contextual_label(args)))
    return specs


# Canonical context-feature layout. This ORDER defines the context-vector layout
# and must stay fixed (operator input weights depend on it). New features are
# appended at the END so any previously used subset emits the exact same vector
# (and old checkpoints stay loadable); the first 11 match the original 'full' set.
CONTEXT_FEATURE_ORDER = [
    "gate_obs", "gate_vel", "gate_ema", "gate_slow_ema", "gate_error",
    "rel_wall_x", "goal_dx", "goal_dy", "approach", "switch_age", "time_to_wall",
    "vel_x", "vel_y",
]
# key -> (group, human label, experiments-it-applies-to). 'fair' = observation +
# own state + known geometry; 'privileged' = gate dynamics / schedule ("cheating").
CONTEXT_FEATURE_META = {
    "gate_obs":      ("fair",       "Observed gate center y",                ("switch", "continuous")),
    "gate_vel":      ("fair",       "Causal observed gate velocity",         ("switch", "continuous")),
    "gate_ema":      ("fair",       "Causal gate EMA (fast)",                ("switch", "continuous")),
    "gate_slow_ema": ("fair",       "Causal gate EMA (slow)",                ("switch", "continuous")),
    "gate_error":    ("fair",       "Gate error (y_robot - gate_obs)",       ("switch", "continuous")),
    "rel_wall_x":    ("fair",       "Distance to wall (x)",                  ("switch", "continuous")),
    "goal_dx":       ("fair",       "Goal offset x",                         ("switch", "continuous")),
    "goal_dy":       ("fair",       "Goal offset y",                         ("switch", "continuous")),
    "approach":      ("fair",       "Approach weight (near wall)",           ("switch", "continuous")),
    "switch_age":    ("privileged", "Steps since last gate switch",          ("switch",)),
    "time_to_wall":  ("privileged", "Normalized time-to-wall (schedule)",    ("switch", "continuous")),
    "vel_x":         ("fair",       "Own velocity vx",                       ("switch", "continuous")),
    "vel_y":         ("fair",       "Own velocity vy",                       ("switch", "continuous")),
}
# Default fair set for the continuous experiment: observation, causal gate-motion
# history, own state, and known geometry.
FAIR_CONTEXT_DEFAULT = [
    "gate_obs", "gate_vel", "gate_ema", "gate_error",
    "rel_wall_x", "goal_dx", "goal_dy", "approach",
]
# Legacy --context_mode 'full' stays the ORIGINAL 11 features (frozen so old
# runs/checkpoints keep their context_dim); opt into velocity via
# --context_features / the launcher checkboxes instead.
_FULL_FEATURES = [
    "gate_obs", "gate_vel", "gate_ema", "gate_slow_ema", "gate_error",
    "rel_wall_x", "goal_dx", "goal_dy", "approach", "switch_age", "time_to_wall",
]
_MINIMAL_FEATURES = ["gate_error", "approach", "switch_age"]  # legacy --context_mode minimal

FULL_CONTEXT_DIM = len(_FULL_FEATURES)
MINIMAL_CONTEXT_DIM = len(_MINIMAL_FEATURES)


def is_continuous_gate(args: argparse.Namespace | None) -> bool:
    return str(getattr(args, "gate_motion", "switch")).lower() == "continuous"


def resolve_context_features(args=None) -> list[str]:
    """Active context features in canonical order.

    --context_features (comma-separated) is authoritative when set; otherwise
    fall back to the legacy --context_mode (full/minimal). Continuous-gate
    runs default to the fair observation/state set instead of legacy
    switch_age-based minimal context.
    """
    raw = str(getattr(args, "context_features", "") or "").strip() if args is not None else ""
    if raw:
        sel = {f.strip() for f in raw.split(",") if f.strip()}
        unknown = sel - set(CONTEXT_FEATURE_ORDER)
        if unknown:
            raise ValueError(
                f"Unknown context feature(s) {sorted(unknown)}. "
                f"Choose from {CONTEXT_FEATURE_ORDER}."
            )
        return [k for k in CONTEXT_FEATURE_ORDER if k in sel]
    if args is not None and is_continuous_gate(args) and getattr(args, "context_mode", "full") == "minimal":
        return list(FAIR_CONTEXT_DEFAULT)
    if args is not None and getattr(args, "context_mode", "full") == "minimal":
        return list(_MINIMAL_FEATURES)
    return list(_FULL_FEATURES)


def context_dim(args=None) -> int:
    return len(resolve_context_features(args))


def epochs_for_mode(args: argparse.Namespace, mode: str) -> int:
    if mode == "disturbance_only":
        return max(1, int(args.disturbance_only_epochs))
    return max(1, int(args.epochs))


def build_gate_features(gates: np.ndarray, ema_alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gate_velocity = np.zeros_like(gates, dtype=np.float32)
    if gates.shape[1] > 1:
        gate_velocity[:, 1:] = gates[:, 1:] - gates[:, :-1]
        gate_velocity[:, 0] = 0.0

    gate_ema = np.zeros_like(gates, dtype=np.float32)
    gate_ema[:, 0] = gates[:, 0]
    for t in range(1, gates.shape[1]):
        gate_ema[:, t] = ema_alpha * gates[:, t] + (1.0 - ema_alpha) * gate_ema[:, t - 1]

    # Slow EMA (long-run gate trend, alpha=0.05 ≈ 20-step window)
    slow_alpha = 0.05
    gate_slow_ema = np.zeros_like(gates, dtype=np.float32)
    gate_slow_ema[:, 0] = gates[:, 0]
    for t in range(1, gates.shape[1]):
        gate_slow_ema[:, t] = slow_alpha * gates[:, t] + (1.0 - slow_alpha) * gate_slow_ema[:, t - 1]

    # Age since the most recent discrete change, vectorized over the batch and
    # horizon. At a change its age is zero; otherwise it increments by one.
    width = gates.shape[1]
    steps = np.arange(width, dtype=np.int64)[None, :]
    changed = np.zeros_like(gates, dtype=bool)
    changed[:, 1:] = np.abs(gates[:, 1:] - gates[:, :-1]) > 1e-6
    last_change = np.maximum.accumulate(np.where(changed, steps, 0), axis=1)
    switch_age = (steps - last_change).astype(np.float32)
    switch_age[:, 0] = 0.0
    return gate_velocity, gate_ema, gate_slow_ema, switch_age


def estimate_expected_cross_index(args: argparse.Namespace) -> int:
    plant = DoubleIntegratorTrue(dt=float(args.dt), pre_kp=float(args.pre_kp), pre_kd=float(args.pre_kd), drag_coeff=float(args.drag_coeff))
    start_x = 0.5 * (float(args.start_x_min) + float(args.start_x_max))
    x = torch.tensor([[[start_x, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    u = torch.zeros(1, 1, 2, dtype=torch.float32)
    xs = []
    for _ in range(int(args.horizon)):
        x = plant.forward(x, u)
        xs.append(float(x[0, 0, 0].item()))
    xs_np = np.asarray(xs, dtype=np.float32)
    crossed = np.where(xs_np <= float(args.wall_x))[0]
    if crossed.size:
        idx = int(crossed[0])
    else:
        idx = int(np.argmin(np.abs(xs_np - float(args.wall_x))))
    if idx <= 0 or idx >= int(args.horizon):
        raise ValueError("Expected wall crossing must lie inside the rollout horizon.")
    return idx


def resolve_start_ranges(args: argparse.Namespace, *, test: bool = False) -> tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) for start sampling; TEST overrides when given.

    y_min defaults to -y_max (legacy symmetric band) unless --start_y_min /
    --test_start_y_min are set.
    """
    x_min = float(args.start_x_min)
    x_max = float(args.start_x_max)
    y_max = float(args.start_y_max)
    y_min = (-y_max if getattr(args, "start_y_min", None) is None
             else float(args.start_y_min))
    if test:
        if getattr(args, "test_start_x_min", None) is not None:
            x_min = float(args.test_start_x_min)
        if getattr(args, "test_start_x_max", None) is not None:
            x_max = float(args.test_start_x_max)
        if getattr(args, "test_start_y_max", None) is not None:
            y_max = float(args.test_start_y_max)
            if getattr(args, "test_start_y_min", None) is None and getattr(args, "start_y_min", None) is None:
                y_min = -y_max  # stay symmetric around 0 unless a y_min was given
        if getattr(args, "test_start_y_min", None) is not None:
            y_min = float(args.test_start_y_min)
    where = "test" if test else "training"
    if x_min > x_max:
        raise ValueError(f"{where} start x range is empty: [{x_min}, {x_max}].")
    if y_min > y_max:
        raise ValueError(f"{where} start y range is empty: [{y_min}, {y_max}].")
    return x_min, x_max, y_min, y_max


def cross_index_interpolator(args: argparse.Namespace, x_lo: float, x_hi: float):
    """Map a start x to its expected nominal wall-crossing step.

    One batched nominal rollout over a 17-point grid of start positions, then
    linear interpolation per episode. Crossing steps are clamped to
    [1, horizon-1] so extreme starts degrade gracefully instead of raising.
    Used by --freeze_per_episode to align each episode's gate freeze with its
    own crossing time when the start x range is wide.
    """
    horizon = int(args.horizon)
    lo, hi = float(min(x_lo, x_hi)), float(max(x_lo, x_hi))
    grid = np.linspace(lo, hi, num=17, dtype=np.float64)
    plant = DoubleIntegratorTrue(dt=float(args.dt), pre_kp=float(args.pre_kp),
                                 pre_kd=float(args.pre_kd), drag_coeff=float(args.drag_coeff))
    x = torch.zeros(len(grid), 1, 4, dtype=torch.float32)
    x[:, 0, 0] = torch.from_numpy(grid.astype(np.float32))
    u = torch.zeros(len(grid), 1, 2, dtype=torch.float32)
    steps = []
    for _ in range(horizon):
        x = plant.forward(x, u)
        steps.append(x[:, 0, 0].clone())
    traj = torch.stack(steps, dim=1).numpy()                     # (G, horizon)
    wall = float(args.wall_x)
    crossed = traj <= wall
    idx = np.where(crossed.any(axis=1), crossed.argmax(axis=1),
                   np.abs(traj - wall).argmin(axis=1)).astype(np.float64)
    idx = np.clip(idx, 1, horizon - 1)

    def expected_cross(start_x: float) -> int:
        return int(round(float(np.interp(float(start_x), grid, idx))))

    return expected_cross


def sample_switching_gate(
    args: argparse.Namespace,
    rng: np.random.Generator,
    freeze_step: int,
) -> np.ndarray:
    gate = np.zeros(int(args.horizon), dtype=np.float32)
    amp = float(args.gate_amplitude)
    t = 0
    last_level = 0.0
    while t < freeze_step:
        remaining = freeze_step - t
        dwell_hi = min(int(args.gate_dwell_max), remaining)
        dwell_lo = min(int(args.gate_dwell_min), dwell_hi)
        dwell_lo = max(1, dwell_lo)
        dwell = int(rng.integers(dwell_lo, dwell_hi + 1))
        # Continuous random level in [-amp, amp], biased away from last to ensure a meaningful switch
        level = float(rng.uniform(-amp, amp))
        if abs(level - last_level) < 0.25 * amp:
            level = float(rng.uniform(-amp, amp))
        gate[t : t + dwell] = level
        last_level = level
        t += dwell
    # Late adversarial switch: 30% chance of one final change close to freeze
    is_adversarial = False
    if rng.random() < 0.30 and freeze_step > 6:
        late_t = int(rng.integers(max(t - 6, 0), freeze_step))
        late_level = float(rng.uniform(-amp, amp))
        if abs(late_level - last_level) < 0.25 * amp:
            late_level = float(rng.uniform(-amp, amp))
        gate[late_t:freeze_step] = late_level
        is_adversarial = True
    gate[freeze_step:] = gate[freeze_step - 1]
    return gate, is_adversarial


def sample_continuous_gate(
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    """Ornstein-Uhlenbeck gate center over the full horizon (continuous experiment).

    Three intuitive, decoupled knobs:
      --gate_corr_time   : correlation time tau (steps). Larger = slower, smoother.
      --gate_range       : within-episode roaming std around this episode's center.
      --gate_center_range: per-episode random center spread (fraction of amplitude),
                           so different episodes explore different heights.

    Discretized as g_{t+1} = mu + (1 - theta)*(g_t - mu) + sigma * N(0, 1) with
    theta = 1 - exp(-1/tau), sigma = gate_range * sqrt(2*theta - theta^2), and a
    per-episode reversion center mu ~ U(-c*amp, c*amp) (c = gate_center_range).
    Within an episode the std around mu is gate_range; across episodes mu roams,
    so the marginal vertical coverage is wider and not biased to the wall center.
    Never freezes or switches discretely, so the controller must infer the motion
    from history. Clamped to +/- gate_amplitude. Returns (gate, is_adversarial=
    False) to match sample_switching_gate's signature.
    """
    T = int(args.horizon)
    amp = float(args.gate_amplitude)
    tau = max(1.0, float(args.gate_corr_time))
    rng_std = float(args.gate_range)
    center_frac = float(getattr(args, "gate_center_range", 0.0))
    theta = 1.0 - float(np.exp(-1.0 / tau))
    sigma = rng_std * float(np.sqrt(max(2.0 * theta - theta * theta, 1e-8)))
    # Per-episode reversion center: different episodes settle around different
    # heights, so the dataset covers the full vertical extent of the wall.
    mu = float(rng.uniform(-center_frac * amp, center_frac * amp))
    gate = np.zeros(T, dtype=np.float32)
    g = mu + float(rng.normal(scale=rng_std))       # start from the stationary distribution
    gate[0] = float(np.clip(g, -amp, amp))
    for t in range(1, T):
        g = mu + (1.0 - theta) * (g - mu) + sigma * float(rng.standard_normal())
        g = float(np.clip(g, -amp, amp))
        gate[t] = g
    return gate, False


def noise_decay_window(args: argparse.Namespace) -> np.ndarray:
    """Multiplicative window g(t) in [0,1] applied to the process noise so the
    reconstructed disturbance w_t fades toward 0 by the end of the horizon.

    The initial-condition term (w_0 = x_0) is unaffected — only the t>=1 process
    noise is scaled. Modes:
      none         g=1 (stationary noise; original behaviour)
      taper        flat 1, then a cosine roll-off to 0 over the trailing
                   --noise_decay_ramp steps (0 -> auto = horizon//2)
      linear       1 - t/(T-1)  (linear ramp to 0 at the last step)
      exponential  --noise_decay_rate ** t  (anytime geometric decay)
    """
    T = int(args.horizon)
    mode = str(getattr(args, "noise_decay", "none")).lower()
    t = np.arange(T, dtype=np.float32)
    if mode == "none":
        return np.ones(T, dtype=np.float32)
    if mode == "linear":
        return np.clip(1.0 - t / max(T - 1, 1), 0.0, 1.0).astype(np.float32)
    if mode == "exponential":
        rate = float(getattr(args, "noise_decay_rate", 0.98))
        return np.power(rate, t).astype(np.float32)
    if mode == "taper":
        ramp = int(getattr(args, "noise_decay_ramp", 0)) or max(1, T // 2)
        ramp = min(max(ramp, 1), T)
        flat_until = T - ramp
        w = np.ones(T, dtype=np.float32)
        idx = np.arange(flat_until, T, dtype=np.float32)
        w[flat_until:] = 0.5 * (1.0 + np.cos(np.pi * (idx - flat_until) / ramp))
        return w
    raise ValueError(f"Unknown --noise_decay mode {mode!r}.")


def sample_paired_process_noise(
    *,
    args: argparse.Namespace,
    rng: np.random.Generator,
    batch_size: int,
    paired: bool,
) -> np.ndarray:
    horizon = int(args.horizon)
    nx = 4
    noise = np.zeros((batch_size, horizon, nx), dtype=np.float32)

    def draw_one() -> np.ndarray:
        seq = np.zeros((horizon, nx), dtype=np.float32)
        seq[:, :2] += rng.normal(scale=float(args.noise_pos_sigma), size=(horizon, 2)).astype(np.float32)
        seq[:, 2:] += rng.normal(scale=float(args.noise_vel_sigma), size=(horizon, 2)).astype(np.float32)
        burst_count = int(rng.integers(int(args.gust_count_min), int(args.gust_count_max) + 1))
        for _ in range(burst_count):
            duration = int(rng.integers(int(args.gust_duration_min), int(args.gust_duration_max) + 1))
            start = int(rng.integers(0, max(1, horizon - duration + 1)))
            amp_y = float(rng.uniform(float(args.gust_vel_y_min), float(args.gust_vel_y_max)))
            amp_y *= float(rng.choice([-1.0, 1.0]))
            amp_x = float(rng.uniform(-float(args.gust_vel_x_max), float(args.gust_vel_x_max)))
            seq[start : start + duration, 3] += amp_y
            seq[start : start + duration, 2] += amp_x
        seq[:, 3] = np.clip(seq[:, 3], -float(args.gust_clip_y), float(args.gust_clip_y))
        return seq

    if paired:
        if batch_size % 2 != 0:
            raise ValueError("Paired noise batches require an even batch size.")
        for i in range(batch_size // 2):
            seq = draw_one()
            noise[2 * i] = seq
            noise[2 * i + 1] = seq
    else:
        for i in range(batch_size):
            noise[i] = draw_one()

    window = noise_decay_window(args)  # (horizon,) in [0, 1]
    if not np.allclose(window, 1.0):
        noise = noise * window[None, :, None]
    return noise


def sample_batch(
    *,
    args: argparse.Namespace,
    batch_size: int,
    seed: int,
    paired: bool,
    shuffle: bool,
    expected_cross_index: int,
    use_test_starts: bool = False,
) -> ScenarioBatch:
    if paired and batch_size % 2 != 0:
        raise ValueError("Paired batches require an even batch size.")

    rng = np.random.default_rng(seed)
    base_count = batch_size // 2 if paired else batch_size
    freeze_step = max(1, expected_cross_index - int(args.gate_settle_steps))
    x_min, x_max, y_min, y_max = resolve_start_ranges(args, test=use_test_starts)
    # Per-episode gate freeze: align each episode's freeze with ITS start's
    # predicted crossing instead of the global mean-start crossing.
    cross_of_x = None
    if not is_continuous_gate(args) and bool(getattr(args, "freeze_per_episode", False)):
        cross_of_x = cross_index_interpolator(args, x_min, x_max)

    starts = []
    goals = []
    gates = []
    pair_ids = []
    adv_flags = []

    for pair_idx in range(base_count):
        start_x = float(rng.uniform(x_min, x_max))
        start_y = float(rng.uniform(y_min, y_max))
        if is_continuous_gate(args):
            gate, is_adv = sample_continuous_gate(args, rng)
        else:
            fs = freeze_step if cross_of_x is None else max(
                1, cross_of_x(start_x) - int(args.gate_settle_steps))
            gate, is_adv = sample_switching_gate(args, rng, fs)

        if paired:
            starts.extend([[start_x, start_y], [start_x, start_y]])
            goals.extend([[0.0, 0.0], [0.0, 0.0]])
            gates.extend([gate, -gate])
            pair_ids.extend([pair_idx, pair_idx])
            adv_flags.extend([is_adv, is_adv])
        else:
            starts.append([start_x, start_y])
            goals.append([0.0, 0.0])
            gates.append(gate)
            pair_ids.append(pair_idx)
            adv_flags.append(is_adv)

    starts_np = np.asarray(starts, dtype=np.float32)
    goals_np = np.asarray(goals, dtype=np.float32)
    gates_np = np.stack(gates, axis=0).astype(np.float32)
    pair_ids_np = np.asarray(pair_ids, dtype=np.int64)
    adv_np = np.asarray(adv_flags, dtype=bool)
    noise_np = sample_paired_process_noise(args=args, rng=rng, batch_size=batch_size, paired=paired)

    if shuffle:
        order = rng.permutation(batch_size)
        starts_np = starts_np[order]
        goals_np = goals_np[order]
        gates_np = gates_np[order]
        pair_ids_np = pair_ids_np[order]
        adv_np = adv_np[order]
        noise_np = noise_np[order]

    gate_v_np, gate_ema_np, gate_slow_ema_np, switch_age_np = build_gate_features(gates_np, float(args.context_ema_alpha))

    return ScenarioBatch(
        start=torch.from_numpy(starts_np),
        goal=torch.from_numpy(goals_np),
        gate_y=torch.from_numpy(gates_np),
        gate_v=torch.from_numpy(gate_v_np),
        gate_ema=torch.from_numpy(gate_ema_np),
        gate_slow_ema=torch.from_numpy(gate_slow_ema_np),
        switch_age=torch.from_numpy(switch_age_np),
        process_noise=torch.from_numpy(noise_np),
        pair_id=torch.from_numpy(pair_ids_np),
        is_adversarial=torch.from_numpy(adv_np),
    )


def make_x0(batch: ScenarioBatch, device: torch.device) -> torch.Tensor:
    vel0 = torch.zeros(batch.start.shape[0], 2, device=device, dtype=batch.start.dtype)
    return torch.cat([batch.start.to(device), vel0], dim=-1).unsqueeze(1)


def build_context(
    *,
    args: argparse.Namespace,
    batch: ScenarioBatch,
    x_t: torch.Tensor,
    t: int,
    expected_cross_index: int,
    training: bool = False,
) -> torch.Tensor:
    state = as_bt(x_t)
    pos = state[..., :2]
    x_pos = pos[..., 0]
    y_pos = pos[..., 1]
    vel_x = state[..., 2]
    vel_y = state[..., 3]

    # Apply observation delay: controller sees gate from `delay` steps ago
    delay = int(args.gate_obs_delay)
    t_obs = max(0, t - delay)
    gate_t = batch.gate_y[:, t_obs : t_obs + 1]
    gate_v_t = batch.gate_v[:, t_obs : t_obs + 1]
    gate_ema_t = batch.gate_ema[:, t_obs : t_obs + 1]
    gate_slow_ema_t = batch.gate_slow_ema[:, t_obs : t_obs + 1]
    switch_age_t = batch.switch_age[:, t_obs : t_obs + 1]

    rel_wall_x = x_pos - float(args.wall_x)
    gate_error = y_pos - gate_t
    approach = torch.exp(-0.5 * (rel_wall_x / float(args.wall_focus_sigma)) ** 2)
    goal_dx = -x_pos
    goal_dy = -y_pos
    # Normalised time remaining until expected wall crossing (positive = before wall)
    time_to_wall = torch.full_like(x_pos, (expected_cross_index - t) / max(float(args.horizon), 1.0))

    x_scale = max(float(args.start_x_max), abs(float(args.wall_x)), 1.0)
    y_scale = max(float(args.corridor_limit), abs(float(args.gate_amplitude)), 1.0)
    age_scale = max(float(args.horizon), 1.0)
    feat_map = {
        "gate_obs": gate_t / y_scale,
        "gate_vel": gate_v_t / y_scale,
        "gate_ema": gate_ema_t / y_scale,
        "gate_slow_ema": gate_slow_ema_t / y_scale,
        "gate_error": gate_error / y_scale,
        "rel_wall_x": rel_wall_x / x_scale,
        "goal_dx": goal_dx / x_scale,
        "goal_dy": goal_dy / y_scale,
        "approach": approach,
        "switch_age": switch_age_t / age_scale,
        "time_to_wall": time_to_wall,
        # Own velocity (fair, endogenous). The PD-pre-stabilized plant keeps
        # |v| at O(1) over the sampled start ranges, so no extra normalization.
        "vel_x": vel_x,
        "vel_y": vel_y,
    }
    feats = resolve_context_features(args)
    z_t = float(args.z_scale) * torch.cat([feat_map[k] for k in feats], dim=-1)
    # Context dropout: during training, randomly zero the gate-OBSERVATION features
    # together (per sample) so the SSM must rely on history. Own-state features
    # (rel_wall_x, goal_*, approach) and schedule features are never dropped.
    dropout_p = float(args.context_dropout_p)
    if training and dropout_p > 0.0:
        gate_obs_keys = {"gate_obs", "gate_vel", "gate_ema", "gate_slow_ema", "gate_error"}
        gate_idx = [i for i, k in enumerate(feats) if k in gate_obs_keys]
        if gate_idx:
            mask = (torch.rand(z_t.shape[0], device=z_t.device) > dropout_p).float()
            idx = torch.tensor(gate_idx, device=z_t.device)
            z_t[:, idx] = z_t[:, idx] * mask.unsqueeze(-1)
    return z_t.unsqueeze(1)


def build_contextual_controller(
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[PBController, DoubleIntegratorTrue]:
    """Build a PBController whose operator is a single neural_ssm.ContextualDeepSSM.

    Context is injected through the ports named in --ctx_modes (mixer/input/gate/
    select).  The SSM core reuses the shared --ssm_* sizing; the 'input' driver is
    L2-projected by --ctx_filter, the 'mixer' router is bounded by
    --ctx_mixer_bound, and an optional prescribed L2 cap is set via --ctx_gamma.
    """
    nx = 4
    nu = 2
    z_dim = context_dim(args)

    modes = tuple(m.strip().lower() for m in str(args.ctx_modes).split(",") if m.strip())
    if getattr(args, "ctx_select", False) and "select" not in modes:
        modes = modes + ("select",)
    if not modes:
        raise ValueError("--ctx_modes must name at least one of mixer,input,gate,select.")
    if "select" in modes and str(args.ssm_param) not in ("tv", "tvc"):
        raise ValueError(
            "The contextual 'select' port injects context into the selective SSM "
            "matrices and needs a selective core. Pass --ssm_param tv or --ssm_param tvc "
            f"(got --ssm_param {args.ssm_param!r}). Either set a selective core or drop "
            "'select' (remove --ctx_select / remove it from --ctx_modes)."
        )

    ramp = int(args.ctx_filter_ramp)
    ramp = ramp if ramp > 0 else None
    gamma = float(args.ctx_gamma)
    gamma = gamma if gamma > 0.0 else None

    w_augmenter = None
    if getattr(args, "use_w_augment", False):
        w_augmenter = WIntegralAugmenter(decay=float(args.w_augment_decay)).to(device)

    nominal_plant = DoubleIntegratorNominal(
        dt=float(args.dt), pre_kp=float(args.pre_kp), pre_kd=float(args.pre_kd),
        drag_coeff=float(args.drag_coeff),
    )
    true_plant = DoubleIntegratorTrue(
        dt=float(args.dt), pre_kp=float(args.pre_kp), pre_kd=float(args.pre_kd),
        drag_coeff=float(args.drag_coeff),
    )

    operator = MpContextualSSM(
        nx,
        z_dim,
        nu,
        context_modes=modes,
        detach_state=False,
        w_augmenter=w_augmenter,
        # context ports
        context_filter=args.ctx_filter,
        horizon=int(args.horizon),
        context_filter_ramp=ramp,
        mixer_bound=float(args.ctx_mixer_bound),
        d_features=int(args.ctx_d_features),
        gate_per_channel=bool(args.ctx_gate_per_channel),
        # shared SSM core sizing (same knobs as the other SSM variants)
        param=args.ssm_param,
        n_layers=int(args.ssm_layers),
        d_model=int(args.ssm_d_model),
        d_state=int(args.ssm_d_state),
        ff=args.ssm_ff,
        bcd_nonlinearity=args.ssm_bcd_nonlinearity,
        gamma=gamma,
    ).to(device)

    controller = PBController(
        plant=nominal_plant,
        operator=operator,
        u_nominal=None,
        u_dim=nu,
        detach_state=False,
    ).to(device)

    x_probe = torch.zeros(4, 1, nx, device=device)
    z_probe = torch.zeros(4, 1, z_dim, device=device)
    ok, msg = validate_component_compatibility(
        controller=controller,
        plant_true=true_plant,
        x0=x_probe,
        z0=z_probe,
        raise_on_error=False,
    )
    if not ok:
        raise RuntimeError(f"PB contextual component compatibility check failed: {msg}")
    return controller, true_plant


def build_controller(
    device: torch.device,
    args: argparse.Namespace,
    *,
    mp_only: bool = False,
    force_no_lift: bool = False,
    factor_rank_override: int | None = None,
    ssm_d_model_override: int | None = None,
    ssm_layers_override: int | None = None,
    contextual: bool = False,
) -> tuple[PBController, DoubleIntegratorTrue]:
    """Build a PBController + true plant.

    Args:
        mp_only: When True, bypass M_b and output M_p(w) directly as u_boost.
                 M_p output dim is set to nu.  Compatible with mp_context_lift.
        factor_rank_override: Override the factorization width s. Setting this to
                              1 recovers the MAD magnitude-and-direction policy.
        ssm_d_model_override: Override --ssm_d_model for M_p (used by mp_only_context
                              to match the factorized parameter budget).
        ssm_layers_override: Override --ssm_layers for M_p (same purpose).
        contextual: When True, build a single neural_ssm.ContextualDeepSSM
                    operator (M_p and M_b fused; context injected via the
                    --ctx_modes ports) instead of the factorized M_b x M_p stack.
    """
    if contextual:
        return build_contextual_controller(device, args)

    nx = 4
    nu = 2
    z_dim = context_dim(args)
    feat_dim = int(factor_rank_override if factor_rank_override is not None else args.feat_dim)
    if feat_dim <= 0:
        raise ValueError(f"factorization width must be > 0, got {feat_dim}")
    mp_context_lifter = None

    # w augmentation doubles the operator's w input dimension
    w_augmenter = None
    w_dim = nx
    if getattr(args, "use_w_augment", False):
        w_augmenter = WIntegralAugmenter(decay=float(args.w_augment_decay)).to(device)
        w_dim = nx * 2
    mp_in_dim = w_dim

    if bool(args.mp_context_lift) and not force_no_lift:
        mp_context_lifter = LpContextLifter(
            z_dim=z_dim,
            out_dim=int(args.mp_context_lift_dim),
            lift_type=args.mp_context_lift_type,
            hidden_dim=int(args.mp_context_hidden_dim),
            decay_law=args.mp_context_decay_law,
            decay_rate=float(args.mp_context_decay_rate),
            decay_power=float(args.mp_context_decay_power),
            decay_horizon=int(args.mp_context_decay_horizon),
            lp_p=float(args.mp_context_lp_p),
            scale=float(args.mp_context_scale),
        ).to(device)
        mp_in_dim = w_dim + int(args.mp_context_lift_dim)

    nominal_plant = DoubleIntegratorNominal(
        dt=float(args.dt),
        pre_kp=float(args.pre_kp),
        pre_kd=float(args.pre_kd),
        drag_coeff=float(args.drag_coeff),
    )
    true_plant = DoubleIntegratorTrue(
        dt=float(args.dt),
        pre_kp=float(args.pre_kp),
        pre_kd=float(args.pre_kd),
        drag_coeff=float(args.drag_coeff),
    )

    ssm_d_model = int(ssm_d_model_override if ssm_d_model_override is not None else args.ssm_d_model)
    ssm_n_layers = int(ssm_layers_override if ssm_layers_override is not None else args.ssm_layers)
    # When mp_only, M_p outputs u directly (dim=nu); feat_dim is only used for factorized mode.
    mp_out_dim = nu if mp_only else feat_dim
    mp = MpDeepSSM(
        mp_in_dim,
        mp_out_dim,
        mode="loop",
        param=args.ssm_param,
        n_layers=ssm_n_layers,
        d_model=ssm_d_model,
        d_state=int(args.ssm_d_state),
        ff=args.ssm_ff,
        bcd_nonlinearity=args.ssm_bcd_nonlinearity,
    ).to(device)
    if mp_only:
        mb = None
    else:
        mb = BoundedMLPOperator(
            w_dim=w_dim,
            z_dim=z_dim,
            r=nu,
            s=feat_dim,
            hidden_dim=int(args.mb_hidden),
            num_layers=int(args.mb_layers),
            use_z_residual=True,
            z_residual_gain=float(args.z_residual_gain),
            bound_mode="softsign",
            clamp_value=float(args.mb_bound),
        ).to(device)
    controller = build_factorized_controller(
        nominal_plant=nominal_plant,
        mp=mp,
        mb=mb,
        u_dim=nu,
        detach_state=False,
        u_nominal=None,
        mp_context_lifter=mp_context_lifter,
        mp_only=mp_only,
        w_augmenter=w_augmenter,
    ).to(device)

    x_probe = torch.zeros(4, 1, nx, device=device)
    z_probe = torch.zeros(4, 1, z_dim, device=device)
    ok, msg = validate_component_compatibility(
        controller=controller,
        plant_true=true_plant,
        x0=x_probe,
        z0=z_probe,
        raise_on_error=False,
    )
    if not ok:
        raise RuntimeError(f"PB component compatibility check failed: {msg}")
    return controller, true_plant


def rollout_nominal(
    *,
    args: argparse.Namespace,
    batch: ScenarioBatch,
    device: torch.device,
) -> RolloutArtifacts:
    batch = batch.to(device)
    plant_true = DoubleIntegratorTrue(dt=float(args.dt), pre_kp=float(args.pre_kp), pre_kd=float(args.pre_kd), drag_coeff=float(args.drag_coeff))
    x = make_x0(batch, device)
    x_log = []
    u_log = []
    w_log = []
    u_zero = torch.zeros(x.shape[0], 1, 2, device=device, dtype=x.dtype)
    for t in range(int(args.horizon)):
        x_next = plant_true.forward(x, u_zero, t=t) + batch.process_noise[:, t : t + 1, :]
        w_t = x_next - plant_true.forward(x, u_zero, t=t)
        x_log.append(x_next)
        u_log.append(u_zero)
        w_log.append(w_t)
        x = x_next
    return RolloutArtifacts(
        x_seq=torch.cat(x_log, dim=1),
        u_seq=torch.cat(u_log, dim=1),
        w_seq=torch.cat(w_log, dim=1),
    )


def rollout_pb_variant(
    *,
    args: argparse.Namespace,
    batch: ScenarioBatch,
    device: torch.device,
    controller: PBController,
    plant_true: DoubleIntegratorTrue,
    zero_context: bool,
    expected_cross_index: int = 0,
    training: bool = False,
) -> RolloutArtifacts:
    batch = batch.to(device)
    x0 = make_x0(batch, device)
    z_dim = context_dim(args)

    if zero_context:
        def context_fn(x_t: torch.Tensor, t: int) -> torch.Tensor:
            return torch.zeros(x_t.shape[0], 1, z_dim, device=x_t.device, dtype=x_t.dtype)
    else:
        def context_fn(x_t: torch.Tensor, t: int) -> torch.Tensor:
            return build_context(args=args, batch=batch, x_t=x_t, t=t, expected_cross_index=expected_cross_index, training=training)

    w0_operator = x0
    if getattr(args, "use_w0_clip", True):
        w0_operator = x0.clamp(-float(args.w0_clip), float(args.w0_clip))
    result = rollout_pb(
        controller=controller,
        plant_true=plant_true,
        x0=x0,
        horizon=int(args.horizon),
        context_fn=context_fn,
        w0=w0_operator,
        process_noise_seq=batch.process_noise,
    )
    return RolloutArtifacts(x_seq=result.x_seq, u_seq=result.u_seq, w_seq=result.w_seq)


def rollout_variant(
    *,
    args: argparse.Namespace,
    batch: ScenarioBatch,
    device: torch.device,
    mode: str,
    controller: PBController | None,
    plant_true: DoubleIntegratorTrue | None,
    expected_cross_index: int = 0,
    training: bool = False,
) -> RolloutArtifacts:
    if mode == "nominal":
        return rollout_nominal(args=args, batch=batch, device=device)
    if controller is None or plant_true is None:
        raise ValueError(f"Controller and plant are required for mode {mode}")
    if mode == "disturbance_only":
        return rollout_pb_variant(
            args=args,
            batch=batch,
            device=device,
            controller=controller,
            plant_true=plant_true,
            zero_context=True,
            expected_cross_index=expected_cross_index,
            training=training,
        )
    if mode in ("context", "mad_context", "mp_only_context", "context_no_lift", "contextual_ssm"):
        return rollout_pb_variant(
            args=args,
            batch=batch,
            device=device,
            controller=controller,
            plant_true=plant_true,
            zero_context=False,
            expected_cross_index=expected_cross_index,
            training=training,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def wall_weights(x_pos: torch.Tensor, wall_x: float, sigma: float) -> torch.Tensor:
    raw = torch.exp(-0.5 * ((x_pos - float(wall_x)) / max(float(sigma), 1e-6)) ** 2)
    return raw / raw.sum(dim=1, keepdim=True).clamp_min(1e-6)


def loss_gate_half_width(args: argparse.Namespace, training: bool) -> float:
    half_width = float(args.gate_half_width)
    if training and is_continuous_gate(args):
        margin = max(0.0, float(getattr(args, "gate_margin_train", 0.0)))
        half_width = max(1e-4, half_width - margin)
    return half_width


# Loss term key -> the legacy raw-mode *_weight argument that scales it.
_LOSS_WEIGHT_ARG = {
    "goal_stage": "goal_stage_weight",
    "goal_term": "goal_terminal_weight",
    "terminal_vel": "terminal_vel_weight",
    "wall_track": "wall_track_weight",
    "wall_collision": "wall_collision_weight",
    "control": "control_weight",
    "corridor": "corridor_weight",
    "settle_vel": "settle_vel_weight",
}


def _raw_loss_terms(
    args: argparse.Namespace,
    batch: ScenarioBatch,
    rollout: RolloutArtifacts,
    collision_sharpness_override: float | None = None,
    training: bool = False,
) -> dict[str, torch.Tensor]:
    """Unweighted per-sample (B,) loss terms — the building blocks of compute_loss."""
    goal = batch.goal.to(rollout.x_seq.device).unsqueeze(1)
    pos = rollout.x_seq[..., :2]
    x_pos = pos[..., 0]
    y_pos = pos[..., 1]
    gate = batch.gate_y.to(rollout.x_seq.device)

    # Smooth L1 (Huber) for goal distance — more robust than pure L2
    goal_delta = pos - goal
    goal_dist = F.huber_loss(pos, goal.expand_as(pos), reduction="none", delta=0.5).sum(dim=-1)
    goal_dist_l2 = torch.norm(goal_delta, dim=-1)

    w_wall = wall_weights(x_pos, float(args.wall_x), float(args.wall_focus_sigma))
    gate_error = y_pos - gate
    sharpness = collision_sharpness_override if collision_sharpness_override is not None else float(args.collision_sharpness)
    gate_half_width_loss = loss_gate_half_width(args, training=training)
    collision_soft = F.softplus(
        sharpness * (gate_error.abs() - gate_half_width_loss)
    ) / max(sharpness, 1e-6)
    corridor_soft = F.softplus(
        float(args.corridor_sharpness) * (y_pos.abs() - float(args.corridor_limit))
    ) / float(args.corridor_sharpness)

    terminal_vel = torch.zeros_like(goal_dist_l2[:, -1])
    if getattr(args, "use_terminal_vel", True):
        terminal_vel = torch.sum(rollout.x_seq[:, -1, 2:].square(), dim=-1)

    # Deceleration-near-goal penalty: speed weighted by a Gaussian in distance-to-
    # goal (normalized over time). Penalizes carrying velocity into the target, so
    # it curbs overshoot without slowing the far approach / gate crossing.
    settle_sigma = max(float(getattr(args, "settle_sigma", 0.35)), 1e-6)
    w_goal = torch.exp(-0.5 * goal_delta.square().sum(dim=-1) / (settle_sigma ** 2))
    w_goal = w_goal / w_goal.sum(dim=1, keepdim=True).clamp_min(1e-6)
    vel_sq = rollout.x_seq[..., 2:].square().sum(dim=-1)
    settle_vel = (w_goal * vel_sq).sum(dim=1)

    return {
        "goal_stage": goal_dist.mean(dim=1),
        "goal_term": goal_dist_l2[:, -1],
        "terminal_vel": terminal_vel,
        "wall_track": (w_wall * gate_error.square()).sum(dim=1),
        "wall_collision": (w_wall * collision_soft).sum(dim=1),
        "control": torch.sum(rollout.u_seq.square(), dim=-1).mean(dim=1),
        "corridor": corridor_soft.mean(dim=1),
        "settle_vel": settle_vel,
    }


def _loss_scales_from_nominal(args: argparse.Namespace, batch: ScenarioBatch,
                              device: torch.device) -> dict[str, float]:
    """Per-term reference scales from the nominal (no-boost) rollout.

    The nominal policy meaningfully incurs the *task* terms (goal distance, wall
    tracking/collision), so those are scaled by "how bad is doing nothing." It does
    NOT exercise the regularizers (control effort = 0, stays in the corridor, ends
    slow), so any term whose nominal scale is negligible falls back to a unit
    reference — its alpha then acts as a direct O(1) weight on the raw term instead
    of exploding by dividing by ~0.
    """
    nominal = rollout_nominal(args=args, batch=batch, device=device)
    raw = _raw_loss_terms(args, batch, nominal)  # default (full) half-width, no override
    scales = {k: float(v.mean().item()) for k, v in raw.items()}
    thresh = 0.05 * (max(scales.values()) if scales else 1.0)
    return {k: (v if v >= thresh else 1.0) for k, v in scales.items()}


def compute_loss(
    *,
    args: argparse.Namespace,
    batch: ScenarioBatch,
    rollout: RolloutArtifacts,
    collision_sharpness_override: float | None = None,
    training: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-objective loss. Two modes:

      raw (default):  total = Σ_i  (*_weight)_i · term_i        (legacy behavior)
      normalized:     total = Σ_i  (alpha_*)_i · term_i / s_i   (--loss_normalize)

    where s_i is term_i's scale on the nominal rollout (computed once, cached on
    args). Normalization decouples each objective's scale from its priority, so the
    --alpha_* weights are interpretable and the balance is invariant to the gate
    regime. Raw mode is bit-for-bit the previous behavior.
    """
    raw = _raw_loss_terms(args, batch, rollout, collision_sharpness_override, training)
    if bool(getattr(args, "loss_normalize", False)):
        scales = getattr(args, "_loss_scales", None)
        if scales is None:
            scales = _loss_scales_from_nominal(args, batch, rollout.x_seq.device)
            args._loss_scales = scales  # cache: one scenario-level reference, shared
        contrib = {
            k: float(getattr(args, f"alpha_{k}", 1.0)) * (raw[k] / max(float(scales.get(k, 1.0)), 1e-6))
            for k in raw
        }
    else:
        contrib = {k: float(getattr(args, _LOSS_WEIGHT_ARG[k])) * raw[k] for k in raw}

    total_per = (contrib["goal_stage"] + contrib["goal_term"] + contrib["terminal_vel"]
                 + contrib["wall_track"] + contrib["wall_collision"]
                 + contrib["control"] + contrib["corridor"] + contrib["settle_vel"])
    total = total_per.mean()
    parts = tensor_scalars_to_floats({
        "loss_total": total,
        "loss_goal_stage": contrib["goal_stage"].mean(),
        "loss_goal_term": contrib["goal_term"].mean(),
        "loss_terminal_vel": contrib["terminal_vel"].mean(),
        "loss_wall_track": contrib["wall_track"].mean(),
        "loss_wall_collision": contrib["wall_collision"].mean(),
        "loss_control": contrib["control"].mean(),
        "loss_corridor": contrib["corridor"].mean(),
        "loss_settle_vel": contrib["settle_vel"].mean(),
    })
    return total, parts


def crossing_indices(x_pos: torch.Tensor, wall_x: float) -> torch.Tensor:
    crossed = x_pos <= float(wall_x)
    has_crossing = crossed.any(dim=1)
    # argmax returns the first True for boolean-as-integer rows; all-False rows
    # are replaced by the nearest-to-wall index below.
    first_crossing = crossed.to(torch.int8).argmax(dim=1)
    nearest = torch.abs(x_pos - float(wall_x)).argmin(dim=1)
    return torch.where(has_crossing, first_crossing, nearest)


def interpolated_wall_crossing(
    *,
    x_pos: torch.Tensor,
    y_pos: torch.Tensor,
    gate: torch.Tensor,
    wall_x: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return discrete crossing indices plus y/g interpolated at x == wall_x."""
    cross_idx = crossing_indices(x_pos, wall_x)
    row = torch.arange(x_pos.shape[0], device=x_pos.device)
    y1 = y_pos[row, cross_idx]
    g1 = gate[row, cross_idx]
    prev_idx = (cross_idx - 1).clamp_min(0)
    x0 = x_pos[row, prev_idx]
    x1 = x_pos[row, cross_idx]
    y0 = y_pos[row, prev_idx]
    g0 = gate[row, prev_idx]
    denom = x1 - x0
    wall = float(wall_x)
    valid = ((cross_idx > 0)
             & (((x0 - wall) * (x1 - wall)) <= 0)
             & (denom.abs() > 1e-8))
    safe_denom = torch.where(valid, denom, torch.ones_like(denom))
    alpha = ((wall - x0) / safe_denom).clamp(0.0, 1.0)
    y_interp = y0 + alpha * (y1 - y0)
    g_interp = g0 + alpha * (g1 - g0)
    y_cross = torch.where(valid, y_interp, y1)
    g_cross = torch.where(valid, g_interp, g1)
    return cross_idx, y_cross, g_cross


@torch.no_grad()
def evaluate_variant(
    *,
    args: argparse.Namespace,
    batch: ScenarioBatch,
    device: torch.device,
    mode: str,
    controller: PBController | None,
    plant_true: DoubleIntegratorTrue | None,
    expected_cross_index: int = 0,
) -> dict:
    # Pure evaluation: no autograd graph. Halves eval peak memory (important on
    # fractional-GPU cluster slices where VRAM is capped) and speeds it up.
    with torch.no_grad():
        batch_dev = batch.to(device)
        with cuda_autocast(args, device):
            rollout = rollout_variant(
                args=args,
                batch=batch_dev,
                device=device,
                mode=mode,
                controller=controller,
                plant_true=plant_true,
                expected_cross_index=expected_cross_index,
            )
            avg_cost, loss_parts = compute_loss(args=args, batch=batch_dev, rollout=rollout)
        pos = rollout.x_seq[..., :2]
        x_pos = pos[..., 0]
        y_pos = pos[..., 1]
        cross_idx, y_cross, g_cross = interpolated_wall_crossing(
            x_pos=x_pos,
            y_pos=y_pos,
            gate=batch_dev.gate_y,
            wall_x=float(args.wall_x),
        )
        cross_error = y_cross - g_cross
        collided = cross_error.abs() > float(args.gate_half_width)
        terminal_dist = torch.norm(pos[:, -1, :] - batch_dev.goal, dim=-1)
        goal_success = terminal_dist < float(args.goal_tol)
        success = (~collided) & goal_success

        metrics = tensor_scalars_to_floats({
            "avg_cost": avg_cost,
            "success_rate": success.float().mean(),
            "wall_success_rate": (~collided).float().mean(),
            "goal_success_rate": goal_success.float().mean(),
            "collision_rate": collided.float().mean(),
            "avg_abs_cross_error": cross_error.abs().mean(),
            "avg_terminal_dist": terminal_dist.mean(),
            "avg_control_energy": torch.sum(rollout.u_seq.square(), dim=-1).mean(),
            "avg_abs_reconstructed_w": rollout.w_seq.abs().mean(),
        })
        metrics.update(loss_parts)
        # Under CUDA BF16 autocast the rollout comes out bfloat16, which numpy
        # (hence every plot function calling .numpy()) cannot represent — store
        # rollouts as float32. No-op on CPU/FP32 runs.
        metrics["rollout"] = {
            "x_seq": rollout.x_seq.detach().cpu().float(),
            "u_seq": rollout.u_seq.detach().cpu().float(),
            "w_seq": rollout.w_seq.detach().cpu().float(),
            "cross_idx": cross_idx.detach().cpu(),
            # Per-episode outcomes (bool, (B,)) for the start-generalization map.
            "success": success.detach().cpu(),
            "collided": collided.detach().cpu(),
        }
    return metrics


def choose_better_result(candidate: tuple[float, float, float], incumbent: tuple[float, float, float] | None) -> bool:
    if incumbent is None:
        return True
    return candidate < incumbent


def print_train_epoch_status(
    *,
    mode: str,
    epoch: int,
    epochs: int,
    record: dict,
    val_metrics: dict | None = None,
    is_best: bool = False,
) -> None:
    train_msg = (
        f"[{mode}] epoch {epoch:03d}/{epochs:03d} "
        f"train={record['train_loss']:.4f} "
        f"goal_s={record['loss_goal_stage']:.4f} "
        f"goal_T={record['loss_goal_term']:.4f} "
        f"wall={record['loss_wall_track']:.4f} "
        f"coll={record['loss_wall_collision']:.4f} "
        f"ctrl={record['loss_control']:.4f} "
        f"corr={record['loss_corridor']:.4f} "
        f"lr={record['lr']:.5f}"
    )
    print(train_msg)
    if val_metrics is None:
        return
    best_tag = " best" if is_best else ""
    val_msg = (
        f"[{mode}]            "
        f"val={val_metrics['avg_cost']:.4f} "
        f"succ={val_metrics['success_rate']:.3f} "
        f"wall={val_metrics['wall_success_rate']:.3f} "
        f"goal={val_metrics['goal_success_rate']:.3f} "
        f"term={val_metrics['avg_terminal_dist']:.3f}"
        f"{best_tag}"
    )
    print(val_msg)


def train_controller(
    *,
    args: argparse.Namespace,
    device: torch.device,
    mode: str,
    val_batch: ScenarioBatch,
    expected_cross_index: int,
    warm_start_state: dict | None = None,
    mp_only: bool = False,
    force_no_lift: bool = False,
    factor_rank_override: int | None = None,
    ssm_d_model_override: int | None = None,
    ssm_layers_override: int | None = None,
    contextual: bool = False,
    run_dir: Path | None = None,
    save_tag: str | None = None,
) -> tuple[PBController | None, DoubleIntegratorTrue | None, list[dict], dict]:
    """Train one variant. When ``run_dir`` is given, the best checkpoint and the
    training history are additionally persisted at every eval (as
    ``{tag}_controller.partial.pt`` / ``train_history_partial.json``) so a
    preempted or killed cluster job never loses a finished-so-far state."""
    # Upload the validation batch once; evaluate_variant's own .to(device) then
    # becomes a no-op instead of a fresh H2D copy at every eval.
    val_batch = val_batch.to(device)
    if mode == "nominal":
        metrics = evaluate_variant(
            args=args,
            batch=val_batch,
            device=device,
            mode=mode,
            controller=None,
            plant_true=None,
            expected_cross_index=expected_cross_index,
        )
        return None, None, [], metrics
    save_tag = save_tag or mode

    mode_epochs = epochs_for_mode(args, mode)
    controller, plant_true = build_controller(
        device,
        args,
        mp_only=mp_only,
        force_no_lift=force_no_lift,
        factor_rank_override=factor_rank_override,
        ssm_d_model_override=ssm_d_model_override,
        ssm_layers_override=ssm_layers_override,
        contextual=contextual,
    )
    print(f"[{mode}] trainable params: {count_params(controller):,}")

    # Warm-start from a previously trained state (e.g. disturbance_only -> context)
    if warm_start_state is not None:
        missing, unexpected = controller.load_state_dict(warm_start_state, strict=False)
        print(f"[{mode}] warm-started from provided checkpoint "
              f"(missing={len(missing)}, unexpected={len(unexpected)}).")

    maybe_compile_operator(controller, args)

    # AdamW with separate LRs for the recurrent SSM core (full lr) and the
    # surrounding bounded heads / MLP operator (half lr).
    #   FactorizedOperator   -> SSM core is operator.mp
    #   MpContextualSSM      -> SSM core is operator.core.core (DeepSSM inside
    #                           ContextualDeepSSM); mixer/gate heads get half lr.
    op = controller.operator
    if hasattr(op, "mp") and getattr(op, "mp") is not None:
        ssm_core = op.mp
    elif hasattr(op, "core"):
        ssm_core = getattr(op.core, "core", op.core)
    else:
        ssm_core = None

    if ssm_core is not None:
        fast_ids = {id(p) for p in ssm_core.parameters()}
        slow_params = [p for p in controller.parameters() if id(p) not in fast_ids]
        param_groups = [{"params": list(ssm_core.parameters()), "lr": float(args.lr)}]
        if slow_params:
            param_groups.append({"params": slow_params, "lr": float(args.lr) * 0.5})
    else:
        param_groups = [{"params": list(controller.parameters()), "lr": float(args.lr)}]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # Linear warmup (8 epochs) then cosine decay
    warmup_epochs = min(8, mode_epochs // 4)
    cosine_epochs = max(1, mode_epochs - warmup_epochs)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=float(args.lr_min)),
        ],
        milestones=[warmup_epochs],
    )

    history: list[dict] = []
    best_state = None
    best_score = None
    best_val_metrics = None

    for epoch, train_batch in iter_training_batches(
        args=args,
        epochs=mode_epochs,
        expected_cross_index=expected_cross_index,
        pin_memory=(device.type == "cuda"),
    ):
        controller.train()
        train_batch_dev = train_batch.to(device)

        # Curriculum: ramp collision sharpness from 20% to 100% over first 40% of training
        sharpness_frac = min(1.0, epoch / max(1, 0.4 * mode_epochs))
        curr_sharpness = float(args.collision_sharpness) * (0.2 + 0.8 * sharpness_frac)

        with cuda_autocast(args, device):
            rollout = rollout_variant(
                args=args,
                batch=train_batch_dev,
                device=device,
                mode=mode,
                controller=controller,
                plant_true=plant_true,
                expected_cross_index=expected_cross_index,
                training=True,
            )
            loss, parts = compute_loss(
                args=args,
                batch=train_batch_dev,
                rollout=rollout,
                collision_sharpness_override=curr_sharpness,
                training=True,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(controller.parameters(), float(args.grad_clip))
        optimizer.step()

        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": parts["loss_total"],
            "lr": to_python_float(scheduler.get_last_lr()[0]),
            "curr_sharpness": curr_sharpness,
        }
        record.update(parts)

        if epoch % int(args.eval_every) == 0 or epoch == mode_epochs:
            controller.eval()
            val_metrics = evaluate_variant(
                args=args,
                batch=val_batch,
                device=device,
                mode=mode,
                controller=controller,
                plant_true=plant_true,
                expected_cross_index=expected_cross_index,
            )
            record["val_cost"] = float(val_metrics["avg_cost"])
            record["val_success_rate"] = float(val_metrics["success_rate"])
            candidate_score = (
                1.0 - float(val_metrics["success_rate"]),
                1.0 - float(val_metrics["wall_success_rate"]),
                float(val_metrics["avg_cost"]),
            )
            is_best = False
            if choose_better_result(candidate_score, best_score):
                best_score = candidate_score
                best_val_metrics = val_metrics
                best_state = {k: v.detach().cpu().clone() for k, v in controller.state_dict().items()}
                is_best = True
                if run_dir is not None:
                    # Preemption safety: persist the best-so-far weights and the
                    # training history immediately. A restarted job can salvage
                    # them; the final save replaces the partial file.
                    torch.save(best_state, run_dir / f"{save_tag}_controller.partial.pt")
                    merge_json(run_dir / "train_history_partial.json",
                               {save_tag: history + [record]})
            print_train_epoch_status(
                mode=mode,
                epoch=epoch,
                epochs=mode_epochs,
                record=record,
                val_metrics=val_metrics,
                is_best=is_best,
            )
        else:
            print_train_epoch_status(
                mode=mode,
                epoch=epoch,
                epochs=mode_epochs,
                record=record,
            )
        history.append(record)

    if best_state is None:
        raise RuntimeError(f"Training for mode {mode} did not produce any validation checkpoint.")
    controller.load_state_dict(best_state)
    return controller, plant_true, history, best_val_metrics


def setup_plot_style(plt) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.22,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


class _VariantColorMap(dict):
    """Color map that auto-assigns a distinct fallback color to any unknown key
    (e.g. context-ablation config labels) so every plot works without changes."""
    _FALLBACK = ["#0891b2", "#ca8a04", "#9333ea", "#dc2626", "#059669", "#1d4ed8",
                 "#db2777", "#65a30d", "#0f766e", "#ea580c", "#4f46e5", "#be123c"]

    def __missing__(self, key):
        color = self._FALLBACK[len(self) % len(self._FALLBACK)]
        self[key] = color
        return color


def variant_colors() -> dict[str, str]:
    return _VariantColorMap({
        "nominal": "#4b5563",
        "disturbance_only": "#d97706",
        "context": "#0f766e",
        "mad_context": "#2563eb",
        "mp_only_context": "#7c3aed",
        "context_no_lift": "#db2777",
        "contextual_ssm": "#16a34a",
    })


def reference_rollout_metrics(test_metrics: dict[str, dict]) -> dict:
    """Return a variant's metrics whose rollout supplies shared reference geometry
    (the wall-crossing index, identical across variants for a given scenario).

    Prefers 'context' for backward-compatible plots, then any other variant that
    was actually run — so plots still work when 'context' is not among --variants.
    """
    for pref in ("context", "contextual_ssm", "mad_context", "mp_only_context",
                 "context_no_lift", "disturbance_only", "nominal"):
        m = test_metrics.get(pref)
        if isinstance(m, dict) and "rollout" in m:
            return m
    for m in test_metrics.values():
        if isinstance(m, dict) and "rollout" in m:
            return m
    raise KeyError("No variant with a rollout is available for reference geometry.")


def plot_wall_style_summary(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_batch: ScenarioBatch,
    test_metrics: dict[str, dict],
    show_plots: bool,
) -> None:
    plt = get_plt(show_plots)
    setup_plot_style(plt)
    colors = variant_colors()
    labels = {mode: label for mode, label in variant_order}

    fig = plt.figure(figsize=(16.0, 12.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.0, 1.2])

    sample_idx = select_trajectory_indices(test_batch, args)[0]
    start = test_batch.start[sample_idx].numpy()
    start_x = float(start[0])
    start_y = float(start[1])
    gate_traj = test_batch.gate_y[sample_idx].numpy()
    gate_x_ref = np.linspace(start_x, 0.0, int(args.horizon))
    context_cross_idx = int(reference_rollout_metrics(test_metrics)["rollout"]["cross_idx"][sample_idx].item())
    gate_center = float(gate_traj[min(context_cross_idx, len(gate_traj) - 1)])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.step(
        gate_x_ref,
        gate_traj,
        where="post",
        color="grey",
        linestyle="--",
        alpha=0.6,
        linewidth=2.0,
        label="Gate schedule $g_t$",
    )
    ax1.fill_between(
        gate_x_ref,
        gate_traj - float(args.gate_half_width),
        gate_traj + float(args.gate_half_width),
        step="post",
        color="grey",
        alpha=0.12,
    )
    draw_wall(ax1, float(args.wall_x), gate_center, float(args.gate_half_width), float(args.corridor_limit))

    for mode, label in variant_order:
        traj = test_metrics[mode]["rollout"]["x_seq"][sample_idx, :, :2].numpy()
        traj_full = np.vstack([start, traj])
        lw = 2.6 if mode in ("context", "mad_context", "mp_only_context", "contextual_ssm") else 2.0
        ax1.plot(traj_full[:, 0], traj_full[:, 1], color=colors[mode], lw=lw, label=label)

    ax1.scatter([start_x], [start_y], color="#6b7280", s=36, zorder=4, label="Start")
    ax1.scatter([0.0], [0.0], color="#111827", marker="*", s=95, zorder=5, label="Goal (0,0)")
    ax1.set_title("Top-Down View: Robot Navigating the Corridor", fontsize=14, fontweight="bold")
    ax1.set_xlabel("x position", fontsize=12)
    ax1.set_ylabel("y position", fontsize=12)
    ax1.set_xlim(-0.15, max(float(args.start_x_max), start_x) + 0.15)
    ax1.set_ylim(-float(args.corridor_limit) - 0.1, float(args.corridor_limit) + 0.1)
    ax1.legend(loc="best", ncol=2)

    ax2 = fig.add_subplot(gs[1, 0])
    bar_labels = [labels[mode] for mode, _ in variant_order]
    success_rates = [100.0 * float(test_metrics[mode]["wall_success_rate"]) for mode, _ in variant_order]
    bar_colors = [colors[mode] for mode, _ in variant_order]
    ax2.bar(bar_labels, success_rates, color=bar_colors, alpha=0.82)
    ax2.set_title("Gate Crossing Success Rate (%)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0.0, 105.0)
    for idx, value in enumerate(success_rates):
        ax2.text(idx, value + 2.0, f"{value:.1f}%", ha="center", fontweight="bold")

    ax3 = fig.add_subplot(gs[1, 1])
    avg_miss = [float(test_metrics[mode]["avg_abs_cross_error"]) for mode, _ in variant_order]
    ax3.bar(bar_labels, avg_miss, color=bar_colors, alpha=0.82)
    ax3.set_title(
        f"Avg |y - g_t| at Wall (Safe threshold < {float(args.gate_half_width):.2f})",
        fontsize=12,
        fontweight="bold",
    )
    ax3.axhline(float(args.gate_half_width), color="red", linestyle="--", label="Safe bound")
    ax3.legend(loc="best")
    y_offset = max(0.03, 0.05 * max(avg_miss + [float(args.gate_half_width)]))
    for idx, value in enumerate(avg_miss):
        ax3.text(idx, value + y_offset, f"{value:.2f}", ha="center", fontweight="bold")

    ax4 = fig.add_subplot(gs[2, :])
    t = np.arange(1, int(args.horizon) + 1)
    ax4.step(
        t,
        gate_traj,
        where="post",
        color="grey",
        linestyle="--",
        alpha=0.7,
        linewidth=2.0,
        label="Gate center $g_t$",
    )
    ax4.fill_between(
        t,
        gate_traj - float(args.gate_half_width),
        gate_traj + float(args.gate_half_width),
        step="post",
        color="grey",
        alpha=0.12,
    )
    for mode, label in variant_order:
        y_seq = test_metrics[mode]["rollout"]["x_seq"][sample_idx, :, 1].numpy()
        cross_idx = int(test_metrics[mode]["rollout"]["cross_idx"][sample_idx].item()) + 1
        ax4.plot(t, y_seq, color=colors[mode], lw=2.2, label=f"{label} $y_t$")
        ax4.axvline(cross_idx, color=colors[mode], linestyle=":", alpha=0.45, linewidth=1.4)
    ax4.axhline(0.0, color="black", linewidth=1.0, alpha=0.45)
    ax4.set_title("Gate Switching Over Time For One Representative Episode", fontsize=14, fontweight="bold")
    ax4.set_xlabel("time step", fontsize=12)
    ax4.set_ylabel("lateral position / gate center", fontsize=12)
    ax4.legend(loc="best", ncol=2)

    plt.tight_layout()
    fig.savefig(run_dir / "wall_style_summary.png", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_loss_curves(
    *,
    run_dir: Path,
    histories: dict[str, list[dict]],
    variant_order: list[tuple[str, str]],
    show_plots: bool,
) -> None:
    plt = get_plt(show_plots)
    setup_plot_style(plt)
    colors = variant_colors()
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for mode, label in variant_order:
        history = histories.get(mode, [])
        if not history:
            continue
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_epochs = [h["epoch"] for h in history if "val_cost" in h]
        val_cost = [h["val_cost"] for h in history if "val_cost" in h]
        ax.plot(epochs, train_loss, color=colors[mode], alpha=0.28, lw=1.5)
        ax.plot(val_epochs, val_cost, color=colors[mode], lw=2.3, label=label)
    ax.set_title("Training / validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curves.png", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_control_magnitude(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_metrics: dict[str, dict],
    show_plots: bool,
) -> None:
    plt = get_plt(show_plots)
    setup_plot_style(plt)
    colors = variant_colors()
    t = np.arange(int(args.horizon))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for mode, label in variant_order:
        u_seq = test_metrics[mode]["rollout"]["u_seq"].numpy()
        u_mag = np.linalg.norm(u_seq, axis=-1)
        mean = np.mean(u_mag, axis=0)
        q10 = np.quantile(u_mag, 0.10, axis=0)
        q90 = np.quantile(u_mag, 0.90, axis=0)
        ax.plot(t, mean, color=colors[mode], lw=2.2, label=label)
        ax.fill_between(t, q10, q90, color=colors[mode], alpha=0.12)
    ax.set_title("Control magnitude over time")
    ax.set_xlabel("time step")
    ax.set_ylabel(r"$\|u_t\|_2$")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run_dir / "control_magnitude_over_time.png", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def select_trajectory_indices(batch: ScenarioBatch, args: argparse.Namespace) -> list[int]:
    gate = batch.gate_y.numpy()
    noise = batch.process_noise.numpy()
    strength = np.mean(np.abs(noise[..., 3]), axis=1)
    cross_gate = np.mean(gate[:, -12:], axis=1)
    idx_pos = int(np.argmax(np.where(cross_gate > 0, strength, -np.inf))) if np.any(cross_gate > 0) else 0
    idx_neg = int(np.argmax(np.where(cross_gate < 0, strength, -np.inf))) if np.any(cross_gate < 0) else idx_pos
    remaining = [i for i in range(gate.shape[0]) if i not in {idx_pos, idx_neg}]
    picks = [idx_pos, idx_neg]
    for idx in remaining[: max(0, int(args.sample_traj_count) - 2)]:
        picks.append(int(idx))
    return picks[: int(args.sample_traj_count)]


def draw_wall(ax, wall_x: float, gate_center: float, half_width: float, y_limit: float) -> None:
    ax.plot([wall_x, wall_x], [-y_limit, gate_center - half_width], color="black", lw=3.0)
    ax.plot([wall_x, wall_x], [gate_center + half_width, y_limit], color="black", lw=3.0)
    ax.plot([wall_x, wall_x], [gate_center - half_width, gate_center + half_width], color="white", lw=5.0)
    # Corridor walls
    ax.axhline(y_limit, color="#6b7280", lw=1.5, ls="-", zorder=1)
    ax.axhline(-y_limit, color="#6b7280", lw=1.5, ls="-", zorder=1)
    ax.axhspan(y_limit, y_limit * 2, color="#e5e7eb", alpha=0.6, zorder=0)
    ax.axhspan(-y_limit * 2, -y_limit, color="#e5e7eb", alpha=0.6, zorder=0)


def plot_trajectory_samples(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_batch: ScenarioBatch,
    test_metrics: dict[str, dict],
    show_plots: bool,
) -> None:
    plt = get_plt(show_plots)
    setup_plot_style(plt)
    colors = variant_colors()
    idxs = select_trajectory_indices(test_batch, args)
    n = len(idxs)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.8 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, idx in zip(axes_flat, idxs):
        ctx_roll = reference_rollout_metrics(test_metrics)["rollout"]
        cross_idx = int(ctx_roll["cross_idx"][idx].item())
        gate_center = float(test_batch.gate_y[idx, cross_idx].item())
        draw_wall(ax, float(args.wall_x), gate_center, float(args.gate_half_width), float(args.corridor_limit))
        for mode, label in variant_order:
            traj = test_metrics[mode]["rollout"]["x_seq"][idx, :, :2].numpy()
            start = test_batch.start[idx].numpy()
            traj_full = np.vstack([start, traj])
            ax.plot(traj_full[:, 0], traj_full[:, 1], color=colors[mode], lw=2.2, label=label)
        ax.scatter([test_batch.start[idx, 0].item()], [test_batch.start[idx, 1].item()], color="#6b7280", s=32, zorder=4)
        ax.scatter([0.0], [0.0], color="#111827", marker="*", s=90, zorder=5, label="Goal (0,0)")
        ax.set_title(f"Sample #{idx} | gate @ wall = {gate_center:+.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-0.25, float(args.start_x_max) + 0.15)
        ax.set_ylim(-float(args.corridor_limit) - 0.15, float(args.corridor_limit) + 0.15)
        ax.legend(loc="best")

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("Representative trajectory samples", y=0.99, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(run_dir / "trajectory_samples.png", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_waiting_behavior(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    test_batch: ScenarioBatch,
    test_metrics: dict[str, dict],
    show_plots: bool,
) -> None:
    """Visualise the 'waiting' strategy: controller slows forward/lateral motion after a recent gate switch."""
    if is_continuous_gate(args):
        print("Skipping waiting-behavior switch-age plot for continuous gate motion.")
        return

    plt = get_plt(show_plots)
    setup_plot_style(plt)

    roll      = reference_rollout_metrics(test_metrics)["rollout"]
    x_seq_t   = roll["x_seq"]                          # (B, T, 4)
    cross_idx = roll["cross_idx"]                       # (B,)
    B, T, _   = x_seq_t.shape

    vx        = x_seq_t[:, :, 2].numpy()               # forward velocity  (B, T)
    vy        = x_seq_t[:, :, 3].numpy()               # lateral velocity  (B, T)
    y_pos     = x_seq_t[:, :, 1].numpy()               # y position        (B, T)
    gate_np   = test_batch.gate_y.numpy()              # (B, T)
    sw_age    = test_batch.switch_age.numpy()          # (B, T)

    # switch_age at the wall-crossing step for each sample
    ci_np     = cross_idx.numpy().astype(int)
    ci_np     = np.clip(ci_np, 0, T - 1)
    age_at_cross = sw_age[np.arange(B), ci_np]         # (B,)

    dwell_min     = int(args.gate_dwell_min)
    recent_mask   = age_at_cross <= dwell_min           # gate switched recently
    committed_mask = ~recent_mask

    # ── align every sample to its crossing step ───────────────────────────────
    W_before, W_after = 28, 8
    W = W_before + W_after
    t_rel = np.arange(-W_before, W_after)

    def _aligned(arr2d: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return (n_valid, W) array of arr2d windows centred on crossing."""
        rows = []
        for i in np.where(mask)[0]:
            lo, hi = int(ci_np[i]) - W_before, int(ci_np[i]) + W_after
            if lo >= 0 and hi <= T:
                rows.append(arr2d[i, lo:hi])
        return np.array(rows) if rows else np.zeros((0, W))

    rec_vx     = _aligned(np.abs(vx),                  recent_mask)
    com_vx     = _aligned(np.abs(vx),                  committed_mask)
    rec_vy     = _aligned(np.abs(vy),                  recent_mask)
    com_vy     = _aligned(np.abs(vy),                  committed_mask)
    rec_err    = _aligned(np.abs(y_pos - gate_np),     recent_mask)
    com_err    = _aligned(np.abs(y_pos - gate_np),     committed_mask)

    # ── pick 3 representative individual traces per group ─────────────────────
    def _pick_examples(mask: np.ndarray, n: int = 3) -> list[int]:
        idxs = list(np.where(mask)[0])
        if not idxs:
            return []
        # sort by age_at_cross and pick spread: min / median / max
        idxs.sort(key=lambda i: age_at_cross[i])
        picks = []
        for pos in [0, len(idxs) // 2, len(idxs) - 1]:
            picks.append(idxs[pos])
        return list(dict.fromkeys(picks))[:n]

    rec_ex  = _pick_examples(recent_mask)
    com_ex  = _pick_examples(committed_mask)

    # ── colours ───────────────────────────────────────────────────────────────
    C_REC  = "#f97316"   # orange  — recent switch (uncertain)
    C_COM  = "#22d3ee"   # cyan    — committed (stable gate)
    C_ZERO = "#94a3b8"   # grey

    # ── figure: 4 rows (vx | vy | gate-error | examples) ─────────────────────
    fig = plt.figure(figsize=(14, 14))
    gs  = fig.add_gridspec(4, 2, hspace=0.52, wspace=0.32,
                           height_ratios=[1.0, 1.0, 1.0, 1.2])

    ax_vx    = fig.add_subplot(gs[0, :])   # full-width: forward speed |vx|
    ax_vy    = fig.add_subplot(gs[1, :])   # full-width: lateral speed |vy|
    ax_err   = fig.add_subplot(gs[2, :])   # full-width: |gate error|
    ax_rec   = fig.add_subplot(gs[3, 0])   # example trajectories — recent
    ax_com   = fig.add_subplot(gs[3, 1])   # example trajectories — committed

    def _style(ax_in):
        ax_in.axvline(0, color="#f1f5f9", lw=1.3, linestyle="--", alpha=0.7, zorder=3)
        ax_in.set_xlabel("steps relative to wall crossing")
        ax_in.tick_params(labelsize=9)

    def _plot_speed_panel(ax_in, rec_data, com_data, ylabel, title):
        for data, color, label in [
            (rec_data, C_REC, f"recent switch  (age \u2264 {dwell_min},  n={len(rec_data)})"),
            (com_data, C_COM, f"committed       (age > {dwell_min},  n={len(com_data)})"),
        ]:
            if data.shape[0] == 0:
                continue
            mu  = data.mean(axis=0)
            p_lo = np.percentile(data, 20, axis=0)
            p_hi = np.percentile(data, 80, axis=0)
            ax_in.fill_between(t_rel, p_lo, p_hi, color=color, alpha=0.18)
            ax_in.plot(t_rel, mu, color=color, lw=2.4, label=label)
            for row in data[::max(1, len(data) // 6)]:
                ax_in.plot(t_rel, row, color=color, lw=0.7, alpha=0.25)
        ax_in.axvspan(-dwell_min, 0, color=C_REC, alpha=0.06,
                      label=f"gate-dwell window ({dwell_min} steps)")
        ax_in.set_ylabel(ylabel)
        ax_in.set_title(title, fontweight="bold")
        ax_in.legend(fontsize=9, loc="best")
        _style(ax_in)

    # ── Panel 1: forward speed |vx| — primary "waiting" signal ───────────────
    _plot_speed_panel(
        ax_vx, rec_vx, com_vx,
        ylabel=r"$|v_x|$  forward speed",
        title=r"Forward speed $|v_x|$ around wall crossing — slowing = buying time",
    )

    # ── Panel 2: lateral speed |vy| ───────────────────────────────────────────
    _plot_speed_panel(
        ax_vy, rec_vy, com_vy,
        ylabel=r"$|v_y|$  lateral speed",
        title=r"Lateral speed $|v_y|$ around wall crossing — repositioning",
    )

    # ── Panel 3: |gate error| ─────────────────────────────────────────────────
    for data, color in [(rec_err, C_REC), (com_err, C_COM)]:
        if data.shape[0] == 0:
            continue
        mu   = data.mean(axis=0)
        p_lo = np.percentile(data, 20, axis=0)
        p_hi = np.percentile(data, 80, axis=0)
        ax_err.fill_between(t_rel, p_lo, p_hi, color=color, alpha=0.18)
        ax_err.plot(t_rel, mu, color=color, lw=2.4)

    ax_err.axhline(float(args.gate_half_width), color="#ef4444", lw=1.3,
                   linestyle=":", label=f"gate half-width {args.gate_half_width:.2f}")
    ax_err.set_ylabel(r"$|y - g_t|$  gate error")
    ax_err.set_title("|y \u2212 gate| around wall crossing", fontweight="bold")
    ax_err.legend(fontsize=9, loc="best")
    _style(ax_err)

    # ── Panels 4 & 5: individual example trajectories ────────────────────────
    for ax_ex, examples, color, title in [
        (ax_rec, rec_ex,  C_REC, f"Recent-switch examples  (age \u2264 {dwell_min})"),
        (ax_com, com_ex,  C_COM, f"Committed examples       (age > {dwell_min})"),
    ]:
        for idx in examples:
            ci   = int(ci_np[idx])
            t_ax = np.arange(T)
            g    = gate_np[idx]
            y    = y_pos[idx]

            ax_ex.fill_between(t_ax, g - float(args.gate_half_width),
                               g + float(args.gate_half_width),
                               step="post", color=C_ZERO, alpha=0.10)
            ax_ex.step(t_ax, g, where="post", color=C_ZERO, lw=1.2,
                       alpha=0.55, linestyle="--")
            ax_ex.plot(t_ax, y, color=color, lw=1.8, alpha=0.85)
            ax_ex.axvline(ci, color=color, lw=1.0, linestyle=":", alpha=0.6)

            # mark switch events as dots on the gate line
            switches = np.where(np.abs(np.diff(g)) > 1e-4)[0]
            ax_ex.scatter(switches, g[switches], color="#fbbf24", s=22, zorder=4)

            # annotate switch_age at crossing
            ax_ex.text(ci + 0.5, float(y[ci]) + 0.05,
                       f"age={int(age_at_cross[idx])}",
                       fontsize=7, color=color, alpha=0.85)

        ax_ex.axhline(0, color=C_ZERO, lw=0.7, alpha=0.4)
        ax_ex.set_xlim(0, T - 1)
        ax_ex.set_ylim(-float(args.corridor_limit) - 0.1, float(args.corridor_limit) + 0.1)
        ax_ex.set_xlabel("time step")
        ax_ex.set_ylabel("y position")
        ax_ex.set_title(title, fontweight="bold", fontsize=10)
        ax_ex.scatter([], [], color="#fbbf24", s=22, label="gate switch event")
        ax_ex.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Waiting-behaviour analysis: does the controller hold back after a recent gate switch?",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(run_dir / "waiting_behavior.png", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def _animate_one_sample(
    *,
    plt,
    args: argparse.Namespace,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_batch: ScenarioBatch,
    test_metrics: dict[str, dict],
    sample_idx: int,
    file_tag: str,
    show_plots: bool,
) -> None:
    """Render and save one animated GIF for the given sample index."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle
    import matplotlib.patheffects as pe

    colors  = variant_colors()
    horizon = int(args.horizon)
    wall_x  = float(args.wall_x)
    half_w  = float(args.gate_half_width)
    corr    = float(args.corridor_limit)
    x_max   = float(args.start_x_max) + 0.15

    gate_traj = test_batch.gate_y[sample_idx].numpy()
    start_np  = test_batch.start[sample_idx].numpy()

    # per-variant trajectories: prepend start -> (T+1, 2)
    trajs: dict[str, np.ndarray] = {}
    for mode, _ in variant_order:
        xy = test_metrics[mode]["rollout"]["x_seq"][sample_idx, :, :2].numpy()
        trajs[mode] = np.vstack([start_np, xy])

    # outcome labels shown in title
    def _outcome(mode: str) -> str:
        roll   = test_metrics[mode]["rollout"]
        ci     = int(roll["cross_idx"][sample_idx].item())
        y_cross = float(trajs[mode][ci + 1, 1])
        g_cross = float(gate_traj[ci])
        hit     = abs(y_cross - g_cross) <= half_w
        x_term  = float(trajs[mode][-1, 0])
        y_term  = float(trajs[mode][-1, 1])
        goal_ok = (x_term ** 2 + y_term ** 2) ** 0.5 < float(args.goal_tol)
        gate_sym = "[G:pass]" if hit else "[G:fail]"
        goal_sym = "[O:pass]" if goal_ok else "[O:fail]"
        return f"{gate_sym} {goal_sym}"

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8))
    gs  = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.38)
    ax_arena = fig.add_subplot(gs[0])
    ax_time  = fig.add_subplot(gs[1])

    fig.patch.set_facecolor("#0f172a")
    for ax in (ax_arena, ax_time):
        ax.set_facecolor("#1e293b")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.title.set_color("#e2e8f0")
        ax.grid(color="#334155", linewidth=0.6, alpha=0.5)

    # ── static arena elements ─────────────────────────────────────────────────
    ax_arena.set_xlim(-0.15, x_max)
    ax_arena.set_ylim(-corr - 0.12, corr + 0.12)
    ax_arena.axhline(corr, color="#475569", lw=1.5, ls="-", zorder=1)
    ax_arena.axhline(-corr, color="#475569", lw=1.5, ls="-", zorder=1)
    ax_arena.axhspan(corr, corr + 0.5, color="#334155", alpha=0.5, zorder=0)
    ax_arena.axhspan(-corr - 0.5, -corr, color="#334155", alpha=0.5, zorder=0)
    ax_arena.set_xlabel("x position")
    ax_arena.set_ylabel("y position")
    ax_arena.scatter([0.0], [0.0], color="#f8fafc", marker="*", s=160, zorder=6,
                     label="Goal",
                     path_effects=[pe.withStroke(linewidth=2, foreground="#0f172a")])
    ax_arena.scatter([start_np[0]], [start_np[1]], color="#94a3b8", s=55,
                     zorder=5, marker="o", label="Start")

    for mode, _ in variant_order:
        ax_arena.plot(trajs[mode][:, 0], trajs[mode][:, 1],
                      color=colors[mode], lw=1.2, alpha=0.15, zorder=2)

    trail_lines: dict[str, object] = {}
    dots: dict[str, object] = {}
    for mode, lbl in variant_order:
        lw = 2.8 if mode in ("context", "mad_context", "mp_only_context", "contextual_ssm") else 1.8
        outcome = _outcome(mode)
        ln, = ax_arena.plot([], [], color=colors[mode], lw=lw, zorder=4,
                            label=f"{lbl}  {outcome}",
                            path_effects=[pe.withStroke(linewidth=lw + 1.5,
                                                        foreground="#0f172a")])
        dot, = ax_arena.plot([], [], "o", color=colors[mode], ms=9, zorder=7,
                             path_effects=[pe.withStroke(linewidth=2.5,
                                                         foreground="#0f172a")])
        trail_lines[mode] = ln
        dots[mode]        = dot

    # wall: two solid rects + transparent gate opening rect (updated each frame)
    rect_upper = Rectangle((wall_x - 0.015, 0.0),    0.030, corr + 0.12,
                            color="#f1f5f9", zorder=3)
    rect_lower = Rectangle((wall_x - 0.015, -corr - 0.12), 0.030, corr + 0.12,
                            color="#f1f5f9", zorder=3)
    gate_patch = Rectangle((wall_x - 0.015, 0.0),    0.030, 2 * half_w,
                            color="#1e293b", zorder=4)
    for p in (rect_upper, rect_lower, gate_patch):
        ax_arena.add_patch(p)

    step_text = ax_arena.text(
        0.02, 0.97, "", transform=ax_arena.transAxes,
        color="#e2e8f0", fontsize=11, fontweight="bold", va="top", ha="left",
        path_effects=[pe.withStroke(linewidth=2, foreground="#0f172a")],
    )

    ax_arena.legend(loc="best", fontsize=8.5, facecolor="#1e293b",
                    edgecolor="#475569", labelcolor="#e2e8f0", framealpha=0.85)
    ax_arena.set_title(
        f"Gate-Crossing Navigation — Sample #{sample_idx}",
        fontsize=13, fontweight="bold", pad=8,
    )

    # ── time-series panel ─────────────────────────────────────────────────────
    t_ax = np.arange(horizon)
    ax_time.step(t_ax, gate_traj, where="post",
                 color="#94a3b8", lw=1.8, alpha=0.7, label="Gate $g_t$")
    ax_time.fill_between(t_ax, gate_traj - half_w, gate_traj + half_w,
                         step="post", color="#94a3b8", alpha=0.12)
    ax_time.axhline(0, color="#475569", lw=0.8, alpha=0.5)

    y_lines: dict[str, tuple] = {}
    for mode, lbl in variant_order:
        y_seq = trajs[mode][1:, 1]
        lw = 2.4 if mode in ("context", "mad_context", "mp_only_context", "contextual_ssm") else 1.6
        ln, = ax_time.plot([], [], color=colors[mode], lw=lw, label=lbl)
        y_lines[mode] = (ln, y_seq)

    vline = ax_time.axvline(0, color="#f8fafc", lw=1.2, alpha=0.7, linestyle="--")
    ax_time.set_xlim(0, horizon - 1)
    ax_time.set_ylim(-corr - 0.1, corr + 0.1)
    ax_time.set_xlabel("time step")
    ax_time.set_ylabel("y / gate center")
    ax_time.set_title("Lateral position over time", fontsize=11, fontweight="bold")
    ax_time.legend(loc="best", fontsize=8, facecolor="#1e293b",
                   edgecolor="#475569", labelcolor="#e2e8f0", framealpha=0.85)

    # ── update function (closed over per-sample data) ─────────────────────────
    def _update(frame: int):
        ti = frame + 1
        g  = float(gate_traj[frame])

        rect_upper.set_y(g + half_w)
        rect_upper.set_height(max(0.0, corr + 0.12 - (g + half_w)))
        rect_lower.set_y(-corr - 0.12)
        rect_lower.set_height(max(0.0, (g - half_w) + corr + 0.12))
        gate_patch.set_y(g - half_w)

        for mode, _ in variant_order:
            xy = trajs[mode][:ti + 1]
            trail_lines[mode].set_data(xy[:, 0], xy[:, 1])
            dots[mode].set_data([xy[-1, 0]], [xy[-1, 1]])

        step_text.set_text(f"step {frame:03d}/{horizon - 1:03d}")
        vline.set_xdata([frame, frame])
        for mode, (ln, y_seq) in y_lines.items():
            ln.set_data(t_ax[:frame + 1], y_seq[:frame + 1])

        return (rect_upper, rect_lower, gate_patch, step_text, vline,
                *trail_lines.values(), *dots.values(),
                *[ln for ln, _ in y_lines.values()])

    frames      = list(range(0, horizon, 2))
    interval_ms = 60
    anim = FuncAnimation(fig, _update, frames=frames,
                         interval=interval_ms, blit=True)

    gif_path = run_dir / f"rollout_animation_{file_tag}.gif"
    anim.save(str(gif_path), writer=PillowWriter(fps=1000 // interval_ms))
    print(f"  Animation saved -> {gif_path}")

    try:
        from matplotlib.animation import FFMpegWriter
        mp4_path = run_dir / f"rollout_animation_{file_tag}.mp4"
        anim.save(str(mp4_path),
                  writer=FFMpegWriter(fps=1000 // interval_ms, bitrate=1200))
        print(f"  Animation saved -> {mp4_path}")
    except Exception:
        pass

    if show_plots:
        plt.show()
    plt.close(fig)


def animate_rollout(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_batch: ScenarioBatch,
    test_metrics: dict[str, dict],
    show_plots: bool,
    n_samples: int = 5,
) -> None:
    """Render per-sample animated GIFs for n_samples representative episodes."""
    plt = get_plt(show_plots)
    setup_plot_style(plt)

    # collect indices: use select_trajectory_indices, pad with random if needed
    idxs = select_trajectory_indices(test_batch, args)
    batch_size = int(test_batch.gate_y.shape[0])
    if len(idxs) < n_samples:
        used = set(idxs)
        extras = [i for i in range(batch_size) if i not in used]
        idxs = list(idxs) + extras[: n_samples - len(idxs)]
    idxs = idxs[:n_samples]

    print(f"\nRendering {len(idxs)} rollout animations…")
    for rank, sample_idx in enumerate(idxs):
        print(f"  [{rank + 1}/{len(idxs)}] sample #{sample_idx}")
        _animate_one_sample(
            plt=plt,
            args=args,
            run_dir=run_dir,
            variant_order=variant_order,
            test_batch=test_batch,
            test_metrics=test_metrics,
            sample_idx=sample_idx,
            file_tag=f"{rank + 1:02d}_idx{sample_idx}",
            show_plots=show_plots,
        )


def animate_adversarial_sample(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_batch: ScenarioBatch,
    test_metrics: dict[str, dict],
    show_plots: bool,
) -> None:
    """Render an animated GIF for one adversarial episode (late gate switch)."""
    adv_mask = test_batch.is_adversarial.numpy().astype(bool)
    adv_indices = np.where(adv_mask)[0]
    if len(adv_indices) == 0:
        print("  No adversarial episodes found — skipping adversarial animation.")
        return

    plt = get_plt(show_plots)
    setup_plot_style(plt)

    # Pick the adversarial sample with the best (lowest) final position error
    # for the first available context variant, so the GIF shows informative behaviour.
    primary_mode = next(
        (m for m, _ in variant_order if m in ("context", "mad_context", "mp_only_context", "contextual_ssm", "disturbance_only")),
        variant_order[0][0],
    )
    x_seq = test_metrics[primary_mode]["rollout"]["x_seq"]  # (B, T, 4)
    goal = test_batch.goal  # (B, 2)
    pos_final = x_seq[adv_indices, -1, :2]  # (n_adv, 2)
    goal_adv = goal[adv_indices, :2]
    err = torch.norm(pos_final - goal_adv.to(pos_final.device), dim=-1)
    best_local = int(err.argmin().item())
    sample_idx = int(adv_indices[best_local])

    print(f"\nRendering adversarial animation for sample #{sample_idx}…")
    _animate_one_sample(
        plt=plt,
        args=args,
        run_dir=run_dir,
        variant_order=variant_order,
        test_batch=test_batch,
        test_metrics=test_metrics,
        sample_idx=sample_idx,
        file_tag=f"adversarial_idx{sample_idx}",
        show_plots=show_plots,
    )


def plot_adversarial_switching(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_batch: ScenarioBatch,
    test_metrics: dict[str, dict],
    show_plots: bool,
) -> None:
    """Compare performance under adversarial (late gate switch) vs. stable gate episodes.

    An episode is labelled adversarial when the gate switch_age at the wall-crossing
    step is <= gate_dwell_min, meaning a switch occurred very close to the freeze step.
    """
    if is_continuous_gate(args):
        print("Skipping adversarial-switching plot for continuous gate motion.")
        return

    plt = get_plt(show_plots)
    setup_plot_style(plt)
    colors = variant_colors()
    labels = {mode: label for mode, label in variant_order}

    gate_np   = test_batch.gate_y.numpy()           # (B, T)
    B, T      = gate_np.shape

    # Use the ground-truth adversarial flag stored during batch generation.
    adv_mask    = test_batch.is_adversarial.numpy().astype(bool)  # (B,)
    stable_mask = ~adv_mask
    n_adv, n_stable = int(adv_mask.sum()), int(stable_mask.sum())

    # ── per-variant cross errors split by regime ──────────────────────────────
    pb_modes = [(m, l) for m, l in variant_order if m != "nominal"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    ax_bar, ax_err_adv, ax_traj = axes

    # Panel 1: success rate under each regime per variant
    bar_width = 0.35
    x_pos = np.arange(len(pb_modes))
    adv_succ, stable_succ = [], []
    for mode, _ in pb_modes:
        roll = test_metrics[mode]["rollout"]
        cross_idx = roll["cross_idx"].numpy().astype(int)
        y_cross = test_metrics[mode]["rollout"]["x_seq"][:, :, 1]
        g_cross = torch.from_numpy(gate_np)[torch.arange(B), torch.from_numpy(cross_idx)]
        err = (y_cross[torch.arange(B), torch.from_numpy(cross_idx)] - g_cross).abs().numpy()
        pos = test_metrics[mode]["rollout"]["x_seq"][:, -1, :2].numpy()
        goal_dist = np.linalg.norm(pos, axis=-1)
        wall_ok  = err <= float(args.gate_half_width)
        goal_ok  = goal_dist < float(args.goal_tol)
        success  = wall_ok & goal_ok
        adv_succ.append(float(success[adv_mask].mean()) if adv_mask.any() else 0.0)
        stable_succ.append(float(success[stable_mask].mean()) if stable_mask.any() else 0.0)

    bar_colors = [colors[m] for m, _ in pb_modes]
    ax_bar.bar(x_pos - bar_width / 2, [100 * v for v in adv_succ],
               bar_width, color=bar_colors, alpha=0.55,
               label=f"Adversarial (n={n_adv})", hatch="//")
    ax_bar.bar(x_pos + bar_width / 2, [100 * v for v in stable_succ],
               bar_width, color=bar_colors, alpha=0.85,
               label=f"Stable (n={n_stable})")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([labels[m] for m, _ in pb_modes], rotation=12, ha="right", fontsize=8)
    ax_bar.set_ylabel("Success rate (%)")
    ax_bar.set_title("Success rate: adversarial vs. stable gate", fontweight="bold")
    ax_bar.set_ylim(0, 108)
    ax_bar.legend(fontsize=8, loc="best")

    # Panel 2: distribution of |cross_error| for adversarial episodes
    err_data, bar_labels_adv = [], []
    for mode, label in pb_modes:
        roll = test_metrics[mode]["rollout"]
        cross_idx = roll["cross_idx"].numpy().astype(int)
        y_cross_t = roll["x_seq"][:, :, 1]
        g_cross = torch.from_numpy(gate_np)[torch.arange(B), torch.from_numpy(cross_idx)]
        err = (y_cross_t[torch.arange(B), torch.from_numpy(cross_idx)] - g_cross).abs().numpy()
        err_data.append(err[adv_mask])
        bar_labels_adv.append(label)
    if n_adv == 0 or any(len(d) == 0 for d in err_data):
        ax_err_adv.text(0.5, 0.5, "No adversarial episodes in test batch",
                        ha="center", va="center", transform=ax_err_adv.transAxes, fontsize=9)
    else:
        vp = ax_err_adv.violinplot(err_data, positions=range(len(pb_modes)),
                                    showmedians=True, showextrema=False)
        for body, (mode, _) in zip(vp["bodies"], pb_modes):
            body.set_facecolor(colors[mode])
            body.set_alpha(0.65)
    ax_err_adv.axhline(float(args.gate_half_width), color="#ef4444", lw=1.5,
                       linestyle="--", label=f"half-width {args.gate_half_width:.2f}")
    ax_err_adv.set_xticks(range(len(pb_modes)))
    ax_err_adv.set_xticklabels([labels[m] for m, _ in pb_modes],
                                 rotation=12, ha="right", fontsize=8)
    ax_err_adv.set_ylabel(r"$|y_{t^\star} - g_{t^\star}|$")
    ax_err_adv.set_title(f"|Cross error| under adversarial switching (n={n_adv})",
                          fontweight="bold")
    ax_err_adv.legend(fontsize=8, loc="best")

    # Panel 3: example top-down trajectories for the 3 hardest adversarial episodes
    # (smallest switch_age => most recent switch)
    ref_mode = "context" if "context" in test_metrics else variant_order[-1][0]
    ci_np = test_metrics[ref_mode]["rollout"]["cross_idx"].numpy().astype(int)
    ci_np = np.clip(ci_np, 0, T - 1)

    adv_indices = np.where(adv_mask)[0]
    if len(adv_indices) > 0:
        adv_indices = adv_indices[:3]  # just take first 3 adversarial episodes
        for idx in adv_indices:
            ci = int(ci_np[idx])
            gate_center = float(gate_np[idx, ci])
            # draw wall opening
            ax_traj.plot([float(args.wall_x), float(args.wall_x)],
                         [-float(args.corridor_limit), gate_center - float(args.gate_half_width)],
                         color="black", lw=2.5)
            ax_traj.plot([float(args.wall_x), float(args.wall_x)],
                         [gate_center + float(args.gate_half_width), float(args.corridor_limit)],
                         color="black", lw=2.5)
            for mode, label in variant_order:
                if mode == "nominal":
                    continue
                xy = test_metrics[mode]["rollout"]["x_seq"][idx, :, :2].numpy()
                start = test_batch.start[idx].numpy()
                traj = np.vstack([start, xy])
                lw = 2.2 if mode in ("context", "mad_context", "mp_only_context", "contextual_ssm") else 1.4
                ax_traj.plot(traj[:, 0], traj[:, 1], color=colors[mode], lw=lw,
                             alpha=0.8, label=label if idx == int(adv_indices[0]) else "")
        corr_adv = float(args.corridor_limit)
        ax_traj.axhline(corr_adv, color="#6b7280", lw=1.5, ls="-", zorder=1)
        ax_traj.axhline(-corr_adv, color="#6b7280", lw=1.5, ls="-", zorder=1)
        ax_traj.axhspan(corr_adv, corr_adv * 2, color="#e5e7eb", alpha=0.6, zorder=0)
        ax_traj.axhspan(-corr_adv * 2, -corr_adv, color="#e5e7eb", alpha=0.6, zorder=0)
        ax_traj.set_xlim(-0.15, float(args.start_x_max) + 0.15)
        ax_traj.set_ylim(-corr_adv - 0.1, corr_adv + 0.1)
        ax_traj.set_xlabel("x position")
        ax_traj.set_ylabel("y position")
        ax_traj.set_title("Trajectories: adversarial episodes", fontweight="bold")
        ax_traj.legend(loc="best", fontsize=8)
    else:
        ax_traj.text(0.5, 0.5, "No adversarial episodes found",
                     ha="center", va="center", transform=ax_traj.transAxes)

    fig.suptitle(
        f"Adversarial gate switching analysis  (n_adv={n_adv}, n_stable={n_stable})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(run_dir / "adversarial_switching.png", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_sample_trajectory(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_batch: ScenarioBatch,
    test_metrics: dict[str, dict],
    expected_cross_index: int,
    show_plots: bool = False,
    sample_idx: int = 0,
) -> None:
    """
    Two-panel paper figure for a single episode:
      Left  – top-down view (x vs y): trajectory per variant + wall + gate gap.
      Right – time series: y_t and gate band g_t±h vs step, with wall crossing marked.
    Variants are overlaid in their canonical colours.
    """
    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    setup_plot_style(plt)
    colors = variant_colors()
    labels = {mode: label for mode, label in variant_order}
    h = float(args.gate_half_width)
    wall_x = float(args.wall_x)
    T = int(args.horizon)
    T_plot = int(args.plot_horizon) if getattr(args, "use_plot_horizon", True) and getattr(args, "plot_horizon", None) else T
    T_plot = max(T_plot, T)
    freeze_step = max(1, expected_cross_index - int(args.gate_settle_steps))
    steps = np.arange(T_plot)

    # Gate trace for the selected episode; hold the last value only for plot extension.
    gate_np_ctrl = test_batch.gate_y[sample_idx].numpy()  # (T,)
    gate_np = np.concatenate([gate_np_ctrl,
                               np.full(T_plot - T, gate_np_ctrl[-1])]) if T_plot > T else gate_np_ctrl

    # Helper: extend a trajectory past the control horizon using zero-input nominal dynamics
    nominal_plant_ext = DoubleIntegratorNominal(
        dt=float(args.dt), pre_kp=float(args.pre_kp), pre_kd=float(args.pre_kd),
        drag_coeff=float(args.drag_coeff),
    )

    def extend_traj(xy: np.ndarray) -> np.ndarray:
        """xy: (T, 2) — extend to (T_plot, 2) with u=0 nominal rollout."""
        if T_plot <= T:
            return xy
        x = torch.tensor(xy[-1], dtype=torch.float32).view(1, 1, 2)
        # Pad velocity to 4-D state — assume zero velocity at T (conservative)
        x4 = torch.zeros(1, 1, 4, dtype=torch.float32)
        x4[..., :2] = x
        tail = [xy[-1]]
        u_zero = torch.zeros(1, 1, 2, dtype=torch.float32)
        for _ in range(T_plot - T):
            x4 = nominal_plant_ext.nominal_dynamics(x4, u_zero)
            tail.append(x4[0, 0, :2].numpy())
        return np.concatenate([xy, np.stack(tail[1:])], axis=0)

    fig, (ax_top, ax_ts) = plt.subplots(
        1, 2, figsize=(11, 4.2),
        gridspec_kw={"width_ratios": [1, 1.6]},
    )

    # ── Left: top-down view ───────────────────────────────────────────────────
    # Corridor walls
    y_lim = float(args.corridor_limit) * 1.05
    corr = float(args.corridor_limit)
    ax_top.axhline(corr, color="#6b7280", lw=1.5, ls="-", zorder=1)
    ax_top.axhline(-corr, color="#6b7280", lw=1.5, ls="-", zorder=1)
    ax_top.axhspan(corr, y_lim, color="#e5e7eb", alpha=0.6, zorder=0)
    ax_top.axhspan(-y_lim, -corr, color="#e5e7eb", alpha=0.6, zorder=0)
    # Transverse wall
    ax_top.axvline(wall_x, color="#ef4444", lw=2.0, zorder=3, label="Wall")

    # Gate opening at crossing time (use context variant if available, else first variant)
    ref_mode = "context" if "context" in test_metrics else variant_order[0][0]
    ref_cross_idx = int(test_metrics[ref_mode]["rollout"]["cross_idx"][sample_idx].item())
    g_cross = float(gate_np[min(ref_cross_idx, T - 1)])
    ax_top.fill_betweenx(
        [g_cross - h, g_cross + h],
        wall_x - 0.02, wall_x + 0.02,
        color="#bbf7d0", zorder=4, label="Gate opening",
    )
    # Wall above and below gate
    ax_top.fill_betweenx([g_cross + h, y_lim], wall_x - 0.01, wall_x + 0.01,
                         color="#fca5a5", zorder=4, alpha=0.7)
    ax_top.fill_betweenx([-y_lim, g_cross - h], wall_x - 0.01, wall_x + 0.01,
                         color="#fca5a5", zorder=4, alpha=0.7)

    for mode, label in variant_order:
        if mode not in test_metrics:
            continue
        traj = test_metrics[mode]["rollout"]["x_seq"][sample_idx, :, :2].numpy()  # (T, 2)
        traj = extend_traj(traj)
        x_traj, y_traj = traj[:, 0], traj[:, 1]
        # Colour trajectory by time
        pts = np.stack([x_traj, y_traj], axis=1)[np.newaxis]  # (1, T, 2)
        segs = np.concatenate([pts[:, :-1], pts[:, 1:]], axis=0).transpose(1, 0, 2)
        lc = LineCollection(segs, cmap="viridis", linewidth=2.0, alpha=0.85, zorder=5)  # noqa
        lc.set_array(np.linspace(0, 1, len(segs)))
        ax_top.add_collection(lc)
        # Mark start and end
        ax_top.scatter(x_traj[0], y_traj[0], s=40, color=colors.get(mode, "#888"),
                       zorder=6, marker="o")
        ax_top.scatter(x_traj[-1], y_traj[-1], s=40, color=colors.get(mode, "#888"),
                       zorder=6, marker="x")

    ax_top.set_xlim(float(args.start_x_max) * 1.05, -0.15)
    ax_top.set_ylim(-y_lim, y_lim)
    ax_top.set_xlabel("x  (m)")
    ax_top.set_ylabel("y  (m)")
    ax_top.set_title("Top-down trajectory", fontweight="bold")
    # Dummy handles for legend
    handles = [Line2D([0], [0], color=colors.get(m, "#888"), lw=2, label=labels[m])
               for m, _ in variant_order if m in test_metrics]
    handles += [Line2D([0], [0], color="#ef4444", lw=2, label="Wall"),
                plt.Rectangle((0, 0), 1, 1, fc="#bbf7d0", label="Gate")]
    ax_top.legend(handles=handles, fontsize=8, loc="best")
    ax_top.invert_xaxis()

    # ── Right: time series ────────────────────────────────────────────────────
    # Gate band
    ax_ts.fill_between(steps, gate_np - h, gate_np + h,
                       color="#bbf7d0", alpha=0.45, label="Gate opening", zorder=1)
    ax_ts.plot(steps, gate_np, color="#16a34a", lw=1.2, ls="--", label="Gate centre $g_t$", zorder=2)

    for mode, label in variant_order:
        if mode not in test_metrics:
            continue
        y_seq_ctrl = test_metrics[mode]["rollout"]["x_seq"][sample_idx, :, 1].numpy()
        traj_ext = extend_traj(test_metrics[mode]["rollout"]["x_seq"][sample_idx, :, :2].numpy())
        y_seq = traj_ext[:, 1]
        cross_idx = int(test_metrics[mode]["rollout"]["cross_idx"][sample_idx].item())
        c = colors.get(mode, "#888")
        # Solid line during control horizon, dashed during extension
        ax_ts.plot(steps[:T], y_seq[:T], color=c, lw=1.8, label=label, zorder=3)
        if T_plot > T:
            ax_ts.plot(steps[T - 1:], y_seq[T - 1:], color=c, lw=1.4, ls="--", zorder=3)
        ax_ts.scatter(cross_idx, y_seq_ctrl[min(cross_idx, T - 1)],
                      color=c, s=55, zorder=5, marker="D")

    # Switching-gate runs have a pre-crossing freeze; continuous OU runs do not.
    if not is_continuous_gate(args):
        ax_ts.axvline(freeze_step, color="#7c3aed", lw=1.4, ls=":", zorder=4,
                      label=f"$t_{{\\mathrm{{freeze}}}}={freeze_step}$")
    ax_ts.axvline(expected_cross_index, color="#ef4444", lw=1.4, ls=":", zorder=4,
                  label=f"$t_{{\\mathrm{{wall}}}}={expected_cross_index}$")

    if T_plot > T:
        ax_ts.axvline(T, color="#94a3b8", lw=1.2, ls="--", zorder=2,
                      label=f"Control horizon $T={T}$")
    ax_ts.axhline(0, color="#94a3b8", lw=0.7, ls="-", zorder=0)
    ax_ts.set_xlabel("Step $t$")
    ax_ts.set_ylabel("$y_t$  (m)")
    ax_ts.set_title("Lateral position vs. gate centre", fontweight="bold")
    ax_ts.legend(fontsize=8, loc="best")

    fig.suptitle(f"Sample episode #{sample_idx}", fontsize=12, fontweight="bold")
    fig.tight_layout()

    out = run_dir / "sample_trajectory.pdf"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Saved sample trajectory figure -> {out}")
    if show_plots:
        plt.show()
    plt.close(fig)


def _storyboard_ref_mode(test_metrics: dict, variant_order: list[tuple[str, str]]) -> str:
    """Reference variant for the storyboards: prefer a context-bearing controller
    (its crossing time anchors the snapshots and it should be the showcased,
    non-colliding trajectory), falling back to the first plotted variant."""
    for mode in ("context", "contextual_ssm", "mad_context", "mp_only_context", "context_no_lift"):
        if mode in test_metrics:
            return mode
    return variant_order[0][0]


def _storyboard_pick_episode(
    *,
    test_batch: ScenarioBatch,
    test_metrics: dict,
    ref_mode: str,
    sample_idx: int,
    continuous_gate: bool,
    analysis_end: int,
) -> int:
    """Showcase episode for the storyboards.

    Preference order: the reference controller SUCCEEDS on it; else at least
    does not hit the wall; else any episode. Switching-gate runs additionally
    require >= 3 gate switches so the panels show visibly different gate
    levels. Falls back to legacy behavior when per-episode outcomes are not
    stored (runs evaluated with older code)."""
    roll = test_metrics[ref_mode]["rollout"]
    n = int(test_batch.gate_y.shape[0])

    def rich_schedule(i: int) -> bool:
        if continuous_gate:
            return True
        g = test_batch.gate_y[i].numpy()
        return int(np.sum(np.abs(np.diff(g[:analysis_end])) > 0.05)) >= 3

    tiers = []
    if "success" in roll:
        tiers.append(lambda i: rich_schedule(i) and bool(roll["success"][i]))
    if "collided" in roll:
        tiers.append(lambda i: rich_schedule(i) and not bool(roll["collided"][i]))
    tiers.append(rich_schedule)
    for ok in tiers:
        if ok(sample_idx):
            return sample_idx
        for i in range(n):
            if ok(i):
                return i
    return sample_idx


def _storyboard_snapshot_steps(
    *,
    gate_np: np.ndarray,
    cross_idx: int,
    analysis_end: int,
    T: int,
    continuous_gate: bool,
) -> tuple[list[int], int]:
    """Chronologically ordered snapshot steps plus the crossing step.

    Three pre-crossing samples (gate-level mids for switching runs; quartiles
    of the episode's OWN crossing time for continuous runs), the crossing
    step, and the final step — sorted and de-duplicated. Sorting matters:
    with diverse/OOD starts an episode can cross the wall earlier than the
    schedule-derived sample points, which previously put the crossing panel
    out of time order."""
    cross_step = int(np.clip(cross_idx, 1, T - 1))
    if continuous_gate:
        base = max(cross_step, 4)
        pre = [max(1, base // 4), max(2, base // 2), max(3, (3 * base) // 4)]
    else:
        switch_pts = np.where(np.abs(np.diff(gate_np[:analysis_end])) > 0.05)[0] + 1
        seg_starts = np.concatenate([[0], switch_pts])
        seg_ends = np.concatenate([switch_pts, [analysis_end]])
        mids = [int(s + max(1, (e - s) // 2)) for s, e in zip(seg_starts, seg_ends)]
        pre = [
            mids[0] if len(mids) > 0 else max(1, analysis_end // 6),
            mids[1] if len(mids) > 1 else analysis_end // 3,
            mids[2] if len(mids) > 2 else 2 * analysis_end // 3,
        ]
    steps = sorted({int(np.clip(t, 0, T - 1)) for t in (*pre, cross_step, T - 1)})
    # De-duplication may leave < 5 panels; refill midpoints of the widest gaps
    # to keep the 5-frame rhythm whenever the horizon allows it.
    while len(steps) < 5:
        gaps = [(b - a, i) for i, (a, b) in enumerate(zip(steps, steps[1:]))]
        width, i = max(gaps)
        if width < 2:
            break
        steps.insert(i + 1, steps[i] + width // 2)
        steps = sorted(set(steps))
    return steps, cross_step


def plot_trajectory_storyboard(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_batch: ScenarioBatch,
    test_metrics: dict[str, dict],
    expected_cross_index: int,
    show_plots: bool = False,
    sample_idx: int = 0,
) -> None:
    """
    Four-panel storyboard figure for a single episode — paper-ready static
    replacement for the animated GIF.

    Automatically selects an episode with ≥ 3 gate switches so each panel
    shows the gate opening at a visibly different y position:
      (1) Mid of gate level 1  ->  agent approaching, gate at position A
      (2) Mid of gate level 2  ->  gate jumped to B, agent adapts
      (3) Mid of gate level 3  ->  gate jumped to C, agent adapts again
      (4) Wall crossing        ->  agent passes through final gate position

    A thin gate-schedule strip below all panels shows the full g(t) step
    function with vertical cursors marking each snapshot, providing clear
    temporal context without needing animation.
    """
    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec

    setup_plot_style(plt)
    colors = variant_colors()
    labels = {mode: label for mode, label in variant_order}

    T      = int(args.horizon)
    wall_x = float(args.wall_x)
    half_w = float(args.gate_half_width)
    corr   = float(args.corridor_limit)
    amp    = float(args.gate_amplitude)
    x_min_ax = -0.15
    x_max_ax = float(args.start_x_max) * 1.08
    y_lim    = corr * 1.08

    freeze_step = max(1, expected_cross_index - int(args.gate_settle_steps))
    continuous_gate = is_continuous_gate(args)
    analysis_end = expected_cross_index if continuous_gate else freeze_step
    analysis_end = int(np.clip(analysis_end, 1, T - 1))
    ref_mode = _storyboard_ref_mode(test_metrics, variant_order)

    # Showcase episode: prefer one the reference controller finishes cleanly
    # (and, for switching runs, with a rich schedule so the panels differ).
    chosen = _storyboard_pick_episode(
        test_batch=test_batch, test_metrics=test_metrics, ref_mode=ref_mode,
        sample_idx=sample_idx, continuous_gate=continuous_gate,
        analysis_end=analysis_end,
    )

    gate_np  = test_batch.gate_y[chosen].numpy()   # (T,)
    start_np = test_batch.start[chosen].numpy()
    cross_idx = int(test_metrics[ref_mode]["rollout"]["cross_idx"][chosen].item())

    snapshot_steps, cross_step = _storyboard_snapshot_steps(
        gate_np=gate_np, cross_idx=cross_idx, analysis_end=analysis_end,
        T=T, continuous_gate=continuous_gate,
    )
    snapshot_labels = []
    for t in snapshot_steps:
        if t == cross_step and t == T - 1:
            tag = "  (crossing, final)"
        elif t == cross_step:
            tag = "  (crossing)"
        elif t == T - 1:
            tag = "  (final)"
        else:
            tag = ""
        snapshot_labels.append(f"$t={t}${tag}\n$g={gate_np[t]:+.2f}$")
    # Cursor colors for the gate strip (one per snapshot)
    cursor_colors = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#7c3aed"]

    # ── Trajectories (prepend start -> T+1 pts) ────────────────────────────────
    trajs: dict[str, np.ndarray] = {}
    for mode, _ in variant_order:
        xy = test_metrics[mode]["rollout"]["x_seq"][chosen, :, :2].numpy()
        trajs[mode] = np.vstack([start_np[:2], xy])

    # ── Layout: 5 arena panels (tall) + 1 gate strip (short) ─────────────────
    fig = plt.figure(figsize=(17.0, 5.2))
    gs  = GridSpec(
        2, 5,
        figure=fig,
        height_ratios=[3.2, 1.0],
        hspace=0.38,
        wspace=0.10,
    )
    arena_axes = [fig.add_subplot(gs[0, col]) for col in range(5)]
    ax_gate    = fig.add_subplot(gs[1, :])   # full-width gate strip

    # ── Arena panels ──────────────────────────────────────────────────────────
    for col, (ax, t_snap, snap_label) in enumerate(
        zip(arena_axes, snapshot_steps, snapshot_labels)
    ):
        # Corridor walls
        ax.axhspan(corr, y_lim,   color="#e5e7eb", alpha=0.7, zorder=0)
        ax.axhspan(-y_lim, -corr, color="#e5e7eb", alpha=0.7, zorder=0)
        ax.axhline( corr, color="#6b7280", lw=1.2, zorder=1)
        ax.axhline(-corr, color="#6b7280", lw=1.2, zorder=1)

        # Transverse wall + gate opening at gate_y[t_snap]
        g_t = float(gate_np[min(t_snap, T - 1)])
        ax.fill_betweenx([-y_lim, g_t - half_w],
                         wall_x - 0.018, wall_x + 0.018,
                         color="#9ca3af", zorder=3)
        ax.fill_betweenx([g_t + half_w, y_lim],
                         wall_x - 0.018, wall_x + 0.018,
                         color="#9ca3af", zorder=3)
        ax.fill_betweenx([g_t - half_w, g_t + half_w],
                         wall_x - 0.018, wall_x + 0.018,
                         color="#bbf7d0", alpha=0.85, zorder=3)

        # Ghost: full trajectory (very faded)
        for mode, _ in variant_order:
            ax.plot(trajs[mode][:, 0], trajs[mode][:, 1],
                    color=colors.get(mode, "#888"), lw=1.0, alpha=0.10, zorder=2)

        # Partial trail + current position dot
        for mode, _ in variant_order:
            trail = trajs[mode][:t_snap + 2]
            ax.plot(trail[:, 0], trail[:, 1],
                    color=colors.get(mode, "#888"), lw=2.0, alpha=0.90, zorder=4)
            ax.scatter(trail[-1, 0], trail[-1, 1],
                       color=colors.get(mode, "#888"), s=45, zorder=6,
                       edgecolors="white", linewidths=0.8)

        # Start marker (first panel only)
        if col == 0:
            ax.scatter(start_np[0], start_np[1], color="#6b7280", s=50,
                       marker="o", zorder=5, edgecolors="white", linewidths=0.8)

        # Goal
        ax.scatter(0.0, 0.0, color="#111827", marker="*", s=110, zorder=7)

        # Coloured snapshot-cursor border
        for spine in ax.spines.values():
            spine.set_edgecolor(cursor_colors[col])
            spine.set_linewidth(1.8)

        ax.set_xlim(x_max_ax, x_min_ax)   # inverted: agent moves right→left
        ax.set_ylim(-y_lim, y_lim)
        ax.set_title(snap_label, fontsize=8.5, fontweight="bold", pad=4,
                     color=cursor_colors[col])
        ax.set_xlabel("$x$ (m)", fontsize=8)
        if col == 0:
            ax.set_ylabel("$y$ (m)", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)

    for ax in arena_axes[len(snapshot_steps):]:
        ax.set_visible(False)

    # ── Gate schedule strip ────────────────────────────────────────────────────
    t_ax = np.arange(T)
    if continuous_gate:
        ax_gate.plot(t_ax, gate_np, color="#6b7280", lw=1.4,
                     label="$g_t$  (gate centre)")
        ax_gate.fill_between(t_ax, gate_np - half_w, gate_np + half_w,
                             color="#bbf7d0", alpha=0.45)
    else:
        ax_gate.step(t_ax, gate_np, where="post", color="#6b7280", lw=1.4,
                     label="$g_t$  (gate centre)")
        ax_gate.fill_between(t_ax, gate_np - half_w, gate_np + half_w,
                             step="post", color="#bbf7d0", alpha=0.45)
        ax_gate.axvline(freeze_step, color="#94a3b8", lw=1.0, ls="--", zorder=2,
                        label=f"freeze $t={freeze_step}$")
    # Snapshot cursors
    for t_snap, cc in zip(snapshot_steps, cursor_colors):
        ax_gate.axvline(t_snap, color=cc, lw=1.6, ls=":", zorder=3)
        ax_gate.scatter([t_snap], [gate_np[min(t_snap, T - 1)]],
                        color=cc, s=40, zorder=4, edgecolors="white", linewidths=0.6)

    ax_gate.set_xlim(0, T - 1)
    ax_gate.set_ylim(-amp * 1.15, amp * 1.15)
    ax_gate.set_xlabel("Step $t$", fontsize=8)
    ax_gate.set_ylabel("$g_t$", fontsize=8)
    ax_gate.tick_params(labelsize=7)
    ax_gate.legend(fontsize=7, loc="upper right", ncol=2)

    # ── Shared legend ──────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=colors.get(m, "#888"), lw=2, label=labels[m])
        for m, _ in variant_order if m in test_metrics
    ]
    legend_handles += [
        Line2D([0], [0], color="#6b7280", lw=1.2, label="Corridor / wall"),
        plt.Rectangle((0, 0), 1, 1, fc="#bbf7d0", alpha=0.85, label="Gate opening"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
    )

    motion_note = (
        "continuous gate drift"
        if continuous_gate
        else f"{_count_switches(gate_np)} gate switches before freeze"
    )
    fig.suptitle(
        f"Trajectory storyboard — episode #{chosen}  ({motion_note})",
        fontsize=10, fontweight="bold", y=1.01,
    )

    out = run_dir / "trajectory_storyboard.pdf"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Saved trajectory storyboard -> {out}")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_trajectory_storyboard_compact(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_batch: ScenarioBatch,
    test_metrics: dict[str, dict],
    expected_cross_index: int,
    show_plots: bool = False,
    sample_idx: int = 0,
) -> None:
    """
    Compact paper-ready storyboard: 5 arena snapshots in a single tight row
    (≈7 inches wide, fits a two-column journal figure) plus a slim gate-strip
    below.  Same episode / snapshot logic as the full storyboard.
    """
    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec

    setup_plot_style(plt)
    colors = variant_colors()
    labels = {mode: label for mode, label in variant_order}

    T      = int(args.horizon)
    wall_x = float(args.wall_x)
    half_w = float(args.gate_half_width)
    corr   = float(args.corridor_limit)
    amp    = float(args.gate_amplitude)
    x_min_ax = -0.15
    x_max_ax = float(args.start_x_max) * 1.08
    y_lim    = corr * 1.08

    freeze_step = max(1, expected_cross_index - int(args.gate_settle_steps))
    continuous_gate = is_continuous_gate(args)
    analysis_end = expected_cross_index if continuous_gate else freeze_step
    analysis_end = int(np.clip(analysis_end, 1, T - 1))
    ref_mode = _storyboard_ref_mode(test_metrics, variant_order)

    # Same episode/snapshot policy as the full storyboard: prefer an episode
    # the reference controller finishes cleanly; panels in chronological order.
    chosen = _storyboard_pick_episode(
        test_batch=test_batch, test_metrics=test_metrics, ref_mode=ref_mode,
        sample_idx=sample_idx, continuous_gate=continuous_gate,
        analysis_end=analysis_end,
    )

    gate_np   = test_batch.gate_y[chosen].numpy()
    start_np  = test_batch.start[chosen].numpy()
    cross_idx = int(test_metrics[ref_mode]["rollout"]["cross_idx"][chosen].item())

    snapshot_steps, cross_step = _storyboard_snapshot_steps(
        gate_np=gate_np, cross_idx=cross_idx, analysis_end=analysis_end,
        T=T, continuous_gate=continuous_gate,
    )
    # Compact titles: single line, letters in chronological order; the gate
    # value is shown in the strip below instead.
    snapshot_titles = [
        f"({chr(ord('a') + i)}) $t={t}$" + (" — cross" if t == cross_step else "")
        for i, t in enumerate(snapshot_steps)
    ]
    cursor_colors = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#7c3aed"]

    trajs: dict[str, np.ndarray] = {}
    for mode, _ in variant_order:
        xy = test_metrics[mode]["rollout"]["x_seq"][chosen, :, :2].numpy()
        trajs[mode] = np.vstack([start_np[:2], xy])

    # ── Compact layout: 5 narrow panels + slim gate strip ─────────────────────
    fig = plt.figure(figsize=(7.2, 3.4))
    gs  = GridSpec(
        2, 5,
        figure=fig,
        height_ratios=[2.8, 0.8],
        hspace=0.30,
        wspace=0.06,
    )
    arena_axes = [fig.add_subplot(gs[0, col]) for col in range(5)]
    ax_gate    = fig.add_subplot(gs[1, :])

    for col, (ax, t_snap, title) in enumerate(
        zip(arena_axes, snapshot_steps, snapshot_titles)
    ):
        # Corridor walls
        ax.axhspan(corr, y_lim,   color="#e5e7eb", alpha=0.7, zorder=0)
        ax.axhspan(-y_lim, -corr, color="#e5e7eb", alpha=0.7, zorder=0)
        ax.axhline( corr, color="#6b7280", lw=0.8, zorder=1)
        ax.axhline(-corr, color="#6b7280", lw=0.8, zorder=1)

        # Wall + gate
        g_t = float(gate_np[min(t_snap, T - 1)])
        ax.fill_betweenx([-y_lim, g_t - half_w],
                         wall_x - 0.018, wall_x + 0.018,
                         color="#9ca3af", zorder=3)
        ax.fill_betweenx([g_t + half_w, y_lim],
                         wall_x - 0.018, wall_x + 0.018,
                         color="#9ca3af", zorder=3)
        ax.fill_betweenx([g_t - half_w, g_t + half_w],
                         wall_x - 0.018, wall_x + 0.018,
                         color="#bbf7d0", alpha=0.85, zorder=3)

        # Ghost + partial trail + dot
        for mode, _ in variant_order:
            ax.plot(trajs[mode][:, 0], trajs[mode][:, 1],
                    color=colors.get(mode, "#888"), lw=0.7, alpha=0.10, zorder=2)
        for mode, _ in variant_order:
            trail = trajs[mode][:t_snap + 2]
            ax.plot(trail[:, 0], trail[:, 1],
                    color=colors.get(mode, "#888"), lw=1.5, alpha=0.90, zorder=4)
            ax.scatter(trail[-1, 0], trail[-1, 1],
                       color=colors.get(mode, "#888"), s=18, zorder=6,
                       edgecolors="white", linewidths=0.5)

        # Start (first panel) + goal
        if col == 0:
            ax.scatter(start_np[0], start_np[1], color="#6b7280", s=18,
                       marker="o", zorder=5, edgecolors="white", linewidths=0.5)
        ax.scatter(0.0, 0.0, color="#111827", marker="*", s=55, zorder=7)

        # Coloured border
        for spine in ax.spines.values():
            spine.set_edgecolor(cursor_colors[col])
            spine.set_linewidth(1.2)

        ax.set_xlim(x_max_ax, x_min_ax)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_title(title, fontsize=6.5, fontweight="bold", pad=2,
                     color=cursor_colors[col])
        ax.set_xlabel("$x$", fontsize=6)
        ax.tick_params(labelsize=5.5)
        if col == 0:
            ax.set_ylabel("$y$ (m)", fontsize=6)
        else:
            ax.set_yticklabels([])

    for ax in arena_axes[len(snapshot_steps):]:
        ax.set_visible(False)

    # ── Gate strip ────────────────────────────────────────────────────────────
    t_ax = np.arange(T)
    if continuous_gate:
        ax_gate.plot(t_ax, gate_np, color="#6b7280", lw=1.0)
        ax_gate.fill_between(t_ax, gate_np - half_w, gate_np + half_w,
                             color="#bbf7d0", alpha=0.45)
    else:
        ax_gate.step(t_ax, gate_np, where="post", color="#6b7280", lw=1.0)
        ax_gate.fill_between(t_ax, gate_np - half_w, gate_np + half_w,
                             step="post", color="#bbf7d0", alpha=0.45)
        ax_gate.axvline(freeze_step, color="#94a3b8", lw=0.8, ls="--", zorder=2)
    for t_snap, cc in zip(snapshot_steps, cursor_colors):
        ax_gate.axvline(t_snap, color=cc, lw=1.2, ls=":", zorder=3)
        ax_gate.scatter([t_snap], [gate_np[min(t_snap, T - 1)]],
                        color=cc, s=18, zorder=4, edgecolors="white", linewidths=0.4)
    ax_gate.set_xlim(0, T - 1)
    ax_gate.set_ylim(-amp * 1.2, amp * 1.2)
    ax_gate.set_xlabel("Step $t$", fontsize=6)
    ax_gate.set_ylabel("$g_t$", fontsize=6)
    ax_gate.tick_params(labelsize=5.5)

    # ── Legend (single row, very compact) ─────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=colors.get(m, "#888"), lw=1.5, label=labels[m])
        for m, _ in variant_order if m in test_metrics
    ]
    legend_handles += [
        plt.Rectangle((0, 0), 1, 1, fc="#bbf7d0", alpha=0.85, label="Gate"),
        plt.Rectangle((0, 0), 1, 1, fc="#9ca3af", label="Wall"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=6,
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        handlelength=1.2,
        columnspacing=0.8,
    )

    title = "Continuous-gate navigation - storyboard" if continuous_gate else "Gate-switching navigation - storyboard"
    fig.suptitle(title, fontsize=8,
                 fontweight="bold", y=1.02)

    out = run_dir / "trajectory_storyboard_compact.pdf"
    fig.savefig(str(out), bbox_inches="tight", dpi=300)
    print(f"Saved compact storyboard -> {out}")
    if show_plots:
        plt.show()
    plt.close(fig)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def merge_json(path: Path, updates: dict) -> None:
    """Read-modify-write a JSON dict, atomically (tmp file + os.replace).

    Lets each variant persist its results the moment it finishes (preemption
    safety) and lets parallel per-variant cluster jobs share one run directory.
    """
    data: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                data = existing
        except (OSError, json.JSONDecodeError):
            pass
    data.update(updates)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    tmp.replace(path)


def load_saved_history(run_dir: Path, tag: str) -> list[dict]:
    """Best-effort recovery of a variant's training history from a previous run."""
    for name in ("train_history.json", "train_history_partial.json"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get(tag), list):
            return data[tag]
    return []


def strip_rollout(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "rollout"}


def build_interpretation(test_metrics: dict[str, dict]) -> str:
    lines = []
    if "context" in test_metrics:
        lines.append(f"Factorized M_b x M_p: success rate {test_metrics['context']['success_rate']:.3f}")
    if "mad_context" in test_metrics:
        lines.append(f"MAD s=1: {test_metrics['mad_context']['success_rate']:.3f}")
    if "mp_only_context" in test_metrics:
        lines.append(f"M_p-only (matched params): {test_metrics['mp_only_context']['success_rate']:.3f}")
    if "contextual_ssm" in test_metrics:
        lines.append(f"ContextualDeepSSM: {test_metrics['contextual_ssm']['success_rate']:.3f}")
    if "disturbance_only" in test_metrics:
        lines.append(f"Disturbance-only PB+SSM: {test_metrics['disturbance_only']['success_rate']:.3f}")
    if "nominal" in test_metrics:
        lines.append(f"Nominal pre-stabilization: {test_metrics['nominal']['success_rate']:.3f}")
    lines.append("Success requires clearing the moving wall opening and ending near the origin.")
    return " | ".join(lines)


def parse_ablation_configs(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """Parse --ablation_configs into [(label, [features])], features in canonical order.

    Empty string -> default leave-one-out sweep over the fair default context set.
    """
    raw = str(getattr(args, "ablation_configs", "") or "").strip()
    if not raw:
        base = [k for k in CONTEXT_FEATURE_ORDER if k in set(FAIR_CONTEXT_DEFAULT)]
        return [("full", base)] + [(f"-{f}", [x for x in base if x != f]) for f in base]
    out: list[tuple[str, list[str]]] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        label, feats_s = chunk.split(":", 1) if ":" in chunk else (chunk, chunk)
        feats = {f.strip() for f in feats_s.split(",") if f.strip()}
        unknown = feats - set(CONTEXT_FEATURE_ORDER)
        if unknown:
            raise ValueError(f"Ablation config {label.strip()!r}: unknown features {sorted(unknown)}.")
        if not feats:
            raise ValueError(f"Ablation config {label.strip()!r} has no features.")
        out.append((label.strip(), [k for k in CONTEXT_FEATURE_ORDER if k in feats]))
    if not out:
        raise ValueError("No valid ablation configs parsed from --ablation_configs.")
    return out


def plot_ablation_comparison(*, run_dir: Path, title: str, labels: list[str],
                             test_metrics: dict, show_plots: bool, fname: str) -> None:
    plt = get_plt(show_plots)
    setup_plot_style(plt)
    succ = [float(test_metrics[l]["success_rate"]) for l in labels]
    wall = [float(test_metrics[l]["wall_success_rate"]) for l in labels]
    cross = [float(test_metrics[l]["avg_abs_cross_error"]) for l in labels]
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8.0, 1.5 * len(labels)), 4.8))
    bw = 0.4
    ax1.bar(x - bw / 2, succ, bw, label="success", color="#2563eb")
    ax1.bar(x + bw / 2, wall, bw, label="wall-clear", color="#93c5fd")
    if "full" in labels:
        ax1.axhline(succ[labels.index("full")], color="#1e3a8a", ls=":", lw=1.2, alpha=0.7)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.0); ax1.set_ylabel("rate")
    ax1.set_title(title); ax1.legend(loc="lower left", fontsize=8)
    ax2.bar(x, cross, color="#ef4444")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("avg |gate cross-error|"); ax2.set_title("Gate cross-error (lower = better)")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(run_dir / fname, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def _run_ablation_sweep(args: argparse.Namespace, device: torch.device, *,
                        items: list, run_prefix: str, title: str,
                        extra_config: dict) -> None:
    """Train ONE architecture once per item (same data/seeds) and compare.

    ``items`` is a list of ``(label, mutate)`` where ``mutate(args)`` applies that
    config's change (e.g. set context_features, or set ssm_layers). Produces the
    comparison bar chart, the full trajectory/GIF suite, and a ranked summary.
    """
    arch = str(args.ablation_variant)
    expected_cross_index = estimate_expected_cross_index(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_id or f"{run_prefix}_{arch}_{timestamp}"
    run_dir = Path(__file__).resolve().parent / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    val_batch = sample_batch(args=args, batch_size=int(args.val_batch), seed=int(args.seed) + 50_000,
                             paired=True, shuffle=False, expected_cross_index=expected_cross_index)
    test_batch = sample_batch(args=args, batch_size=int(args.test_batch), seed=int(args.seed) + 60_000,
                              paired=True, shuffle=False, expected_cross_index=expected_cross_index,
                              use_test_starts=True)

    use_mp_only = (arch == "mp_only_context")
    use_mad = (arch == "mad_context")
    use_contextual = (arch == "contextual_ssm")

    print(f"[{run_prefix}] architecture={arch}; {len(items)} configs; run -> {run_dir}")
    controllers: dict = {}
    plants: dict = {}
    test_metrics: dict[str, dict] = {}
    val_metrics: dict[str, dict] = {}
    histories: dict[str, list[dict]] = {}
    labels: list[str] = []
    for label, mutate in items:
        mutate(args)
        labels.append(label)
        print(f"\n[{run_prefix}] '{label}'  (context_dim={context_dim(args)}, ssm_layers={args.ssm_layers})")

        # Resume: reuse a config's checkpoint from a previous (e.g. preempted)
        # run of the same run_id instead of retraining. --fresh forces retrain.
        controller = plant_true = None
        resumed = False
        pt_path = run_dir / f"{label}_controller.pt"
        if pt_path.exists() and not getattr(args, "fresh", False):
            try:
                controller, plant_true = build_controller(
                    device, args, mp_only=use_mp_only,
                    factor_rank_override=1 if use_mad else None,
                    contextual=use_contextual,
                )
                controller.load_state_dict(torch.load(pt_path, map_location=device))
                controller.eval()
                resumed = True
                print(f"[resume] Loaded {pt_path.name} — skipping training for "
                      f"'{label}' (pass --fresh to retrain).")
            except Exception as exc:
                controller = plant_true = None
                print(f"[resume] Could not reuse {pt_path.name} ({exc}); retraining from scratch.")
        if resumed:
            history = load_saved_history(run_dir, label)
            best_val = evaluate_variant(
                args=args, batch=val_batch, device=device, mode=arch,
                controller=controller, plant_true=plant_true,
                expected_cross_index=expected_cross_index)
        else:
            controller, plant_true, history, best_val = train_controller(
                args=args, device=device, mode=arch, val_batch=val_batch,
                expected_cross_index=expected_cross_index,
                mp_only=use_mp_only, factor_rank_override=1 if use_mad else None,
                contextual=use_contextual,
                run_dir=run_dir, save_tag=label,
            )
        controllers[label] = controller
        plants[label] = plant_true
        test_metrics[label] = evaluate_variant(
            args=args, batch=test_batch, device=device, mode=arch,
            controller=controller, plant_true=plant_true, expected_cross_index=expected_cross_index)
        val_metrics[label] = best_val
        histories[label] = history
        if controller is not None and not resumed:
            torch.save(controller.state_dict(), pt_path)
            (run_dir / f"{label}_controller.partial.pt").unlink(missing_ok=True)
        # Persist each config's results as soon as it finishes (preemption safety).
        merge_json(run_dir / "metrics.json", {label: strip_rollout(test_metrics[label])})
        merge_json(run_dir / "val_metrics.json", {label: strip_rollout(best_val)})
        merge_json(run_dir / "train_history.json", {label: history})

    config_payload = dict(vars(args))
    config_payload["expected_cross_index"] = int(expected_cross_index)
    config_payload.update(extra_config)
    save_json(run_dir / "config.json", config_payload)
    save_json(run_dir / "metrics.json", {lab: strip_rollout(test_metrics[lab]) for lab in labels})
    save_json(run_dir / "val_metrics.json", {lab: strip_rollout(val_metrics[lab]) for lab in labels})
    save_json(run_dir / "train_history.json", histories)
    plot_ablation_comparison(run_dir=run_dir, title=title, labels=labels,
                             test_metrics=test_metrics, show_plots=not args.no_show_plots,
                             fname=f"{run_prefix}_comparison.png")
    if getattr(args, "skip_plots", False):
        print(f"[skip_plots] Full plot suite skipped; kept {run_prefix}_comparison.png.")
    else:
        # Full comparison suite (trajectory overlays, control curves, storyboards, GIFs),
        # each config treated as a "variant" (its stored rollout carries its own setup).
        _run_all_plots(
            args=args, run_dir=run_dir, specs=[(lab, lab) for lab in labels],
            controllers=controllers, plants=plants,
            val_batch=val_batch, test_batch=test_batch,
            val_metrics=val_metrics, test_metrics=test_metrics, histories=histories,
            show_plots=not args.no_show_plots, expected_cross_index=expected_cross_index,
        )

    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)
    for label in sorted(labels, key=lambda l: test_metrics[l]["success_rate"], reverse=True):
        m = test_metrics[label]
        print(f"{label:22s} success={m['success_rate']:.3f} wall={m['wall_success_rate']:.3f} "
              f"goal={m['goal_success_rate']:.3f} cross_err={m['avg_abs_cross_error']:.3f}")
    print("=" * 76)


def run_context_ablation(args: argparse.Namespace, device: torch.device) -> None:
    """Sweep context-feature subsets for one architecture (same data/seeds)."""
    configs = parse_ablation_configs(args)

    def _mk(feats):
        return lambda a: setattr(a, "context_features", ",".join(feats))

    items = [(lab, _mk(feats)) for lab, feats in configs]
    _run_ablation_sweep(
        args, device, items=items, run_prefix="ablation",
        title=f"Context ablation — {args.ablation_variant}",
        extra_config={"ablation_configs_parsed": {lab: feats for lab, feats in configs}},
    )


def parse_ablation_layers(args: argparse.Namespace) -> list[int]:
    """Parse --ablation_layers 'n1,n2,...' into a de-duplicated list of layer counts.

    Empty -> a small sweep around the current --ssm_layers (base//2, base, base*2).
    """
    raw = str(getattr(args, "ablation_layers", "") or "").strip()
    if not raw:
        base = max(1, int(args.ssm_layers))
        return sorted({max(1, base // 2), base, base * 2})
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        n = int(tok)
        if n < 1:
            raise ValueError(f"--ablation_layers entries must be >= 1 (got {n}).")
        if n not in out:
            out.append(n)
    if not out:
        raise ValueError("No valid layer counts parsed from --ablation_layers.")
    return out


def run_layers_ablation(args: argparse.Namespace, device: torch.device) -> None:
    """Sweep SSM depth (--ssm_layers) for one architecture; same context/data/seeds."""
    layers = parse_ablation_layers(args)

    def _mk(n):
        return lambda a: setattr(a, "ssm_layers", int(n))

    items = [(f"L{n}", _mk(n)) for n in layers]
    _run_ablation_sweep(
        args, device, items=items, run_prefix="layers",
        title=f"SSM depth comparison — {args.ablation_variant}",
        extra_config={"ablation_layers": layers},
    )


def run_custom_rollout(args: argparse.Namespace, device: torch.device) -> None:
    """Roll out a trained run's controllers from a user-chosen start position.

    Loads config + checkpoints from ``--custom_rollout <run>``, pins every
    sampled episode to (--custom_start_x, --custom_start_y), simulates one
    fresh scenario (--custom_seed: gate motion + gusts), prints per-variant
    outcomes, and renders the trajectory GIF under ``<run>/custom/``.
    No training happens."""
    run_dir = Path(str(args.custom_rollout)).expanduser()
    if not run_dir.is_absolute():
        run_dir = Path(__file__).resolve().parent / "runs" / str(args.custom_rollout)
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"--custom_rollout run directory not found: {run_dir}")
    if args.custom_start_x is None or args.custom_start_y is None:
        raise ValueError("--custom_rollout requires --custom_start_x and --custom_start_y.")

    config_path = run_dir / "config.json"
    if config_path.exists():
        saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))
        cli_overrides = set(sys.argv[1:])
        for k, v in saved_cfg.items():
            if f"--{k}" not in cli_overrides and hasattr(args, k):
                setattr(args, k, v)
        print(f"[custom] Loaded run config from {config_path}")
    else:
        print(f"[custom] Warning: no config.json in {run_dir}; using current CLI args.")

    sx = float(args.custom_start_x)
    sy = float(args.custom_start_y)
    # Pin the start distribution to the requested point; scenario randomness
    # (gate motion + gusts) comes only from --custom_seed.
    args.start_x_min = args.start_x_max = sx
    args.start_y_min = sy
    args.start_y_max = sy
    args.test_start_x_min = args.test_start_x_max = None
    args.test_start_y_min = args.test_start_y_max = None
    try:
        expected_cross_index = estimate_expected_cross_index(args)
    except ValueError:
        expected_cross_index = max(1, int(args.horizon) // 2)
        print(f"[custom] No in-horizon nominal crossing from x={sx:+.2f}; "
              f"anchoring schedule/context at t={expected_cross_index}.")

    batch = sample_batch(
        args=args, batch_size=2, seed=int(args.custom_seed), paired=True,
        shuffle=False, expected_cross_index=expected_cross_index,
    )

    available = sorted(
        p.name[: -len("_controller.pt")]
        for p in run_dir.glob("*_controller.pt")
        if p.name[: -len("_controller.pt")] in ALL_VARIANTS
    )
    requested = [m.strip() for m in str(args.custom_variants or "").split(",") if m.strip()]
    modes = requested or (available + ["nominal"])
    unknown = [m for m in modes if m != "nominal" and m not in ALL_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variant(s) {unknown}. Choose from {list(ALL_VARIANTS)} or 'nominal'.")

    # Reproduce the architecture sizing used during training. In particular,
    # mp_only_context may have an automatically matched d_model, and
    # --match_to_contextual changes the d_model of every comparison variant.
    mp_only_d_model = int(args.ssm_d_model)
    mp_only_layers = int(args.mp_only_ssm_layers or args.ssm_layers)
    if "mp_only_context" in modes:
        if args.mp_only_ssm_d_model is not None:
            mp_only_d_model = int(args.mp_only_ssm_d_model)
        else:
            probe, _ = build_controller(torch.device("cpu"), args)
            target = count_params(probe)
            del probe
            base_w_dim = 8 if getattr(args, "use_w_augment", False) else 4
            mp_in_dim = base_w_dim + (
                int(args.mp_context_lift_dim) if bool(args.mp_context_lift) else 0
            )
            mp_only_d_model = find_matched_ssm_d_model(
                args=args, target_params=target, mp_in_dim=mp_in_dim,
            )

    matched_d_model: dict[str, int] = {}
    trained_modes = {mode for mode, _ in variant_specs(args)}
    if getattr(args, "match_to_contextual", False) and "contextual_ssm" in trained_modes:
        probe, _ = build_controller(torch.device("cpu"), args, contextual=True)
        target = count_params(probe)
        del probe
        shape_cache: dict[tuple[bool, bool], int] = {}
        for mode in modes:
            if mode in ("nominal", "contextual_ssm"):
                continue
            shape = (mode == "mp_only_context", mode == "mad_context")
            if shape not in shape_cache:
                shape_cache[shape] = find_matched_d_model(
                    args=args, target_params=target,
                    mp_only=shape[0], use_mad=shape[1],
                )
            matched_d_model[mode] = shape_cache[shape]

    specs: list[tuple[str, str]] = []
    test_metrics: dict[str, dict] = {}
    for mode in modes:
        label = contextual_label(args) if mode == "contextual_ssm" else ALL_VARIANTS[mode]
        if mode == "nominal":
            controller = plant_true = None
        else:
            pt_path = run_dir / f"{mode}_controller.pt"
            if not pt_path.exists():
                print(f"[custom] Skipping {mode}: {pt_path.name} not found in this run.")
                continue
            try:
                use_mp_only = mode == "mp_only_context"
                controller, plant_true = build_controller(
                    device, args,
                    mp_only=use_mp_only,
                    force_no_lift=(mode == "context_no_lift"),
                    factor_rank_override=1 if mode == "mad_context" else None,
                    ssm_d_model_override=matched_d_model.get(
                        mode, mp_only_d_model if use_mp_only else None),
                    ssm_layers_override=(
                        mp_only_layers if use_mp_only and mode not in matched_d_model else None),
                    contextual=(mode == "contextual_ssm"),
                )
                controller.load_state_dict(torch.load(pt_path, map_location=device))
                controller.eval()
            except Exception as exc:
                print(f"[custom] Skipping {mode}: could not rebuild/load ({exc}).")
                continue
        test_metrics[mode] = evaluate_variant(
            args=args, batch=batch, device=device, mode=mode,
            controller=controller, plant_true=plant_true,
            expected_cross_index=expected_cross_index,
        )
        specs.append((mode, label))
    if not specs:
        raise RuntimeError("No variant could be rolled out (no loadable checkpoints in this run).")

    out_dir = run_dir / "custom"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"x{sx:.2f}_y{sy:.2f}_s{int(args.custom_seed)}"

    print("\n" + "=" * 72)
    print(f"CUSTOM ROLLOUT  start=({sx:+.2f}, {sy:+.2f})  seed={int(args.custom_seed)}  "
          f"gate={'continuous' if is_continuous_gate(args) else 'switch'}")
    print("=" * 72)
    summary: dict[str, dict] = {}
    for mode, label in specs:
        m = test_metrics[mode]
        roll = m["rollout"]
        outcome = ("SUCCESS" if bool(roll["success"][0])
                   else ("WALL HIT" if bool(roll["collided"][0]) else "MISSED GOAL"))
        print(f"{label:40s} {outcome:12s} cross_err={m['avg_abs_cross_error']:.3f} "
              f"terminal={m['avg_terminal_dist']:.3f}")
        summary[mode] = {"outcome": outcome,
                         **{k: v for k, v in strip_rollout(m).items()
                            if isinstance(v, (int, float))}}
    print("=" * 72)
    merge_json(out_dir / "custom_results.json", {tag: summary})

    # Also save a static trajectory overview for quick inspection in the UI.
    plt = get_plt(False)
    setup_plot_style(plt)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    colors = variant_colors()
    start = batch.start[0].numpy()
    gate = batch.gate_y[0].numpy()
    ax.axvline(float(args.wall_x), color="#64748b", lw=2.0, label="wall")
    gate_at_cross = float(gate[min(expected_cross_index, len(gate) - 1)])
    ax.plot([float(args.wall_x)] * 2,
            [gate_at_cross - float(args.gate_half_width),
             gate_at_cross + float(args.gate_half_width)],
            color="#4ade80", lw=6.0, solid_capstyle="butt", label="gate opening",
            zorder=3)
    ax.scatter([0.0], [0.0], marker="*", s=150, color="#f59e0b", label="goal", zorder=5)
    ax.scatter([start[0]], [start[1]], s=55, color="#334155", label="start", zorder=5)
    for mode, label in specs:
        xy = test_metrics[mode]["rollout"]["x_seq"][0, :, :2].numpy()
        xy = np.vstack([start, xy])
        ax.plot(xy[:, 0], xy[:, 1], color=colors[mode], lw=2.0,
                label=f"{label} — {summary[mode]['outcome'].lower()}")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_title(f"Custom rollout from ({sx:+.2f}, {sy:+.2f}), seed {int(args.custom_seed)}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    trajectory_path = out_dir / f"trajectory_custom_{tag}.png"
    fig.savefig(trajectory_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[custom] Trajectory saved -> {trajectory_path}")

    _animate_one_sample(
        plt=plt, args=args, run_dir=out_dir, variant_order=specs,
        test_batch=batch, test_metrics=test_metrics, sample_idx=0,
        file_tag=f"custom_{tag}", show_plots=False,
    )
    print(f"[custom] Done -> {out_dir / f'rollout_animation_custom_{tag}.gif'}")


def main() -> None:
    args = parse_args()
    set_seeds(int(args.seed))
    # Respect the cluster CPU allocation: on a K8s node torch defaults its
    # thread pool to the NODE's core count, not the cgroup request, causing
    # CFS throttling. The launcher exports OMP_NUM_THREADS with the requested
    # cores; honor it explicitly (belt and braces with the env var itself).
    _omp_threads = os.environ.get("OMP_NUM_THREADS", "").strip()
    if _omp_threads:
        try:
            torch.set_num_threads(max(1, int(float(_omp_threads))))
        except ValueError:
            pass
    print(f"[cpu] torch intra-op threads: {torch.get_num_threads()}")

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        if getattr(args, "require_cuda", False):
            raise RuntimeError(
                "CUDA was requested with --require_cuda but no CUDA device is "
                "available in this environment. Failing fast instead of silently "
                "training on CPU (check the image/driver or drop --require_cuda).")
        print("CUDA requested but not available. Falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    if device.type == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
        if bool(getattr(args, "tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("[cuda] TF32 matmul/cudnn kernels enabled.")
        if bool(getattr(args, "cuda_amp", True)):
            if torch.cuda.is_bf16_supported():
                print("[cuda] BF16 autocast enabled for rollout and loss.")
            else:
                print("[cuda] BF16 is not supported by this GPU; using FP32.")
        if bool(getattr(args, "prefetch_batches", True)):
            print("[cuda] Pinned-memory training-batch prefetch enabled.")

    # Custom inference must run before the training/ablation dispatch below.
    if args.custom_rollout:
        run_custom_rollout(args, device)
        return

    # ── Context-ablation mode ─────────────────────────────────────────────────
    if getattr(args, "ablate_context", False):
        run_context_ablation(args, device)
        return

    # ── SSM-depth comparison mode ─────────────────────────────────────────────
    if getattr(args, "ablate_layers", False):
        run_layers_ablation(args, device)
        return

    # ── Plot-only mode ────────────────────────────────────────────────────────
    if args.plot_only:
        _plot_only_path = Path(args.plot_only).expanduser()
        if not _plot_only_path.is_absolute():
            _plot_only_path = Path(__file__).resolve().parent / "runs" / args.plot_only
        run_dir = _plot_only_path.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--plot_only path does not exist: {run_dir}")
        config_path = run_dir / "config.json"
        if config_path.exists():
            import json as _json
            saved_cfg = _json.loads(config_path.read_text(encoding="utf-8"))
            # Merge saved config into args (CLI overrides saved values where specified)
            cli_overrides = set(sys.argv[1:])
            for k, v in saved_cfg.items():
                if f"--{k}" not in cli_overrides and hasattr(args, k):
                    setattr(args, k, v)
            if "--no_mad_comparison" in cli_overrides:
                args.mad_comparison = False
            if "mad_comparison" not in saved_cfg and "--mad_comparison" not in cli_overrides:
                args.mad_comparison = False
            print(f"[plot_only] Loaded config from {config_path}")
        else:
            print(f"[plot_only] Warning: no config.json found in {run_dir}, using current args.")

        expected_cross_index = estimate_expected_cross_index(args)
        specs = variant_specs(args)

        test_batch = sample_batch(
            args=args,
            batch_size=int(args.test_batch),
            seed=int(args.seed) + 60_000,
            paired=True,
            shuffle=False,
            expected_cross_index=expected_cross_index,
            use_test_starts=True,
        )
        val_batch = sample_batch(
            args=args,
            batch_size=int(args.val_batch),
            seed=int(args.seed) + 50_000,
            paired=True,
            shuffle=False,
            expected_cross_index=expected_cross_index,
        )

        controllers: dict[str, PBController | None] = {}
        plants: dict[str, DoubleIntegratorTrue | None] = {}
        val_metrics: dict[str, dict] = {}
        test_metrics: dict[str, dict] = {}
        histories: dict[str, list[dict]] = {}

        mp_only_d_model = int(args.ssm_d_model)
        mp_only_layers = int(args.ssm_layers)

        for mode, label in specs:
            pt_path = run_dir / f"{mode}_controller.pt"
            use_mp_only = (mode == "mp_only_context")
            use_mad = (mode == "mad_context")
            use_contextual = (mode == "contextual_ssm")
            controller, plant_true = build_controller(
                device,
                args,
                mp_only=use_mp_only,
                factor_rank_override=1 if use_mad else None,
                contextual=use_contextual,
            )
            if pt_path.exists():
                controller.load_state_dict(torch.load(pt_path, map_location=device))
                print(f"[plot_only] Loaded {pt_path.name}")
            else:
                print(f"[plot_only] Warning: {pt_path.name} not found — using random weights for {mode}.")
            controllers[mode] = controller
            plants[mode] = plant_true
            histories[mode] = []
            val_metrics[mode] = evaluate_variant(
                args=args, batch=val_batch, device=device, mode=mode,
                controller=controller, plant_true=plant_true,
                expected_cross_index=expected_cross_index,
            )
            test_metrics[mode] = evaluate_variant(
                args=args, batch=test_batch, device=device, mode=mode,
                controller=controller, plant_true=plant_true,
                expected_cross_index=expected_cross_index,
            )

        show_plots = not args.no_show_plots
        _run_all_plots(
            args=args, run_dir=run_dir, specs=specs,
            controllers=controllers, plants=plants,
            val_batch=val_batch, test_batch=test_batch,
            val_metrics=val_metrics, test_metrics=test_metrics,
            histories=histories, show_plots=show_plots,
            expected_cross_index=expected_cross_index,
        )
        return

    # ─────────────────────────────────────────────────────────────────────────

    expected_cross_index = estimate_expected_cross_index(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _gate_tag = "continuous" if is_continuous_gate(args) else "controlled"
    run_name = args.run_id or f"{_gate_tag}_xy_{timestamp}"
    run_dir = Path(__file__).resolve().parent / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload = dict(vars(args))
    config_payload["context_dim"] = int(context_dim(args))
    config_payload["expected_cross_index"] = int(expected_cross_index)
    save_json(run_dir / "config.json", config_payload)

    val_batch = sample_batch(
        args=args,
        batch_size=int(args.val_batch),
        seed=int(args.seed) + 50_000,
        paired=True,
        shuffle=False,
        expected_cross_index=expected_cross_index,
    )
    test_batch = sample_batch(
        args=args,
        batch_size=int(args.test_batch),
        seed=int(args.seed) + 60_000,
        paired=True,
        shuffle=False,
        expected_cross_index=expected_cross_index,
        use_test_starts=True,
    )
    _tr = resolve_start_ranges(args)
    _te = resolve_start_ranges(args, test=True)
    if _te != _tr:
        print(f"[generalization] TEST starts: x in [{_te[0]:.2f}, {_te[1]:.2f}], "
              f"y in [{_te[2]:.2f}, {_te[3]:.2f}] (training: x in [{_tr[0]:.2f}, {_tr[1]:.2f}], "
              f"y in [{_tr[2]:.2f}, {_tr[3]:.2f}]).")

    specs = variant_specs(args)
    controllers: dict[str, PBController | None] = {}
    plants: dict[str, DoubleIntegratorTrue | None] = {}
    histories: dict[str, list[dict]] = {}
    val_metrics: dict[str, dict] = {}
    test_metrics: dict[str, dict] = {}

    print(f"Run directory: {run_dir}")
    print(f"Expected nominal wall crossing occurs near step {expected_cross_index}.")

    # ── Parameter matching (v3) ───────────────────────────────────────────────
    # Build a temporary factorized controller (context variant) to count its params.
    # Then find the M_p-only SSM size that covers the same budget.
    # Skipped in --simple_comparison mode (no mp_only_context variant).
    mp_only_d_model: int = int(args.ssm_d_model)
    mp_only_layers: int = int(args.ssm_layers)
    _probe_device = torch.device("cpu")  # param counting is device-independent
    if any(m == "mp_only_context" for m, _ in specs):
        _tmp_ctrl, _ = build_controller(_probe_device, args, mp_only=False)
        factorized_total = count_params(_tmp_ctrl)
        del _tmp_ctrl

        nx = 4
        base_w_dim = nx * 2 if getattr(args, "use_w_augment", False) else nx
        mp_in_dim_with_lift = base_w_dim + (int(args.mp_context_lift_dim) if bool(args.mp_context_lift) else 0)
        if args.mp_only_ssm_d_model is not None:
            mp_only_d_model = int(args.mp_only_ssm_d_model)
            print(f"[v3] M_p-only SSM d_model overridden to {mp_only_d_model} (manual).")
        else:
            mp_only_d_model = find_matched_ssm_d_model(
                args=args,
                target_params=factorized_total,
                mp_in_dim=mp_in_dim_with_lift,
            )
            print(f"[v3] Auto-matched M_p-only SSM d_model = {mp_only_d_model} "
                  f"(target >= {factorized_total:,} params).")
        mp_only_layers = int(args.mp_only_ssm_layers or args.ssm_layers)

    # Optional: size every OTHER trainable variant so its total param count matches
    # contextual_ssm as closely as possible (fair comparison; layers stay fixed).
    matched_d_model: dict[str, int] = {}
    if getattr(args, "match_to_contextual", False) and any(m == "contextual_ssm" for m, _ in specs):
        _ctx_ctrl, _ = build_controller(_probe_device, args, contextual=True)
        _target = count_params(_ctx_ctrl)
        del _ctx_ctrl
        print(f"[match] target = contextual_ssm params = {_target:,}")
        _shape_cache: dict = {}
        for _mode, _ in specs:
            if _mode in ("nominal", "contextual_ssm"):
                continue
            shape = (_mode == "mp_only_context", _mode == "mad_context")  # (mp_only, use_mad)
            if shape not in _shape_cache:
                _shape_cache[shape] = find_matched_d_model(
                    args=args, target_params=_target,
                    mp_only=shape[0], use_mad=shape[1])
            matched_d_model[_mode] = _shape_cache[shape]
            _c, _ = build_controller(
                _probe_device, args, mp_only=shape[0],
                factor_rank_override=1 if shape[1] else None,
                ssm_d_model_override=matched_d_model[_mode])
            print(f"[match] {_mode}: ssm_d_model={matched_d_model[_mode]} -> "
                  f"{count_params(_c):,} params (target {_target:,})")
            del _c

    for mode, label in specs:
        print(f"\nTraining/evaluating {label}...")
        # Warm-start the context mode from the disturbance_only checkpoint.
        # mp_only_context starts from scratch (different architecture / size).
        warm_start = None
        if mode == "context" and args.warm_start and controllers.get("disturbance_only") is not None:
            warm_start = {k: v.detach().cpu().clone() for k, v in controllers["disturbance_only"].state_dict().items()}
            print("[context] warm-starting from disturbance_only checkpoint.")
        use_mp_only = (mode == "mp_only_context")
        use_no_lift = (mode == "context_no_lift")
        use_mad = (mode == "mad_context")
        use_contextual = (mode == "contextual_ssm")
        mode_overrides = dict(
            mp_only=use_mp_only,
            force_no_lift=use_no_lift,
            factor_rank_override=1 if use_mad else None,
            ssm_d_model_override=matched_d_model.get(mode, mp_only_d_model if use_mp_only else None),
            ssm_layers_override=(mp_only_layers if (use_mp_only and mode not in matched_d_model) else None),
            contextual=use_contextual,
        )

        # Resume: a completed variant (checkpoint already in the run dir, e.g.
        # after a preempted-and-restarted cluster job) is loaded and re-evaluated
        # instead of retrained. --fresh forces retraining.
        controller = plant_true = None
        resumed = False
        pt_path = run_dir / f"{mode}_controller.pt"
        if mode != "nominal" and pt_path.exists() and not getattr(args, "fresh", False):
            try:
                controller, plant_true = build_controller(device, args, **mode_overrides)
                controller.load_state_dict(torch.load(pt_path, map_location=device))
                controller.eval()
                resumed = True
                print(f"[resume] Loaded {pt_path.name} — skipping training for "
                      f"{mode} (pass --fresh to retrain).")
            except Exception as exc:
                controller = plant_true = None
                print(f"[resume] Could not reuse {pt_path.name} ({exc}); retraining from scratch.")
        if resumed:
            history = load_saved_history(run_dir, mode)
            best_val_metrics = evaluate_variant(
                args=args, batch=val_batch, device=device, mode=mode,
                controller=controller, plant_true=plant_true,
                expected_cross_index=expected_cross_index,
            )
        else:
            controller, plant_true, history, best_val_metrics = train_controller(
                args=args,
                device=device,
                mode=mode,
                val_batch=val_batch,
                expected_cross_index=expected_cross_index,
                warm_start_state=warm_start,
                run_dir=run_dir,
                **mode_overrides,
            )
        controllers[mode] = controller
        plants[mode] = plant_true
        histories[mode] = history
        val_metrics[mode] = best_val_metrics
        test_metrics[mode] = evaluate_variant(
            args=args,
            batch=test_batch,
            device=device,
            mode=mode,
            controller=controller,
            plant_true=plant_true,
            expected_cross_index=expected_cross_index,
        )
        if controller is not None and not resumed:
            torch.save(controller.state_dict(), pt_path)
            (run_dir / f"{mode}_controller.partial.pt").unlink(missing_ok=True)
        # Persist this variant's results immediately (preemption safety; also
        # lets parallel per-variant jobs merge into one shared run directory).
        merge_json(run_dir / "metrics.json", {mode: strip_rollout(test_metrics[mode])})
        merge_json(run_dir / "val_metrics.json", {mode: strip_rollout(best_val_metrics)})
        merge_json(run_dir / "train_history.json", {mode: history})

    show_plots = not args.no_show_plots
    if getattr(args, "skip_plots", False):
        print("\n[skip_plots] Figures/animations skipped; metrics and checkpoints "
              "were saved incrementally.")
        print(f"Re-generate all plots later with: --plot_only {run_dir.name}")
        print("\n" + "=" * 76)
        print("RESULTS OVERVIEW")
        print("=" * 76)
        for mode, label in specs:
            metrics = test_metrics[mode]
            print(
                f"{label:30s} "
                f"success={metrics['success_rate']:.3f} "
                f"wall={metrics['wall_success_rate']:.3f} "
                f"goal={metrics['goal_success_rate']:.3f} "
                f"term={metrics['avg_terminal_dist']:.3f} "
                f"cross_err={metrics['avg_abs_cross_error']:.3f}"
            )
        print("=" * 76)
        return
    _run_all_plots(
        args=args, run_dir=run_dir, specs=specs,
        controllers=controllers, plants=plants,
        val_batch=val_batch, test_batch=test_batch,
        val_metrics=val_metrics, test_metrics=test_metrics,
        histories=histories, show_plots=show_plots,
        expected_cross_index=expected_cross_index,
    )


def plot_start_generalization(
    *,
    args,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_batch: ScenarioBatch,
    test_metrics: dict,
    show_plots: bool,
) -> None:
    """Per-variant map of TEST-episode outcomes at their start positions.

    Green = success, red x = wall hit, hollow orange = reached the wall but
    missed the goal. The dotted box marks the TRAINING start region, so
    out-of-distribution starts (via --test_start_*) are immediately visible.
    """
    if any("success" not in test_metrics[m]["rollout"] for m, _ in variant_order):
        print("[plots] start_generalization skipped (per-episode outcomes missing; "
              "re-run evaluation with this code version).")
        return
    from matplotlib.patches import Rectangle

    plt = get_plt(show_plots)
    setup_plot_style(plt)
    n = len(variant_order)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n + 0.8, 4.2),
                             sharex=True, sharey=True, squeeze=False)
    starts = test_batch.start.numpy()
    tr_x_min, tr_x_max, tr_y_min, tr_y_max = resolve_start_ranges(args)
    for ax, (mode, label) in zip(axes[0], variant_order):
        roll = test_metrics[mode]["rollout"]
        succ = roll["success"].numpy().astype(bool)
        coll = roll["collided"].numpy().astype(bool)
        miss = (~succ) & (~coll)
        ax.scatter(starts[succ, 0], starts[succ, 1], s=14, color="#16a34a",
                   alpha=0.75, label="success", zorder=3)
        ax.scatter(starts[coll, 0], starts[coll, 1], s=20, color="#dc2626",
                   marker="x", alpha=0.85, label="wall hit", zorder=3)
        ax.scatter(starts[miss, 0], starts[miss, 1], s=18, facecolors="none",
                   edgecolors="#f59e0b", alpha=0.85, label="missed goal", zorder=3)
        ax.axvline(float(args.wall_x), color="#475569", lw=1.2, ls="--")
        ax.scatter([0.0], [0.0], marker="*", s=90, color="#2563eb", zorder=4)
        ax.add_patch(Rectangle((tr_x_min, tr_y_min), tr_x_max - tr_x_min,
                               tr_y_max - tr_y_min, fill=False, ls=":", lw=1.3,
                               ec="#64748b", zorder=2))
        ax.set_title(f"{label}\nsuccess {100.0 * float(succ.mean()):.1f}%", fontsize=10)
        ax.set_xlabel("start x")
    axes[0][0].set_ylabel("start y")
    axes[0][0].legend(loc="best", fontsize=8, framealpha=0.9)
    fig.suptitle("Generalization over initial positions (dotted box = training starts)",
                 y=1.04)
    fig.savefig(run_dir / "start_generalization.png", bbox_inches="tight")
    print(f"Saved start-generalization map -> {run_dir / 'start_generalization.png'}")
    if show_plots:
        plt.show()
    plt.close(fig)


def _run_all_plots(
    *,
    args,
    run_dir: Path,
    specs: list[tuple[str, str]],
    controllers: dict,
    plants: dict,
    val_batch: ScenarioBatch,
    test_batch: ScenarioBatch,
    val_metrics: dict,
    test_metrics: dict,
    histories: dict,
    show_plots: bool,
    expected_cross_index: int,
) -> None:
    """Run all plots and save JSON metrics. Shared by normal and --plot_only paths."""
    plot_loss_curves(
        run_dir=run_dir,
        histories=histories,
        variant_order=specs,
        show_plots=show_plots,
    )
    plot_wall_style_summary(
        args=args,
        run_dir=run_dir,
        variant_order=specs,
        test_batch=test_batch,
        test_metrics=test_metrics,
        show_plots=show_plots,
    )
    plot_control_magnitude(
        args=args,
        run_dir=run_dir,
        variant_order=specs,
        test_metrics=test_metrics,
        show_plots=show_plots,
    )
    plot_trajectory_samples(
        args=args,
        run_dir=run_dir,
        variant_order=specs,
        test_batch=test_batch,
        test_metrics=test_metrics,
        show_plots=show_plots,
    )
    plot_start_generalization(
        args=args,
        run_dir=run_dir,
        variant_order=specs,
        test_batch=test_batch,
        test_metrics=test_metrics,
        show_plots=show_plots,
    )
    animate_rollout(
        args=args,
        run_dir=run_dir,
        variant_order=specs,
        test_batch=test_batch,
        test_metrics=test_metrics,
        show_plots=show_plots,
    )
    if not is_continuous_gate(args):
        animate_adversarial_sample(
            args=args,
            run_dir=run_dir,
            variant_order=specs,
            test_batch=test_batch,
            test_metrics=test_metrics,
            show_plots=show_plots,
        )
        plot_waiting_behavior(
            args=args,
            run_dir=run_dir,
            test_batch=test_batch,
            test_metrics=test_metrics,
            show_plots=show_plots,
        )
        plot_adversarial_switching(
            args=args,
            run_dir=run_dir,
            variant_order=specs,
            test_batch=test_batch,
            test_metrics=test_metrics,
            show_plots=show_plots,
        )
    plot_sample_trajectory(
        args=args,
        run_dir=run_dir,
        variant_order=specs,
        test_batch=test_batch,
        test_metrics=test_metrics,
        expected_cross_index=expected_cross_index,
        show_plots=show_plots,
    )
    if getattr(args, "use_storyboard", True):
        plot_trajectory_storyboard(
            args=args,
            run_dir=run_dir,
            variant_order=specs,
            test_batch=test_batch,
            test_metrics=test_metrics,
            expected_cross_index=expected_cross_index,
            show_plots=show_plots,
        )
    if getattr(args, "use_storyboard_compact", True):
        plot_trajectory_storyboard_compact(
            args=args,
            run_dir=run_dir,
            variant_order=specs,
            test_batch=test_batch,
            test_metrics=test_metrics,
            expected_cross_index=expected_cross_index,
            show_plots=show_plots,
        )

    save_json(run_dir / "metrics.json", {mode: strip_rollout(test_metrics[mode]) for mode, _ in specs})
    save_json(run_dir / "val_metrics.json", {mode: strip_rollout(val_metrics[mode]) for mode, _ in specs})
    save_json(run_dir / "train_history.json", histories)

    interpretation = build_interpretation(test_metrics)
    (run_dir / "interpretation.txt").write_text(interpretation + "\n", encoding="utf-8")

    print("\n" + "=" * 76)
    print("RESULTS OVERVIEW")
    print("=" * 76)
    for mode, label in specs:
        metrics = test_metrics[mode]
        print(
            f"{label:30s} "
            f"success={metrics['success_rate']:.3f} "
            f"wall={metrics['wall_success_rate']:.3f} "
            f"goal={metrics['goal_success_rate']:.3f} "
            f"term={metrics['avg_terminal_dist']:.3f} "
            f"cross_err={metrics['avg_abs_cross_error']:.3f}"
        )
    print("=" * 76)
    print(interpretation)


if __name__ == "__main__":
    main()
