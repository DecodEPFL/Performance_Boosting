"""Deterministic invariants for the contextual payload-regime task."""
from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from Moving_payload_exp import parse_args
from nav_plants import PayloadSwitchingDoubleIntegratorTrue
from payload_core import build_context, rollout_variant, sample_batch


class PayloadExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = parse_args(["--horizon", "96", "--train_batch", "4", "--val_batch", "4", "--test_batch", "4"])

    def test_paired_payload_switches_are_mirrored(self) -> None:
        batch = sample_batch(self.args, batch_size=4, seed=12, paired=True, shuffle=False)
        for first in (0, 2):
            second = first + 1
            self.assertTrue(torch.equal(batch.mass_true[first], batch.mass_true[second]))
            self.assertTrue(torch.equal(batch.actuator_gain_true[first], batch.actuator_gain_true[second]))
            self.assertTrue(torch.equal(batch.lateral_bias_true[first], -batch.lateral_bias_true[second]))
            step = int(batch.switch_step[first])
            self.assertGreaterEqual(step, self.args.payload_switch_min)
            self.assertLessEqual(step, self.args.payload_switch_max)
            self.assertTrue(torch.all(batch.mass_true[first, :step] == batch.mass_before[first]))
            self.assertTrue(torch.all(batch.mass_true[first, step:] == batch.mass_after[first]))

    def test_heavier_payload_has_less_control_acceleration(self) -> None:
        plant = PayloadSwitchingDoubleIntegratorTrue(dt=.1, pre_kp=0.0, pre_kd=0.0)
        mass = torch.tensor([[1.0], [2.0]])
        plant.set_payload_schedule(mass=mass, actuator_gain=torch.ones_like(mass), drag=torch.zeros_like(mass), lateral_bias=torch.zeros_like(mass))
        x = torch.zeros(2, 1, 4); u = torch.ones(2, 1, 2); next_x = plant.forward(x, u, t=0)
        self.assertGreater(float(next_x[0, 0, 2]), float(next_x[1, 0, 2]))

    def test_delayed_context_uses_observed_past_only(self) -> None:
        self.args.context_features = "payload_mass"
        self.args.payload_context_delay = 3
        batch = sample_batch(self.args, batch_size=4, seed=18, paired=True, shuffle=False)
        x = torch.zeros(4, 1, 4); time = int(batch.switch_step[0])
        z = build_context(self.args, batch, x, time, mode="context", training=False)
        expected = batch.mass_obs[:, time - 3:time - 2] / self.args.payload_mass_ref * self.args.z_scale
        self.assertTrue(torch.allclose(z[:, 0], expected, atol=1e-6))

    def test_nominal_rollout_preserves_batch_and_time_shapes(self) -> None:
        batch = sample_batch(self.args, batch_size=4, seed=26, paired=True, shuffle=False)
        result = rollout_variant(self.args, batch, torch.device("cpu"), mode="nominal", controller=None, plant=None)
        self.assertEqual(tuple(result.x_seq.shape), (4, self.args.horizon, 4))
        self.assertEqual(tuple(result.u_seq.shape), (4, self.args.horizon, 2))

    def test_bias_settles_and_noise_decays_by_default(self) -> None:
        batch = sample_batch(self.args, batch_size=4, seed=33, paired=True, shuffle=False)
        settle = int(self.args.payload_bias_settle_steps)
        for i in range(4):
            step = int(batch.switch_step[i])
            if step >= 0 and step + settle < self.args.horizon:
                self.assertTrue(torch.all(batch.lateral_bias_true[i, step + settle:] == 0.0))
        # taper noise window: last step ~0, early steps untouched
        self.assertLess(float(batch.process_noise[:, -1].abs().max()), 1e-6)
        legacy = parse_args(["--horizon", "96", "--noise_decay", "none", "--payload_bias_settle_steps", "0"])
        legacy_batch = sample_batch(legacy, batch_size=4, seed=33, paired=True, shuffle=False)
        self.assertTrue(torch.allclose(batch.process_noise[:, :20], legacy_batch.process_noise[:, :20]))

    def test_contextual_controller_builds_and_rolls_out(self) -> None:
        from payload_core import build_controller
        args = parse_args(["--horizon", "96", "--ssm_layers", "2", "--ssm_d_model", "16", "--ssm_d_state", "16", "--ctx_d_features", "8"])
        controller, plant = build_controller(torch.device("cpu"), args, mad=False, contextual=True)
        batch = sample_batch(args, batch_size=4, seed=44, paired=True, shuffle=False)
        with torch.no_grad():
            result = rollout_variant(args, batch, torch.device("cpu"), mode="contextual_ssm", controller=controller, plant=plant)
        self.assertEqual(tuple(result.u_seq.shape), (4, args.horizon, 2))
        with self.assertRaises(ValueError):  # select port needs a selective core
            bad = parse_args(["--ssm_param", "lru", "--ctx_select"])
            build_controller(torch.device("cpu"), bad, mad=False, contextual=True)


if __name__ == "__main__":
    unittest.main()
