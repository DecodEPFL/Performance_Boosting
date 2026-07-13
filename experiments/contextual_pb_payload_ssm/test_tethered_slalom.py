"""Structural and numerical invariants for Tethered Cargo Slalom."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Moving_payload_exp import parse_args
from payload_core import CONTEXT_FEATURE_ORDER, PAYLOAD_CONTEXT_FEATURES
from tethered_experiment import hard_gate_metrics
from tethered_payload import (NonlinearCarrierNominal, SlalomRollout,
                              TetheredPayloadPlant,
                              build_slalom_context, build_slalom_controller,
                              nonlinear_carrier_acceleration,
                              nonlinear_carrier_energy, rollout_slalom,
                              sample_slalom_batch)


class TetheredSlalomTest(unittest.TestCase):
    def args(self, horizon: int = 36):
        return parse_args([
            "--horizon", str(horizon), "--train_batch", "4", "--val_batch", "4",
            "--test_batch", "4", "--ssm_layers", "1", "--ssm_d_model", "8",
            "--ssm_d_state", "8", "--feat_dim", "8", "--mb_hidden", "12",
            "--mb_layers", "2", "--payload_context_dropout_p", "0",
        ])

    def test_alias_pairs_share_observation_but_reverse_hidden_swing(self) -> None:
        args = self.args(); batch = sample_slalom_batch(
            args, batch_size=4, seed=8, paired=True, shuffle=False)
        for first in (0, 2):
            second = first + 1
            self.assertTrue(torch.equal(batch.start[first], batch.start[second]))
            self.assertTrue(torch.equal(batch.gate_centers[first], batch.gate_centers[second]))
            self.assertEqual(float(batch.payload_mass[first]), float(batch.payload_mass[second]))
            self.assertEqual(float(batch.tether_length[first]), float(batch.tether_length[second]))
            self.assertTrue(torch.equal(batch.process_noise[first], batch.process_noise[second]))
            self.assertTrue(torch.equal(batch.payload_velocity_start[first],
                                        -batch.payload_velocity_start[second]))

    def test_route_only_is_aliased_but_payload_context_separates_pair(self) -> None:
        args = self.args(); batch = sample_slalom_batch(
            args, batch_size=4, seed=11, paired=True, shuffle=False)
        plant = TetheredPayloadPlant(args); plant.bind(batch, torch.device("cpu"))
        state = torch.cat([batch.start, torch.zeros(4, 2)], -1).unsqueeze(1)
        route = build_slalom_context(args, batch, plant, state, 0,
                                      mode="route_context", training=False)
        full = build_slalom_context(args, batch, plant, state, 0,
                                     mode="context", training=False)
        self.assertTrue(torch.equal(route[0], route[1]))
        self.assertFalse(torch.equal(full[0], full[1]))

    def test_missing_telemetry_zeros_every_payload_slot(self) -> None:
        args = self.args(); batch = sample_slalom_batch(
            args, batch_size=4, seed=13, paired=True, shuffle=False)
        plant = TetheredPayloadPlant(args); plant.bind(batch, torch.device("cpu"))
        state = torch.cat([batch.start, torch.zeros(4, 2)], -1).unsqueeze(1)
        missing = build_slalom_context(args, batch, plant, state, 0,
                                        mode="context", training=False, intervention="dropout")
        indices = [CONTEXT_FEATURE_ORDER.index(name) for name in PAYLOAD_CONTEXT_FEATURES]
        self.assertTrue(torch.equal(missing[:, :, indices], torch.zeros_like(missing[:, :, indices])))

    def test_route_and_full_controllers_have_identical_capacity(self) -> None:
        args = self.args()
        route, _ = build_slalom_controller(torch.device("cpu"), args, "route_context")
        full, _ = build_slalom_controller(torch.device("cpu"), args, "context")
        self.assertEqual(sum(p.numel() for p in route.parameters()),
                         sum(p.numel() for p in full.parameters()))
        self.assertEqual({key: tuple(value.shape) for key, value in route.state_dict().items()},
                         {key: tuple(value.shape) for key, value in full.state_dict().items()})

    def test_private_plant_rollout_is_deterministic_and_shape_safe(self) -> None:
        args = self.args(20); batch = sample_slalom_batch(
            args, batch_size=4, seed=19, paired=True, shuffle=False)
        first = rollout_slalom(args, batch, torch.device("cpu"), mode="nominal",
                               controller=None, plant=None)
        second = rollout_slalom(args, batch, torch.device("cpu"), mode="nominal",
                                controller=None, plant=None)
        self.assertEqual(tuple(first.x_seq.shape), (4, 20, 4))
        self.assertEqual(tuple(first.payload_pos_seq.shape), (4, 20, 2))
        self.assertTrue(torch.equal(first.x_seq, second.x_seq))
        self.assertTrue(torch.equal(first.payload_pos_seq, second.payload_pos_seq))

    def test_nominal_and_true_carrier_transitions_match_exactly(self) -> None:
        args = self.args(20)
        self.assertEqual(args.slalom_tether_reaction, 0.0)
        batch = sample_slalom_batch(
            args, batch_size=4, seed=29, paired=True, shuffle=False)
        true = TetheredPayloadPlant(args); true.bind(batch, torch.device("cpu"))
        nominal = NonlinearCarrierNominal(args)
        x = torch.tensor([
            [[2.1, .25, -.4, .3]], [[1.4, -.35, -.8, -.2]],
            [[.7, .18, .5, -.6]], [[-.3, -.2, .35, .4]],
        ])
        u = torch.tensor([
            [[1.7, -.9]], [[-2.0, .4]], [[.6, 1.9]], [[-1.1, -1.4]],
        ])
        self.assertTrue(torch.equal(
            true.forward(x, u), nominal.nominal_dynamics(x, u)))

    def test_origin_is_equilibrium_and_unforced_energy_decays(self) -> None:
        args = self.args(20); nominal = NonlinearCarrierNominal(args)
        zero_x, zero_u = torch.zeros(4, 1, 4), torch.zeros(4, 1, 2)
        self.assertTrue(torch.equal(
            nominal.nominal_dynamics(zero_x, zero_u), zero_x))

        batch = sample_slalom_batch(
            args, batch_size=4, seed=31, paired=True, shuffle=False)
        batch.start.zero_(); batch.payload_velocity_start.zero_()
        batch.payload_start.zero_()
        batch.payload_start[:, 0] = batch.tether_length
        true = TetheredPayloadPlant(args); true.bind(batch, torch.device("cpu"))
        self.assertTrue(torch.equal(true.forward(zero_x, zero_u), zero_x))
        payload_position, payload_velocity, tension = true.observe()
        self.assertTrue(torch.equal(payload_position[:, 0, 0], batch.tether_length))
        self.assertTrue(torch.equal(payload_velocity, torch.zeros_like(payload_velocity)))
        self.assertTrue(torch.equal(tension, torch.zeros_like(tension)))

        x = torch.tensor([
            [[2.2, .5, -.8, .4]], [[-1.6, .8, .6, -.7]],
            [[.9, -1.1, -.4, .9]], [[-.7, -.6, .8, .5]],
        ])
        initial_energy = nonlinear_carrier_energy(args, x)
        for _ in range(300):
            x = nominal.nominal_dynamics(x, zero_u)
        final_energy = nonlinear_carrier_energy(args, x)
        self.assertTrue(torch.all(final_energy < .02 * initial_energy))

    def test_carrier_dynamics_are_genuinely_nonlinear(self) -> None:
        args = self.args()
        q1 = torch.tensor([[[1.5, .4]]]); v1 = torch.tensor([[[-.7, .3]]])
        q2 = torch.tensor([[[.6, -.8]]]); v2 = torch.tensor([[[.2, -.5]]])
        zero_u = torch.zeros(1, 1, 2)
        a_sum = nonlinear_carrier_acceleration(args, q1 + q2, v1 + v2, zero_u)
        sum_a = (
            nonlinear_carrier_acceleration(args, q1, v1, zero_u)
            +nonlinear_carrier_acceleration(args, q2, v2, zero_u)
        )
        self.assertGreater(float(torch.linalg.vector_norm(a_sum - sum_a)), .05)

    def test_reconstructed_disturbance_is_exactly_tapered_process_noise(self) -> None:
        args = self.args(32); batch = sample_slalom_batch(
            args, batch_size=4, seed=37, paired=True, shuffle=False)
        controller, plant = build_slalom_controller(
            torch.device("cpu"), args, "disturbance_only")
        with torch.no_grad():
            rollout = rollout_slalom(
                args, batch, torch.device("cpu"), mode="disturbance_only",
                controller=controller, plant=plant)
        self.assertTrue(torch.equal(rollout.w_seq[:, 0], torch.zeros(4, 4)))
        self.assertTrue(torch.allclose(
            rollout.w_seq[:, 1:], batch.process_noise[:, :-1],
            atol=2e-7, rtol=0.0))
        self.assertTrue(torch.equal(
            batch.process_noise[:, -1], torch.zeros_like(batch.process_noise[:, -1])))

    def test_hard_gate_metric_counts_pass_collision_and_no_crossing(self) -> None:
        args = self.args(12); batch = sample_slalom_batch(
            args, batch_size=4, seed=23, paired=True, shuffle=False)
        batch.gate_centers = torch.zeros_like(batch.gate_centers)
        x_line = torch.linspace(2.4, 0.0, 7)
        carrier = torch.zeros(4, 7, 2); carrier[..., 0] = x_line
        payload = carrier.clone(); payload[..., 0] += .4
        batch.start = carrier[:, 0].clone(); batch.payload_start = payload[:, 0].clone()
        payload[1, :, 1] = .65; batch.payload_start[1, 1] = .65
        carrier[2:, :, 0] = 2.4; payload[2:, :, 0] = 2.8
        batch.start[2:, 0] = 2.4; batch.payload_start[2:, 0] = 2.8
        state = torch.cat([carrier, torch.zeros(4, 7, 2)], -1)
        rollout = SlalomRollout(
            x_seq=state, u_seq=torch.zeros(4, 7, 2), w_seq=torch.zeros(4, 7, 4),
            payload_pos_seq=payload, payload_vel_seq=torch.zeros(4, 7, 2),
            tension_seq=torch.zeros(4, 7, 1))
        hard = hard_gate_metrics(args, batch, rollout)
        self.assertTrue(bool(hard["gate_safe"][0]))
        self.assertFalse(bool(hard["payload_safe"][1]))
        self.assertFalse(bool(hard["gate_safe"][2]))
        self.assertLess(float(hard["clearance"][2].amax()), 0.0)


if __name__ == "__main__":
    unittest.main()
