from pathlib import Path
import unittest

from rcp_backend import (
    RCPConfig, build_scp_command, build_submit_command, gpu_request_args,
    parse_workload_states, remote_run_dir,
)


class RCPBackendTests(unittest.TestCase):
    def test_fractional_and_whole_gpu_flags(self):
        self.assertEqual(gpu_request_args("0.1"), ["--gpu-portion-request", "0.1"])
        self.assertEqual(gpu_request_args("1"), ["--gpu-devices-request", "1"])
        self.assertEqual(gpu_request_args(2), ["--gpu-devices-request", "2"])
        self.assertEqual(gpu_request_args("0"), [])

    def test_submit_preserves_launcher_argv(self):
        cfg = RCPConfig(job_name="gate-demo")
        argv = ["--no_show_plots", "--run_id", "run one", "--variants", "nominal,context"]
        cmd = build_submit_command(cfg, argv)
        remote = cmd[-1]
        self.assertIn("--run_id 'run one'", remote)
        self.assertIn("--variants nominal,context", remote)
        self.assertEqual(cmd[cmd.index("--gpu-portion-request") + 1], "0.1")
        self.assertEqual(cmd[cmd.index("--cpu-core-request") + 1], "4")
        self.assertEqual(cmd[cmd.index("--cpu-memory-request") + 1], "16G")
        self.assertEqual(cmd[-4:-1], ["--", "bash", "-lc"])

    def test_parse_workload_states_from_cli_table(self):
        table = (
            " Workload               Type        Framework   Status      Project              \n"
            "──────────────────────────────────────────────────────────────────────────────────\n"
            " arch-contextual-ssm    Training    Runai       Completed   sci-sti-gft-lmassai  \n"
            " arch-mad-context       Training    Runai       Running     sci-sti-gft-lmassai  \n"
            "\n"
        )
        self.assertEqual(parse_workload_states(table),
                         [("arch-contextual-ssm", "Completed"),
                          ("arch-mad-context", "Running")])

    def test_result_paths_and_scp_destination(self):
        cfg = RCPConfig(job_name="gate-demo")
        expected = "/home/lmassai/Performance_Boosting/experiments/contextual_pb_gate_ssm/runs/r42"
        self.assertEqual(str(remote_run_dir(cfg, "r42")), expected)
        cmd = build_scp_command(cfg, "r42", Path("/tmp/stage"))
        self.assertEqual(cmd[-1], "/tmp/stage")
        self.assertIn(expected, cmd[-2])

    def test_script_parameterization_targets_other_experiments(self):
        payload = "experiments/contextual_pb_payload_ssm/Moving_payload_exp.py"
        cfg = RCPConfig(job_name="payload-demo", script=payload)
        self.assertEqual(
            str(remote_run_dir(cfg, "r7")),
            "/home/lmassai/Performance_Boosting/experiments/contextual_pb_payload_ssm/runs/r7")
        remote = build_submit_command(cfg, ["--run_id", "r7"])[-1]
        self.assertIn(payload, remote)
        self.assertNotIn("Moving_gate_exp.py", remote)
        # default stays on the gate experiment (backward compatibility)
        legacy = build_submit_command(RCPConfig(job_name="gate-demo"), [])[-1]
        self.assertIn("experiments/contextual_pb_gate_ssm/Moving_gate_exp.py", legacy)


if __name__ == "__main__":
    unittest.main()
