"""Unit tests for the compile-cache host RAM probe (issue #1429)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

_REPO = Path(__file__).resolve().parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmark.tpu import compile_cache_host_ram_probe as probe  # noqa: E402

# Pure logic over a fake /proc tree, no server or TPU needed.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _write_proc(root: str, pid: int, *, cmdline: str, comm: str, rss_pages: int):
    d = Path(root, str(pid))
    d.mkdir()
    (d / "cmdline").write_bytes(cmdline.encode() + b"\0--extra\0")
    (d / "comm").write_text(comm + "\n")
    # statm: size resident shared text lib data dt
    (d / "statm").write_text(f"{rss_pages * 2} {rss_pages} 10 1 0 5 0\n")


class TestRankDiscovery(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.root = tempfile.mkdtemp()
        self.addCleanup(lambda: os.system(f"rm -rf {self.root}"))

    def test_finds_scheduler_processes_by_title_only(self):
        _write_proc(
            self.root, 300, cmdline="sglang::scheduler_TP1", comm="x", rss_pages=4
        )
        _write_proc(
            self.root, 200, cmdline="sglang::scheduler_TP0", comm="x", rss_pages=3
        )
        _write_proc(
            self.root, 400, cmdline="python3 -m sglang", comm="python3", rss_pages=9
        )
        Path(self.root, "self").mkdir()  # non-numeric entries are skipped
        procs = probe.find_rank_processes(self.root)
        self.assertEqual(
            [(p.pid, p.title) for p in procs],
            [
                (200, "sglang::scheduler_TP0"),
                (300, "sglang::scheduler_TP1"),
            ],
        )

    def test_comm_prefix_identifies_a_scheduler_without_cmdline(self):
        # cmdline unreadable (missing), comm truncated to 15 bytes by the kernel.
        d = Path(self.root, "500")
        d.mkdir()
        (d / "comm").write_text("sglang::schedul\n")  # codespell:ignore schedul
        (d / "statm").write_text("20 10 1 1 0 1 0\n")
        procs = probe.find_rank_processes(self.root)
        self.assertEqual([p.pid for p in procs], [500])

    def test_rss_reads_statm_pages_and_tolerates_exited_processes(self):
        _write_proc(
            self.root, 200, cmdline="sglang::scheduler_TP0", comm="x", rss_pages=3
        )
        self.assertEqual(probe.read_rss_bytes(200, self.root), 3 * probe._PAGE_SIZE)
        self.assertEqual(probe.read_rss_bytes(999, self.root), 0)


class TestGrowthEstimate(CustomTestCase):
    def _samples(self, per_rank_steps, ranks=8, base=1000 * probe._MIB):
        samples = []
        for i, step in enumerate(per_rank_steps):
            rss = {pid: int(base + step) for pid in range(ranks)}
            samples.append(probe.Sample(prompt_len=1 + 4 * i, rss_bytes=rss))
        return samples

    def test_linear_growth_is_attributed_per_rank_and_per_host(self):
        # Every rank grows 0.5 MiB per new length: the issue's 8 x 0.5 = 4 MiB/host.
        half = 0.5 * probe._MIB
        growth = probe.estimate_growth(self._samples([0, half, 2 * half, 3 * half]))
        self.assertEqual(growth.ranks, 8)
        self.assertEqual(growth.lengths, 4)
        self.assertAlmostEqual(growth.per_rank_per_length_bytes / probe._MIB, 0.5)
        self.assertAlmostEqual(growth.host_per_length_bytes / probe._MIB, 4.0)
        self.assertAlmostEqual(growth.host_total_bytes / probe._MIB, 12.0)
        self.assertIn("0.50 MiB per rank per length", probe.format_growth(growth))
        self.assertIn("4.00 MiB per host per length", probe.format_growth(growth))

    def test_flat_rss_reports_zero_growth(self):
        growth = probe.estimate_growth(self._samples([0, 0, 0]))
        self.assertEqual(growth.per_rank_per_length_bytes, 0.0)
        self.assertEqual(growth.host_total_bytes, 0)

    def test_empty_run_does_not_divide_by_zero(self):
        growth = probe.estimate_growth([])
        self.assertEqual((growth.ranks, growth.lengths), (0, 0))
        self.assertEqual(growth.host_per_length_bytes, 0.0)

    def test_rows_show_host_delta_since_the_previous_length(self):
        first, second = self._samples([0, 2 * probe._MIB])
        row = probe.format_row(first, second)
        self.assertIn(f"{second.prompt_len:>8}", row)
        self.assertIn("+16.0", row)  # 8 ranks x 2 MiB
        self.assertEqual(len(probe.format_header().split()), 8)


class TestCli(CustomTestCase):
    def test_exits_when_no_scheduler_process_exists(self):
        with tempfile.TemporaryDirectory() as root:
            with _CaptureStderr() as err:
                code = probe.main(["--proc-root", root, "--count", "0"])
        self.assertEqual(code, 2)
        self.assertIn("sglang::scheduler", err.text)

    def test_explicit_pids_skip_discovery_and_zero_count_prints_header_only(self):
        with _CaptureStdout() as out:
            code = probe.main(["--pids", "1", "--count", "0"])
        self.assertEqual(code, 0)
        self.assertEqual(out.text.strip(), probe.format_header().strip())


class _CaptureStderr:
    def __enter__(self):
        import io

        self._old = sys.stderr
        self._buf = io.StringIO()
        sys.stderr = self._buf
        return self

    def __exit__(self, *exc):
        sys.stderr = self._old
        self.text = self._buf.getvalue()
        return False


class _CaptureStdout(_CaptureStderr):
    def __enter__(self):
        import io

        self._old = sys.stdout
        self._buf = io.StringIO()
        sys.stdout = self._buf
        return self

    def __exit__(self, *exc):
        sys.stdout = self._old
        self.text = self._buf.getvalue()
        return False


if __name__ == "__main__":
    unittest.main()
