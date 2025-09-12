from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from nanoslurm import stats


class Dummy:
    def __init__(self, stdout: str):
        self.stdout = stdout


def test_partition_caps_gpu_total(monkeypatch):
    sinfo_out = "p1|32/64|gpu:4|4\n"

    def fake_run(cmd, check=False):
        return Dummy(stdout=sinfo_out)

    monkeypatch.setattr(stats, "_run", fake_run)
    monkeypatch.setattr(stats, "_require", lambda cmd: None)

    caps = stats._partition_caps()
    assert caps["p1"]["gpus"] == 16
    assert caps["p1"]["cpus"] == 64
