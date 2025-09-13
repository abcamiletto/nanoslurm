from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from nanoslurm import stats


def test_partition_caps_gpu_total(monkeypatch):
    monkeypatch.setattr(
        stats,
        "_sinfo",
        lambda **kwargs: [{"part": "p1", "cpus": "32/64", "gres": "gpu:4", "nodes": "4"}],
    )
    caps = stats._partition_caps()
    assert caps["p1"]["gpus"] == 16
    assert caps["p1"]["cpus"] == 64
