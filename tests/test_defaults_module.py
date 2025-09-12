from nanoslurm import DEFAULTS, KEY_TYPES, load_defaults, save_defaults


def test_roundtrip_load_save(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfg = load_defaults()
    assert cfg["name"] == DEFAULTS["name"]
    cfg["cpus"] = 42
    save_defaults(cfg)
    cfg2 = load_defaults()
    assert cfg2["cpus"] == 42
    assert "name" in KEY_TYPES
