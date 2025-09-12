from typer.testing import CliRunner
from nanoslurm.cli import app

runner = CliRunner()


def test_defaults_set_help_lists_keys():
    result = runner.invoke(app, ["defaults", "set", "--help"])
    assert result.exit_code == 0
    assert "Configuration key to set" in result.output
    assert "name (str)" in result.output
    assert "cpus (int)" in result.output


def test_defaults_set_unknown_key_error(tmp_path):
    env = {"XDG_CONFIG_HOME": str(tmp_path)}
    result = runner.invoke(app, ["defaults", "set", "foo", "bar"], env=env)
    assert result.exit_code != 0
    assert "Unknown key: foo" in result.output
    assert "Allowed keys" in result.output


def test_defaults_set_type_error(tmp_path):
    env = {"XDG_CONFIG_HOME": str(tmp_path)}
    result = runner.invoke(app, ["defaults", "set", "cpus", "notint"], env=env)
    assert result.exit_code != 0
    assert "cpus expects type int" in result.output
