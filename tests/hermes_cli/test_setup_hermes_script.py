from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO_ROOT / "setup-hermes.sh"


def test_setup_hermes_script_is_valid_shell():
    result = subprocess.run(["bash", "-n", str(SETUP_SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_setup_hermes_script_has_termux_path():
    content = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "is_termux()" in content
    assert ".[termux]" in content
    assert "constraints-termux.txt" in content
    assert "$PREFIX/bin" in content


def test_setup_hermes_script_supports_manual_install_flags():
    content = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "--dev" in content
    assert "--run-tests" in content
    assert "--skip-setup" in content
    assert "--skip-ripgrep" in content
    assert ".[all,dev]" in content
    assert 'scripts/run_tests.sh"' in content


def test_readme_manual_path_uses_setup_script():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "./setup-hermes.sh --dev --run-tests --skip-setup --skip-ripgrep" in readme
