"""CLI smoke tests."""

import subprocess
import sys


def test_hello_command():
    result = subprocess.run([sys.executable, "-m", "app.cli", "hello"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Hello" in result.stdout


def test_help_command():
    result = subprocess.run([sys.executable, "-m", "app.cli", "play", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--rounds" in result.stdout
