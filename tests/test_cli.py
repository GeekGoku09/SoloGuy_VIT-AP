import subprocess
import sys


def test_cli_help():
    result = subprocess.run([sys.executable, "-m", "dtwin", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Digital Twin T2D CLI" in result.stdout
