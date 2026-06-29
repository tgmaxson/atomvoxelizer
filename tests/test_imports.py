import subprocess
import sys


def test_top_level_import_does_not_import_optional_backends():
    script = (
        "import sys; "
        "import atomvoxelizer; "
        "print('numba' in sys.modules, 'cupy' in sys.modules, 'taichi' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

    assert result.stdout.strip() == "False False False"
