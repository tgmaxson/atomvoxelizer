import subprocess
import sys

import pytest


def test_top_level_import_does_not_import_optional_backends():
    script = (
        "import sys; "
        "import atomvoxelizer; "
        "print('numba' in sys.modules, 'cupy' in sys.modules, 'taichi' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

    assert result.stdout.strip() == "False False False"


def test_lazy_numba_backend_imports_and_unknown_attribute():
    pytest.importorskip("numba")

    import atomvoxelizer

    assert atomvoxelizer.VoxelGridNumba.backend_name == "numba"
    assert atomvoxelizer.FieldVoxelGridNumba.backend_name == "numba-field"
    assert atomvoxelizer.VectorVoxelGridNumba.backend_name == "numba-field"

    with pytest.raises(AttributeError, match="does_not_exist"):
        getattr(atomvoxelizer, "does_not_exist")
