import importlib.util
from pathlib import Path

import numpy as np
import pytest
from ase.io import read


pytestmark = pytest.mark.skipif(importlib.util.find_spec("skimage") is None, reason="scikit-image is not installed")


def load_zeolite_analysis_module():
    module_path = Path("examples") / "zeolite_analysis.py"
    spec = importlib.util.spec_from_file_location("zeolite_analysis_example", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zeolite_analysis_convergence_and_plot(tmp_path):
    module = load_zeolite_analysis_module()
    atoms = read(Path("examples") / "BEA.cif")
    results = [module.analyze_zeolite(atoms, resolution, core_scale=0.9) for resolution in (1.5, 1.0)]

    for result in results:
        assert result["regions"] >= 1
        assert np.isfinite(result["pore_volume_cm3_g"])
        assert np.isfinite(result["surface_area_m2_g"])
        assert result["pore_volume_cm3_g"] > 0.0
        assert result["surface_area_m2_g"] > 0.0

    coarse_volume, finer_volume = [result["pore_volume_cm3_g"] for result in results]
    assert abs(finer_volume - coarse_volume) / finer_volume < 0.05

    output_path = tmp_path / "bea_convergence.png"
    module.plot_convergence(results, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
