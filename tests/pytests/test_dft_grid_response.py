import numpy as np
import pytest

import psi4

from utils import compare_values

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.scf, pytest.mark.dft]

_HCN = """
    0 1
    N   -0.0034118    3.5353926    0.0000000
    C    0.0751963    2.3707040    0.0000000
    H    0.1476295    1.3052847    0.0000000
"""


@pytest.mark.findif
@pytest.mark.parametrize("func,ref_key,geom,scheme", [
    pytest.param("SVWN", "rks", _HCN, "TREUTLER", id="SVWN RKS"),
    pytest.param("SVWN", "rks", _HCN, "STRATMANN", id="SVWN RKS Stratmann"),
    pytest.param("SVWN", "rks", _HCN, "SBECKE", id="SVWN RKS smooth Becke"),
    pytest.param("BLYP", "rks", _HCN, "TREUTLER", id="BLYP RKS"),
])
def test_dft_grid_response_gradient(func, ref_key, geom, scheme):
    # Deliberately coarse grid so the neglected grid response is visible; with
    # DFT_GRID_RESPONSE the agreement must return to the findif noise floor.
    psi4.set_num_threads(1)

    mol = psi4.geometry(geom)

    psi4.set_options({
        "basis": "cc-pvdz",
        "reference": ref_key,
        "scf_type": "df",
        "dft_radial_points": 35,
        "dft_spherical_points": 110,
        "dft_nuclear_scheme": scheme,
        "e_convergence": 9,
        "d_convergence": 9,
        "points": 5,
    })

    psi4.set_options({"dft_grid_response": False})
    base_gradient = psi4.gradient(func, dertype=1)
    psi4.core.clean()

    psi4.set_options({"dft_grid_response": True})
    analytic_gradient = psi4.gradient(func, dertype=1)
    psi4.core.clean()

    # The grid response restores exact translational invariance of the XC part
    net_force = np.abs(analytic_gradient.np.sum(axis=0)).max()
    assert net_force < 1.0e-10, f"net translation {net_force:.2e}"

    findif_gradient = psi4.gradient(func, dertype=0)
    psi4.core.clean()

    # Negative control: without the response the coarse grid must be visible
    base_dev = np.abs(base_gradient.np - findif_gradient.np).max()
    assert base_dev > 1.0e-6, f"grid response not exposed: base deviation {base_dev:.2e}"

    assert compare_values(findif_gradient, analytic_gradient, 5,
                          f"{func} {ref_key.upper()} {scheme} grid-response analytic vs. findif gradient")
