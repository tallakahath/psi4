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

_NH2 = """
    0 2
    N
    H 1 1.03
    H 1 1.03 2 103.0
"""


@pytest.mark.findif
@pytest.mark.parametrize("func,ref_key,geom", [
    pytest.param("SVWN", "rks", _HCN, id="SVWN RKS"),
    pytest.param("BLYP", "rks", _HCN, id="BLYP RKS"),
    pytest.param("TPSS", "rks", _HCN, id="TPSS RKS"),
    pytest.param("TPSS", "uks", _NH2, id="TPSS UKS doublet"),
])
def test_dft_grid_response_gradient(func, ref_key, geom):
    # A deliberately coarse grid, where the neglected grid response leaves the
    # analytic gradient 1e-4 to 2e-3 away from finite differences; with
    # DFT_GRID_RESPONSE the agreement must return to the findif noise floor.
    psi4.set_num_threads(1)

    mol = psi4.geometry(geom)

    psi4.set_options({
        "basis": "cc-pvdz",
        "reference": ref_key,
        "scf_type": "df",
        "dft_radial_points": 35,
        "dft_spherical_points": 110,
        "dft_grid_response": True,
        "e_convergence": 9,
        "d_convergence": 9,
        "points": 5,
    })

    analytic_gradient = psi4.gradient(func, dertype=1)

    # The grid response restores exact translational invariance of the XC part
    net_force = np.abs(analytic_gradient.np.sum(axis=0)).max()
    assert net_force < 1.0e-10, f"net translation {net_force:.2e}"

    psi4.core.clean()
    findif_gradient = psi4.gradient(func, dertype=0)

    assert compare_values(findif_gradient, analytic_gradient, 5,
                          f"{func} {ref_key.upper()} grid-response analytic vs. findif gradient")
    psi4.core.clean()
