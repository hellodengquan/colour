"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.itur_bt_2100` module.
"""

from __future__ import annotations

import typing

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    eotf_BT2100_PQ,
    eotf_inverse_BT2100_PQ,
    oetf_BT2100_HLG,
    oetf_BT2100_PQ,
    oetf_inverse_BT2100_HLG,
    oetf_inverse_BT2100_PQ,
    ootf_BT2100_PQ,
    ootf_inverse_BT2100_PQ,
)
from colour.models.rgb.transfer_functions.itur_bt_2100 import (
    black_level_lift_BT2100_HLG,
    eotf_BT2100_HLG,
    eotf_BT2100_HLG_1,
    eotf_BT2100_HLG_2,
    eotf_inverse_BT2100_HLG,
    eotf_inverse_BT2100_HLG_1,
    eotf_inverse_BT2100_HLG_2,
    gamma_function_BT2100_HLG,
    ootf_BT2100_HLG,
    ootf_BT2100_HLG_1,
    ootf_BT2100_HLG_2,
    ootf_inverse_BT2100_HLG,
    ootf_inverse_BT2100_HLG_1,
    ootf_inverse_BT2100_HLG_2,
)
from colour.utilities import (
    as_ndarray,
    domain_range_scale,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_reshape,
)

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestOetf_BT2100_PQ",
    "TestOetf_inverse_BT2100_PQ",
    "TestEotf_BT2100_PQ",
    "TestEotf_inverse_BT2100_PQ",
    "TestOotf_BT2100_PQ",
    "TestOotf_inverse_BT2100_PQ",
    "TestGamma_function_BT2100_HLG",
    "TestOetf_BT2100_HLG",
    "TestOetf_inverse_BT2100_HLG",
    "TestEotf_BT2100_HLG_1",
    "TestEotf_BT2100_HLG_2",
    "TestEotf_inverse_BT2100_HLG_1",
    "TestEotf_inverse_BT2100_HLG_2",
    "TestOotf_BT2100_HLG_1",
    "TestOotf_BT2100_HLG_2",
    "TestOotfBT2100HLG",
    "TestOotf_inverse_BT2100_HLG_1",
    "TestOotf_inverse_BT2100_HLG_2",
    "TestOotfInverseBT2100HLG",
    "TestBlack_level_lift_BT2100_HLG",
    "TestBT2100HLGMonitorAbsoluteLuminance",
    "TestBT2100HLGDisplayToSceneRecovery",
    "TestBT2100HLGMonitoringChainRoundTrip",
]


class TestOetf_BT2100_PQ:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_PQ` definition unit tests methods.
    """

    def test_oetf_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_PQ` definition.
        """

        xp_assert_close(
            oetf_BT2100_PQ(xp_as_array(0.0, xp=xp)),
            0.000000730955903,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_BT2100_PQ(xp_as_array(0.1, xp=xp)),
            0.724769816665726,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_BT2100_PQ(xp_as_array(1.0, xp=xp)),
            0.999999934308041,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_PQ` definition n-dimensional arrays support.
        """

        E = 0.1
        E_p = as_ndarray(oetf_BT2100_PQ(xp_as_array(E, xp=xp)))

        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        xp_assert_close(oetf_BT2100_PQ(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(oetf_BT2100_PQ(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(oetf_BT2100_PQ(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_PQ` definition domain and range scale support.
        """

        E = 0.1
        E_p = as_ndarray(oetf_BT2100_PQ(xp_as_array(E, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_BT2100_PQ(xp_as_array(E * factor, xp=xp)),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_BT2100_PQ(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_PQ` definition nan support.
        """

        oetf_BT2100_PQ(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_BT2100_PQ:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_PQ` definition unit tests methods.
    """

    def test_oetf_inverse_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_PQ` definition.
        """

        xp_assert_close(
            oetf_inverse_BT2100_PQ(xp_as_array(0.000000730955903, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_BT2100_PQ(xp_as_array(0.724769816665726, xp=xp)),
            0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_BT2100_PQ(xp_as_array(0.999999934308041, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_inverse_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_PQ` definition n-dimensional arrays support.
        """

        E_p = 0.724769816665726
        E = as_ndarray(oetf_inverse_BT2100_PQ(xp_as_array(E_p, xp=xp)))

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        xp_assert_close(oetf_inverse_BT2100_PQ(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        xp_assert_close(oetf_inverse_BT2100_PQ(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(oetf_inverse_BT2100_PQ(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_inverse_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_PQ` definition domain and range scale support.
        """

        E_p = 0.724769816665726
        E = as_ndarray(oetf_inverse_BT2100_PQ(xp_as_array(E_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_inverse_BT2100_PQ(xp_as_array(E_p * factor, xp=xp)),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_inverse_BT2100_PQ(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_PQ` definition nan support.
        """

        oetf_inverse_BT2100_PQ(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_BT2100_PQ:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_PQ` definition unit tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(1e-1)
    def test_eotf_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_PQ` definition.
        """

        xp_assert_close(
            eotf_BT2100_PQ(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_BT2100_PQ(xp_as_array(0.724769816665726, xp=xp)),
            779.98836083408537,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_BT2100_PQ(xp_as_array(1.0, xp=xp)),
            10000.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_PQ` definition n-dimensional arrays support.
        """

        E_p = 0.724769816665726
        F_D = as_ndarray(eotf_BT2100_PQ(xp_as_array(E_p, xp=xp)))

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        xp_assert_close(eotf_BT2100_PQ(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_BT2100_PQ(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_BT2100_PQ(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_PQ` definition domain and range scale support.
        """

        E_p = 0.724769816665726
        F_D = as_ndarray(eotf_BT2100_PQ(xp_as_array(E_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_BT2100_PQ(xp_as_array(E_p * factor, xp=xp)),
                    F_D * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_BT2100_PQ(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_PQ` definition nan support.
        """

        eotf_BT2100_PQ(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_inverse_BT2100_PQ:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_PQ` definition unit tests methods.
    """

    def test_eotf_inverse_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_PQ` definition.
        """

        xp_assert_close(
            eotf_inverse_BT2100_PQ(xp_as_array(0.0, xp=xp)),
            0.000000730955903,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT2100_PQ(xp_as_array(779.98836083408537, xp=xp)),
            0.724769816665726,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT2100_PQ(xp_as_array(10000.0, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_inverse_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_PQ` definition n-dimensional arrays support.
        """

        F_D = 779.98836083408537
        E_p = as_ndarray(eotf_inverse_BT2100_PQ(xp_as_array(F_D, xp=xp)))

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        xp_assert_close(eotf_inverse_BT2100_PQ(F_D), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_inverse_BT2100_PQ(F_D), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_inverse_BT2100_PQ(F_D), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_inverse_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_PQ` definition domain and range scale support.
        """

        F_D = 779.98836083408537
        E_p = as_ndarray(eotf_inverse_BT2100_PQ(xp_as_array(F_D, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_inverse_BT2100_PQ(xp_as_array(F_D * factor, xp=xp)),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_inverse_BT2100_PQ(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_PQ` definition nan support.
        """

        eotf_inverse_BT2100_PQ(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_BT2100_PQ:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_PQ` definition unit tests methods.
    """

    def test_ootf_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_PQ` definition.
        """

        xp_assert_close(
            ootf_BT2100_PQ(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_BT2100_PQ(xp_as_array(0.1, xp=xp)),
            779.98836083411584,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_BT2100_PQ(xp_as_array(1.0, xp=xp)),
            9999.993723673924300,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_ootf_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_PQ` definition n-dimensional arrays support.
        """

        E = 0.1
        F_D = as_ndarray(ootf_BT2100_PQ(xp_as_array(E, xp=xp)))

        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        xp_assert_close(ootf_BT2100_PQ(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        xp_assert_close(ootf_BT2100_PQ(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(ootf_BT2100_PQ(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_ootf_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_PQ` definition domain and range scale support.
        """

        E = 0.1
        F_D = as_ndarray(ootf_BT2100_PQ(xp_as_array(E, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    ootf_BT2100_PQ(xp_as_array(E * factor, xp=xp)),
                    F_D * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_ootf_BT2100_PQ(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_PQ` definition nan support.
        """

        ootf_BT2100_PQ(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_inverse_BT2100_PQ:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_PQ` definition unit tests methods.
    """

    def test_ootf_inverse_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_PQ` definition.
        """

        xp_assert_close(
            ootf_inverse_BT2100_PQ(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_inverse_BT2100_PQ(xp_as_array(779.98836083411584, xp=xp)),
            0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_inverse_BT2100_PQ(xp_as_array(9999.993723673924300, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_ootf_inverse_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_PQ` definition n-dimensional arrays support.
        """

        F_D = 779.98836083411584
        E = as_ndarray(ootf_inverse_BT2100_PQ(xp_as_array(F_D, xp=xp)))

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        xp_assert_close(ootf_inverse_BT2100_PQ(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        xp_assert_close(ootf_inverse_BT2100_PQ(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(ootf_inverse_BT2100_PQ(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_ootf_inverse_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_PQ` definition domain and range scale support.
        """

        F_D = 779.98836083411584
        E = as_ndarray(ootf_inverse_BT2100_PQ(xp_as_array(F_D, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    ootf_inverse_BT2100_PQ(xp_as_array(F_D * factor, xp=xp)),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_ootf_inverse_BT2100_PQ(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_PQ` definition nan support.
        """

        ootf_inverse_BT2100_PQ(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestGamma_function_BT2100_HLG:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
gamma_function_BT2100_HLG` definition unit tests methods.
    """

    def test_gamma_function_BT2100_HLG(self, xp: ModuleType) -> None:  # noqa: ARG002
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
gamma_function_BT2100_HLG` definition.
        """

        xp_assert_close(
            gamma_function_BT2100_HLG(1000.0),
            1.2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            gamma_function_BT2100_HLG(2000.0),
            1.326432598178872,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            gamma_function_BT2100_HLG(4000.0),
            1.452865196357744,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            gamma_function_BT2100_HLG(10000.0),
            1.619999999999999,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestOetf_BT2100_HLG:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_HLG` definition unit tests methods.
    """

    def test_oetf_BT2100_HLG(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_HLG` definition.
        """

        xp_assert_close(
            oetf_BT2100_HLG(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_BT2100_HLG(xp_as_array(0.18 / 12, xp=xp)),
            0.212132034355964,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_BT2100_HLG(xp_as_array(1.0, xp=xp)),
            0.999999995536569,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_BT2100_HLG(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_HLG` definition n-dimensional arrays support.
        """

        E = 0.18 / 12
        E_p = as_ndarray(oetf_BT2100_HLG(xp_as_array(E, xp=xp)))

        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        xp_assert_close(oetf_BT2100_HLG(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(oetf_BT2100_HLG(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(oetf_BT2100_HLG(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_BT2100_HLG(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_HLG` definition domain and range scale support.
        """

        E = 0.18 / 12
        E_p = as_ndarray(oetf_BT2100_HLG(xp_as_array(E, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_BT2100_HLG(xp_as_array(E * factor, xp=xp)),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_BT2100_HLG(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_HLG` definition nan support.
        """

        oetf_BT2100_HLG(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_BT2100_HLG:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_HLG` definition unit tests methods.
    """

    def test_oetf_inverse_BT2100_HLG(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_HLG` definition.
        """

        xp_assert_close(
            oetf_inverse_BT2100_HLG(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_BT2100_HLG(xp_as_array(0.212132034355964, xp=xp)),
            0.18 / 12,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_BT2100_HLG(xp_as_array(0.999999995536569, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_inverse_BT2100_HLG(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_HLG` definition n-dimensional arrays support.
        """

        E_p = 0.212132034355964
        E = as_ndarray(oetf_inverse_BT2100_HLG(xp_as_array(E_p, xp=xp)))

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        xp_assert_close(oetf_inverse_BT2100_HLG(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        xp_assert_close(oetf_inverse_BT2100_HLG(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(oetf_inverse_BT2100_HLG(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_inverse_BT2100_HLG(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_HLG` definition domain and range scale support.
        """

        E_p = 0.212132034355964
        E = as_ndarray(oetf_inverse_BT2100_HLG(xp_as_array(E_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_inverse_BT2100_HLG(xp_as_array(E_p * factor, xp=xp)),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_inverse_BT2100_HLG(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_HLG` definition nan support.
        """

        oetf_inverse_BT2100_HLG(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_BT2100_HLG_1:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_1` definition unit tests methods.
    """

    def test_eotf_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_1` definition.
        """

        xp_assert_close(
            eotf_BT2100_HLG_1(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_BT2100_HLG_1(xp_as_array(0.212132034355964, xp=xp)),
            6.476039825649814,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_BT2100_HLG_1(xp_as_array(1.0, xp=xp)),
            1000.000032321769100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_BT2100_HLG_1(xp_as_array(0.212132034355964, xp=xp), 0.001, 10000, 1.4),
            27.96039175299561,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_1` definition n-dimensional arrays support.
        """

        E_p = 0.212132034355964
        F_D = as_ndarray(eotf_BT2100_HLG_1(xp_as_array(E_p, xp=xp)))

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        xp_assert_close(eotf_BT2100_HLG_1(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_BT2100_HLG_1(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_BT2100_HLG_1(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (6, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (6, 1), xp=xp)
        xp_assert_close(eotf_BT2100_HLG_1(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_as_array([0.25, 0.50, 0.75], xp=xp)
        F_D = np.array([12.49759413, 49.99037650, 158.94693786])
        xp_assert_close(eotf_BT2100_HLG_1(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6, 1))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6, 1))
        xp_assert_close(eotf_BT2100_HLG_1(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(eotf_BT2100_HLG_1(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_1` definition domain and range scale support.
        """

        E_p = 0.212132034355964
        F_D = as_ndarray(eotf_BT2100_HLG_1(xp_as_array(E_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_BT2100_HLG_1(xp_as_array(E_p * factor, xp=xp)),
                    F_D * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_BT2100_HLG_1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_1` definition nan support.
        """

        eotf_BT2100_HLG_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_BT2100_HLG_2:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_2` definition unit tests methods.
    """

    def test_eotf_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_2` definition.
        """

        xp_assert_close(
            eotf_BT2100_HLG_2(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_BT2100_HLG_2(xp_as_array(0.212132034355964, xp=xp)),
            6.476039825649814,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_BT2100_HLG_2(xp_as_array(1.0, xp=xp)),
            1000.000032321769100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_BT2100_HLG_2(xp_as_array(0.212132034355964, xp=xp), 0.001, 10000, 1.4),
            29.581261576946076,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_2` definition n-dimensional arrays support.
        """

        E_p = 0.212132034355964
        F_D = as_ndarray(eotf_BT2100_HLG_2(xp_as_array(E_p, xp=xp)))

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        xp_assert_close(eotf_BT2100_HLG_2(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_BT2100_HLG_2(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_BT2100_HLG_2(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (6, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (6, 1), xp=xp)
        xp_assert_close(eotf_BT2100_HLG_2(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_as_array([0.25, 0.50, 0.75], xp=xp)
        F_D = np.array([12.49759413, 49.99037650, 158.94693786])
        xp_assert_close(eotf_BT2100_HLG_2(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6, 1))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6, 1))
        xp_assert_close(eotf_BT2100_HLG_2(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(eotf_BT2100_HLG_2(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_2` definition domain and range scale support.
        """

        E_p = 0.212132034355964
        F_D = as_ndarray(eotf_BT2100_HLG_2(xp_as_array(E_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_BT2100_HLG_2(xp_as_array(E_p * factor, xp=xp)),
                    F_D * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_BT2100_HLG_2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_2` definition nan support.
        """

        eotf_BT2100_HLG_2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_inverse_BT2100_HLG_1:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_1` definition unit tests methods.
    """

    def test_eotf_inverse_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_1` definition.
        """

        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(xp_as_array(6.476039825649814, xp=xp)),
            0.212132034355964,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(xp_as_array(1000.000032321769100, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(
                xp_as_array(27.96039175299561, xp=xp), 0.001, 10000, 1.4
            ),
            0.212132034355964,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_inverse_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_1` definition n-dimensional arrays support.
        """

        F_D = 6.476039825649814
        E_p = as_ndarray(eotf_inverse_BT2100_HLG_1(xp_as_array(F_D, xp=xp)))

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (6, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (6, 1), xp=xp)
        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_as_array([12.49759413, 49.99037650, 158.94693786], xp=xp)
        E_p = np.array([0.25, 0.50, 0.75])
        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6, 1))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6, 1))
        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_eotf_inverse_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_1` definition domain and range scale support.
        """

        F_D = 6.476039825649814
        E_p = as_ndarray(eotf_inverse_BT2100_HLG_1(xp_as_array(F_D, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_inverse_BT2100_HLG_1(xp_as_array(F_D * factor, xp=xp)),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_inverse_BT2100_HLG_1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_1` definition nan support.
        """

        eotf_inverse_BT2100_HLG_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_inverse_BT2100_HLG_2:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_2` definition unit tests methods.
    """

    def test_eotf_inverse_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_2` definition.
        """

        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(xp_as_array(6.476039825649814, xp=xp)),
            0.212132034355964,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(xp_as_array(1000.000032321769100, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(
                xp_as_array(29.581261576946076, xp=xp), 0.001, 10000, 1.4
            ),
            0.212132034355964,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_inverse_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_2` definition n-dimensional arrays support.
        """

        F_D = 6.476039825649814
        E_p = as_ndarray(eotf_inverse_BT2100_HLG_2(xp_as_array(F_D, xp=xp)))

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (6, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (6, 1), xp=xp)
        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_as_array([12.49759413, 49.99037650, 158.94693786], xp=xp)
        E_p = np.array([0.25, 0.50, 0.75])
        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6, 1))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6, 1))
        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_eotf_inverse_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_2` definition domain and range scale support.
        """

        F_D = 6.476039825649814
        E_p = as_ndarray(eotf_inverse_BT2100_HLG_2(xp_as_array(F_D, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_inverse_BT2100_HLG_2(xp_as_array(F_D * factor, xp=xp)),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_inverse_BT2100_HLG_2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_2` definition nan support.
        """

        eotf_inverse_BT2100_HLG_2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_BT2100_HLG_1:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_1` definition unit tests methods.
    """

    def test_ootf_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_1` definition.
        """

        xp_assert_close(
            ootf_BT2100_HLG_1(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_BT2100_HLG_1(xp_as_array(0.1, xp=xp)),
            63.095734448019336,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_BT2100_HLG_1(xp_as_array(1.0, xp=xp)),
            1000.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_BT2100_HLG_1(xp_as_array(0.1, xp=xp), 0.001, 10000, 1.4),
            398.108130742780300,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.array(
            [
                [45.884942278760597, 0.000000000000000, -45.884942278760597],
                [
                    -63.095734448019336,
                    -63.095734448019336,
                    -63.095734448019336,
                ],
                [63.095734448019336, 63.095734448019336, 63.095734448019336],
                [51.320396090100672, -51.320396090100672, 51.320396090100672],
            ],
        )
        xp_assert_close(
            ootf_BT2100_HLG_1(
                xp_as_array(
                    [
                        [0.1, 0.0, -0.1],
                        [-0.1, -0.1, -0.1],
                        [0.1, 0.1, 0.1],
                        [0.1, -0.1, 0.1],
                    ],
                    xp=xp,
                )
            ),
            a,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_ootf_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_1` definition n-dimensional arrays support.
        """

        E = 0.1
        F_D = as_ndarray(ootf_BT2100_HLG_1(xp_as_array(E, xp=xp)))

        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        xp_assert_close(ootf_BT2100_HLG_1(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        xp_assert_close(ootf_BT2100_HLG_1(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(ootf_BT2100_HLG_1(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (6, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (6, 1), xp=xp)
        xp_assert_close(ootf_BT2100_HLG_1(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_as_array([0.25, 0.50, 0.75], xp=xp)
        F_D = np.array([213.01897444, 426.03794887, 639.05692331])
        xp_assert_close(ootf_BT2100_HLG_1(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp.tile(xp_as_array(E, xp=xp), (6, 1))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6, 1))
        xp_assert_close(ootf_BT2100_HLG_1(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(ootf_BT2100_HLG_1(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_ootf_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_1` definition domain and range scale support.
        """

        E = 0.1
        F_D = as_ndarray(ootf_BT2100_HLG_1(xp_as_array(E, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    ootf_BT2100_HLG_1(xp_as_array(E * factor, xp=xp)),
                    F_D * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_ootf_BT2100_HLG_1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_1` definition nan support.
        """

        ootf_BT2100_HLG_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_BT2100_HLG_2:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_2` definition unit tests methods.
    """

    def test_ootf_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_2` definition.
        """

        xp_assert_close(
            ootf_BT2100_HLG_2(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_BT2100_HLG_2(xp_as_array(0.1, xp=xp)),
            63.095734448019336,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_BT2100_HLG_2(xp_as_array(1.0, xp=xp)),
            1000.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_BT2100_HLG_2(xp_as_array(0.1, xp=xp), 10000, 1.4),
            398.107170553497380,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.array(
            [
                [45.884942278760597, 0.000000000000000, -45.884942278760597],
                [
                    -63.095734448019336,
                    -63.095734448019336,
                    -63.095734448019336,
                ],
                [63.095734448019336, 63.095734448019336, 63.095734448019336],
                [51.320396090100672, -51.320396090100672, 51.320396090100672],
            ],
        )
        xp_assert_close(
            ootf_BT2100_HLG_2(
                xp_as_array(
                    [
                        [0.1, 0.0, -0.1],
                        [-0.1, -0.1, -0.1],
                        [0.1, 0.1, 0.1],
                        [0.1, -0.1, 0.1],
                    ],
                    xp=xp,
                )
            ),
            a,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_ootf_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_2` definition n-dimensional arrays support.
        """

        E = 0.1
        F_D = as_ndarray(ootf_BT2100_HLG_2(xp_as_array(E, xp=xp)))

        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        xp_assert_close(ootf_BT2100_HLG_2(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        xp_assert_close(ootf_BT2100_HLG_2(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(ootf_BT2100_HLG_2(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (6, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (6, 1), xp=xp)
        xp_assert_close(ootf_BT2100_HLG_2(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_as_array([0.25, 0.50, 0.75], xp=xp)
        F_D = np.array([213.01897444, 426.03794887, 639.05692331])
        xp_assert_close(ootf_BT2100_HLG_2(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp.tile(xp_as_array(E, xp=xp), (6, 1))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6, 1))
        xp_assert_close(ootf_BT2100_HLG_2(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(ootf_BT2100_HLG_2(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_ootf_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_2` definition domain and range scale support.
        """

        E = 0.1
        F_D = as_ndarray(ootf_BT2100_HLG_1(xp_as_array(E, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    ootf_BT2100_HLG_1(xp_as_array(E * factor, xp=xp)),
                    F_D * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_ootf_BT2100_HLG_1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_1` definition nan support.
        """

        ootf_BT2100_HLG_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_inverse_BT2100_HLG_1:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_1` definition unit tests methods.
    """

    def test_ootf_inverse_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_1` definition.
        """

        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(xp_as_array(63.095734448019336, xp=xp)),
            0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(xp_as_array(1000.0, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(
                xp_as_array(398.108130742780300, xp=xp), 0.001, 10000, 1.4
            ),
            0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.array(
            [
                [45.884942278760597, 0.000000000000000, -45.884942278760597],
                [
                    -63.095734448019336,
                    -63.095734448019336,
                    -63.095734448019336,
                ],
                [63.095734448019336, 63.095734448019336, 63.095734448019336],
                [51.320396090100672, -51.320396090100672, 51.320396090100672],
            ]
        )
        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(xp_as_array(a, xp=xp)),
            [
                [0.1, 0.0, -0.1],
                [-0.1, -0.1, -0.1],
                [0.1, 0.1, 0.1],
                [0.1, -0.1, 0.1],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_ootf_inverse_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_1` definition n-dimensional arrays support.
        """

        F_D = 63.095734448019336
        E = as_ndarray(ootf_inverse_BT2100_HLG_1(xp_as_array(F_D, xp=xp)))

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (6, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (6, 1), xp=xp)
        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_as_array([213.01897444, 426.03794887, 639.05692331], xp=xp)
        E = np.array([0.25, 0.50, 0.75])
        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6, 1))
        E = xp.tile(xp_as_array(E, xp=xp), (6, 1))
        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_ootf_inverse_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_1` definition domain and range scale support.
        """

        F_D = 63.095734448019336
        E = as_ndarray(ootf_inverse_BT2100_HLG_1(xp_as_array(F_D, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    ootf_inverse_BT2100_HLG_1(xp_as_array(F_D * factor, xp=xp)),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_ootf_inverse_BT2100_HLG_1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_1` definition nan support.
        """

        ootf_inverse_BT2100_HLG_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_inverse_BT2100_HLG_2:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_2` definition unit tests methods.
    """

    def test_ootf_inverse_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_2` definition.
        """

        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(xp_as_array(63.095734448019336, xp=xp)),
            0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(xp_as_array(1000.0, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(
                xp_as_array(398.107170553497380, xp=xp), 10000, 1.4
            ),
            0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.array(
            [
                [45.884942278760597, 0.000000000000000, -45.884942278760597],
                [
                    -63.095734448019336,
                    -63.095734448019336,
                    -63.095734448019336,
                ],
                [63.095734448019336, 63.095734448019336, 63.095734448019336],
                [51.320396090100672, -51.320396090100672, 51.320396090100672],
            ]
        )
        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(xp_as_array(a, xp=xp)),
            [
                [0.1, 0.0, -0.1],
                [-0.1, -0.1, -0.1],
                [0.1, 0.1, 0.1],
                [0.1, -0.1, 0.1],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_ootf_inverse_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_2` definition n-dimensional arrays support.
        """

        F_D = 63.095734448019336
        E = as_ndarray(ootf_inverse_BT2100_HLG_2(xp_as_array(F_D, xp=xp)))

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (6, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (6, 1), xp=xp)
        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_as_array([213.01897444, 426.03794887, 639.05692331], xp=xp)
        E = np.array([0.25, 0.50, 0.75])
        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6, 1))
        E = xp.tile(xp_as_array(E, xp=xp), (6, 1))
        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_ootf_inverse_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_2` definition domain and range scale support.
        """

        F_D = 63.095734448019336
        E = as_ndarray(ootf_inverse_BT2100_HLG_2(xp_as_array(F_D, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    ootf_inverse_BT2100_HLG_2(xp_as_array(F_D * factor, xp=xp)),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_ootf_inverse_BT2100_HLG_2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_2` definition nan support.
        """

        ootf_inverse_BT2100_HLG_2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotfBT2100HLG:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG` definition unit tests methods.
    """

    def test_ootf_BT2100_HLG(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG` definition.
        """

        # Test default method (ITU-R BT.2100-2)
        xp_assert_close(
            ootf_BT2100_HLG(xp_as_array(0.1, xp=xp)),
            63.095734448019336,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test ITU-R BT.2100-1 method
        xp_assert_close(
            ootf_BT2100_HLG(xp_as_array(0.1, xp=xp), 0.01, method="ITU-R BT.2100-1"),
            63.105103490674857,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test with different L_W value
        xp_assert_close(
            ootf_BT2100_HLG(xp_as_array(0.1, xp=xp), L_W=2000),
            94.3186112317,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestOotfInverseBT2100HLG:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG` definition unit tests methods.
    """

    def test_ootf_inverse_BT2100_HLG(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG` definition.
        """

        # Test default method (ITU-R BT.2100-2)
        xp_assert_close(
            ootf_inverse_BT2100_HLG(xp_as_array(63.095734448019336, xp=xp)),
            0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test ITU-R BT.2100-1 method
        xp_assert_close(
            ootf_inverse_BT2100_HLG(
                xp_as_array(63.105103490674857, xp=xp), 0.01, method="ITU-R BT.2100-1"
            ),
            0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test with different L_W value
        xp_assert_close(
            ootf_inverse_BT2100_HLG(xp_as_array(94.3186112317, xp=xp), L_W=2000),
            0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


# ---------------------------------------------------------------------------
# Independent *Recommendation ITU-R BT.2100* *Reference HLG* photometric
# reference implementation.
#
# The helpers below restate the OETF / OOTF / EOTF photometric relationships
# published in ITU-R BT.2100-1 and ITU-R BT.2100-2 using only :mod:`numpy`,
# i.e. without calling any of the definitions under test. Expected display
# luminances and recovered scene light in the following test classes are
# derived from these equations so that the tests judge the implementation
# against the standard rather than against itself.
#
# References
# ----------
# :cite:`InternationalTelecommunicationUnion2017`,
# :cite:`InternationalTelecommunicationUnion2018`
# ---------------------------------------------------------------------------

_BT2100_HLG_A: float = 0.17883277
_BT2100_HLG_B: float = 1.0 - 4.0 * _BT2100_HLG_A
_BT2100_HLG_C: float = 0.5 - _BT2100_HLG_A * np.log(4.0 * _BT2100_HLG_A)
_BT2100_HLG_WEIGHTS: np.ndarray = np.array([0.2627, 0.6780, 0.0593])


def _reference_gamma_BT2100_HLG(L_W: float) -> float:
    """Return the BT.2100 system gamma for nominal peak ``L_W``."""

    return 1.2 + 0.42 * np.log10(L_W / 1000.0)


def _reference_oetf_BT2100_HLG(E: np.ndarray) -> np.ndarray:
    """Apply independently the BT.2100 HLG OETF to relative scene light."""

    E = np.asarray(E, dtype=np.float64)
    # The BT.2100 HLG OETF encodes 12 times the relative scene light through
    # the ARIB STD-B67 relationship, expressed analytically with a 1/2
    # normalisation factor on the lower branch.
    scene = 12.0 * E

    return np.where(
        scene <= 1.0,
        0.5 * np.sqrt(scene),
        _BT2100_HLG_A * np.log(np.maximum(scene - _BT2100_HLG_B, 1e-12))
        + _BT2100_HLG_C,
    )


def _reference_oetf_inverse_BT2100_HLG(E_p: np.ndarray) -> np.ndarray:
    """Apply independently the BT.2100 HLG inverse OETF."""

    E_p = np.asarray(E_p, dtype=np.float64)

    return np.where(
        E_p <= 0.5,
        E_p**2 / 3.0,
        (np.exp((E_p - _BT2100_HLG_C) / _BT2100_HLG_A) + _BT2100_HLG_B) / 12.0,
    )


def _reference_black_level_lift_BT2100_HLG(
    L_B: float, L_W: float, gamma: float
) -> float:
    """Compute independently the BT.2100-2 black level lift beta."""

    return np.sqrt(3.0 * (L_B / L_W) ** (1.0 / gamma))


def _reference_ootf_BT2100_HLG_1(
    E: np.ndarray, L_B: float, L_W: float, gamma: float
) -> np.ndarray:
    """Apply independently the BT.2100-1 HLG OOTF (scene to display light)."""

    E = np.asarray(E, dtype=np.float64)
    Y_S = np.sum(_BT2100_HLG_WEIGHTS * E, axis=-1, keepdims=True)
    luminance_power = np.where(Y_S != 0.0, np.abs(Y_S) ** (gamma - 1.0), 0.0)

    return (L_W - L_B) * E * luminance_power + L_B


def _reference_ootf_BT2100_HLG_2(E: np.ndarray, L_W: float, gamma: float) -> np.ndarray:
    """Apply independently the BT.2100-2 HLG OOTF (scene to display light)."""

    E = np.asarray(E, dtype=np.float64)
    Y_S = np.sum(_BT2100_HLG_WEIGHTS * E, axis=-1, keepdims=True)
    luminance_power = np.where(Y_S != 0.0, np.abs(Y_S) ** (gamma - 1.0), 0.0)

    return L_W * E * luminance_power


def _reference_ootf_inverse_BT2100_HLG_1(
    F_D: np.ndarray, L_B: float, L_W: float, gamma: float
) -> np.ndarray:
    """Apply independently the BT.2100-1 inverse HLG OOTF."""

    F_D = np.asarray(F_D, dtype=np.float64)
    Y_D = np.sum(_BT2100_HLG_WEIGHTS * F_D, axis=-1, keepdims=True)
    alpha = L_W - L_B
    factor = np.abs((Y_D - L_B) / alpha) ** ((1.0 - gamma) / gamma)

    return np.where(Y_D == L_B, 0.0, factor * (F_D - L_B) / alpha)


def _reference_ootf_inverse_BT2100_HLG_2(
    F_D: np.ndarray, L_W: float, gamma: float
) -> np.ndarray:
    """Apply independently the BT.2100-2 inverse HLG OOTF."""

    F_D = np.asarray(F_D, dtype=np.float64)
    Y_D = np.sum(_BT2100_HLG_WEIGHTS * F_D, axis=-1, keepdims=True)
    factor = np.abs(Y_D / L_W) ** ((1.0 - gamma) / gamma)

    return np.where(Y_D == 0.0, 0.0, factor * F_D / L_W)


def _reference_eotf_BT2100_HLG_1(
    E_p: np.ndarray, L_B: float, L_W: float, gamma: float
) -> np.ndarray:
    """Apply independently the BT.2100-1 HLG EOTF (signal to cd/m^2)."""

    return _reference_ootf_BT2100_HLG_1(
        _reference_oetf_inverse_BT2100_HLG(E_p), L_B, L_W, gamma
    )


def _reference_eotf_BT2100_HLG_2(
    E_p: np.ndarray, L_B: float, L_W: float, gamma: float
) -> np.ndarray:
    """Apply independently the BT.2100-2 HLG EOTF (signal to cd/m^2)."""

    beta = _reference_black_level_lift_BT2100_HLG(L_B, L_W, gamma)
    lifted = (1.0 - beta) * np.asarray(E_p, dtype=np.float64) + beta

    return _reference_ootf_BT2100_HLG_2(
        _reference_oetf_inverse_BT2100_HLG(lifted), L_W, gamma
    )


def _reference_eotf_inverse_BT2100_HLG_1(
    F_D: np.ndarray, L_B: float, L_W: float, gamma: float
) -> np.ndarray:
    """Apply independently the BT.2100-1 inverse HLG EOTF."""

    return _reference_oetf_BT2100_HLG(
        _reference_ootf_inverse_BT2100_HLG_1(F_D, L_B, L_W, gamma)
    )


def _reference_eotf_inverse_BT2100_HLG_2(
    F_D: np.ndarray, L_B: float, L_W: float, gamma: float
) -> np.ndarray:
    """Apply independently the BT.2100-2 inverse HLG EOTF."""

    beta = _reference_black_level_lift_BT2100_HLG(L_B, L_W, gamma)
    E_p = _reference_oetf_BT2100_HLG(
        _reference_ootf_inverse_BT2100_HLG_2(F_D, L_W, gamma)
    )

    return (E_p - beta) / (1.0 - beta)


# Coloured scene samples spanning both OETF branches and including the
# achromatic black and reference white anchors.
_BT2100_HLG_SCENE_SAMPLES: np.ndarray = np.array(
    [
        [0.00000000, 0.00000000, 0.00000000],
        [0.01499999, 0.03000000, 0.07500000],
        [0.01500000, 0.25000000, 0.50000000],
        [0.18000000, 0.42000000, 0.09000000],
        [0.64000000, 0.32000000, 0.88000000],
        [1.00000000, 1.00000000, 1.00000000],
    ]
)

# Display monitoring environments: (display black L_B, nominal peak L_W).
_BT2100_HLG_MONITORS: tuple[tuple[float, float], ...] = (
    (0.000, 1000),
    (0.010, 1000),
    (0.001, 2000),
    (0.100, 4000),
    (0.050, 10000),
)


class TestBlack_level_lift_BT2100_HLG:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
black_level_lift_BT2100_HLG` definition unit tests methods.
    """

    def test_black_level_lift_BT2100_HLG(self, xp: ModuleType) -> None:  # noqa: ARG002
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
black_level_lift_BT2100_HLG` definition against the independently evaluated
        BT.2100-2 black level lift equation.
        """

        for L_B, L_W in _BT2100_HLG_MONITORS:
            gamma = _reference_gamma_BT2100_HLG(L_W)
            beta = _reference_black_level_lift_BT2100_HLG(L_B, L_W, gamma)

            xp_assert_close(
                black_level_lift_BT2100_HLG(L_B, L_W, gamma),
                beta,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

        # A zero display black must produce a zero lift on any monitor.
        for L_W in (1000, 2000, 4000, 10000):
            xp_assert_close(
                black_level_lift_BT2100_HLG(0.0, L_W, _reference_gamma_BT2100_HLG(L_W)),
                0.0,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )


class TestBT2100HLGMonitorAbsoluteLuminance:
    """
    Define unit tests checking that a colourist can independently determine
    the absolute display luminance, in :math:`cd/m^2`, reached by every
    channel of a coloured image on monitors with a non-zero black level, for
    both versions of *Recommendation ITU-R BT.2100* *Reference HLG*.
    """

    def test_ootf_absolute_luminance_BT2100_1(self, xp: ModuleType) -> None:
        """
        Test the BT.2100-1 OOTF per-channel absolute luminance against the
        independently evaluated photometric equation, including monitors with
        a display luminance for black.
        """

        for L_B, L_W in _BT2100_HLG_MONITORS:
            gamma = _reference_gamma_BT2100_HLG(L_W)
            F_D = _reference_ootf_BT2100_HLG_1(
                _BT2100_HLG_SCENE_SAMPLES, L_B, L_W, gamma
            )

            xp_assert_close(
                ootf_BT2100_HLG_1(
                    xp_as_array(_BT2100_HLG_SCENE_SAMPLES, xp=xp),
                    L_B,
                    L_W,
                    gamma,
                ),
                F_D,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

            # The achromatic black scene must render at the display black
            # luminance on every channel.
            xp_assert_close(
                ootf_BT2100_HLG_1(
                    xp_as_array(np.zeros((1, 3)), xp=xp), L_B, L_W, gamma
                ),
                np.full((1, 3), L_B),
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

    def test_ootf_absolute_luminance_BT2100_2(self, xp: ModuleType) -> None:
        """
        Test the BT.2100-2 OOTF per-channel absolute luminance against the
        independently evaluated photometric equation.
        """

        for _L_B, L_W in _BT2100_HLG_MONITORS:
            gamma = _reference_gamma_BT2100_HLG(L_W)
            F_D = _reference_ootf_BT2100_HLG_2(_BT2100_HLG_SCENE_SAMPLES, L_W, gamma)

            xp_assert_close(
                ootf_BT2100_HLG_2(
                    xp_as_array(_BT2100_HLG_SCENE_SAMPLES, xp=xp), L_W, gamma
                ),
                F_D,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

            # Scene reference white must reach the nominal peak luminance on
            # every channel, independently of the monitor black modelling.
            xp_assert_close(
                ootf_BT2100_HLG_2(xp_as_array(np.ones((1, 3)), xp=xp), L_W, gamma),
                np.full((1, 3), L_W),
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

    def test_eotf_absolute_luminance_with_black_level(self, xp: ModuleType) -> None:
        """
        Test the BT.2100-1 and BT.2100-2 EOTF absolute display luminance for
        non-linear coloured HLG signals on monitors with a black level,
        including the BT.2100-2 black level lift applied to the signal.
        """

        E_p = _reference_oetf_BT2100_HLG(_BT2100_HLG_SCENE_SAMPLES)

        for L_B, L_W in _BT2100_HLG_MONITORS:
            gamma = _reference_gamma_BT2100_HLG(L_W)

            F_D_1 = _reference_eotf_BT2100_HLG_1(E_p, L_B, L_W, gamma)
            xp_assert_close(
                eotf_BT2100_HLG_1(xp_as_array(E_p, xp=xp), L_B, L_W, gamma),
                F_D_1,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

            F_D_2 = _reference_eotf_BT2100_HLG_2(E_p, L_B, L_W, gamma)
            xp_assert_close(
                eotf_BT2100_HLG_2(xp_as_array(E_p, xp=xp), L_B, L_W, gamma),
                F_D_2,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

            # The black signal must render at the display black luminance on
            # every channel with the BT.2100-2 lifted black behaviour.
            if L_B > 0.0:
                xp_assert_close(
                    eotf_BT2100_HLG_2(
                        xp_as_array(np.zeros((1, 3)), xp=xp),
                        L_B,
                        L_W,
                        gamma,
                    ),
                    np.full((1, 3), L_B),
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    def test_eotf_dispatcher_absolute_luminance(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG` definition dispatch against the independently evaluated
        BT.2100-1 and BT.2100-2 photometric equations on a black-level
        monitor.
        """

        L_B, L_W = 0.01, 1000
        gamma = _reference_gamma_BT2100_HLG(L_W)
        E_p = _reference_oetf_BT2100_HLG(_BT2100_HLG_SCENE_SAMPLES)

        xp_assert_close(
            eotf_BT2100_HLG(
                xp_as_array(E_p, xp=xp),
                L_B,
                L_W,
                gamma,
                method="ITU-R BT.2100-1",
            ),
            _reference_eotf_BT2100_HLG_1(E_p, L_B, L_W, gamma),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_BT2100_HLG(
                xp_as_array(E_p, xp=xp),
                L_B,
                L_W,
                gamma,
                method="ITU-R BT.2100-2",
            ),
            _reference_eotf_BT2100_HLG_2(E_p, L_B, L_W, gamma),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestBT2100HLGDisplayToSceneRecovery:
    """
    Define unit tests checking that the display light of a coloured image can
    be inverted back to the original scene light independently for monitors
    with different nominal peak luminances, for both versions of
    *Recommendation ITU-R BT.2100* *Reference HLG*.
    """

    def test_inverse_ootf_recovers_scene_light_BT2100_1(self, xp: ModuleType) -> None:
        """
        Test the BT.2100-1 inverse OOTF against the independently evaluated
        inverse photometric equation and the original coloured scene light.
        """

        for L_B, L_W in _BT2100_HLG_MONITORS:
            gamma = _reference_gamma_BT2100_HLG(L_W)
            F_D = _reference_ootf_BT2100_HLG_1(
                _BT2100_HLG_SCENE_SAMPLES, L_B, L_W, gamma
            )
            E = _reference_ootf_inverse_BT2100_HLG_1(F_D, L_B, L_W, gamma)

            xp_assert_close(
                ootf_inverse_BT2100_HLG_1(xp_as_array(F_D, xp=xp), L_B, L_W, gamma),
                E,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )
            xp_assert_close(
                E,
                _BT2100_HLG_SCENE_SAMPLES,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

    def test_inverse_ootf_recovers_scene_light_BT2100_2(self, xp: ModuleType) -> None:
        """
        Test the BT.2100-2 inverse OOTF against the independently evaluated
        inverse photometric equation and the original coloured scene light.
        """

        for _L_B, L_W in _BT2100_HLG_MONITORS:
            gamma = _reference_gamma_BT2100_HLG(L_W)
            F_D = _reference_ootf_BT2100_HLG_2(_BT2100_HLG_SCENE_SAMPLES, L_W, gamma)
            E = _reference_ootf_inverse_BT2100_HLG_2(F_D, L_W, gamma)

            xp_assert_close(
                ootf_inverse_BT2100_HLG_2(xp_as_array(F_D, xp=xp), L_W, gamma),
                E,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )
            xp_assert_close(
                E,
                _BT2100_HLG_SCENE_SAMPLES,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

    def test_scene_recovery_invariant_across_monitor_peaks(
        self, xp: ModuleType
    ) -> None:
        """
        Test that the same HLG programme rendered on monitors with different
        nominal peak luminances is inverted back to the same scene light, so
        that display-environment luminance changes are distinguishable from
        camera exposure adjustments.
        """

        scene = xp_as_array(_BT2100_HLG_SCENE_SAMPLES, xp=xp)

        recovered_1 = []
        for L_B, L_W in ((0.01, 1000), (0.01, 2000), (0.01, 4000)):
            gamma = _reference_gamma_BT2100_HLG(L_W)
            F_D = ootf_BT2100_HLG_1(scene, L_B, L_W, gamma)
            recovered_1.append(ootf_inverse_BT2100_HLG_1(F_D, L_B, L_W, gamma))

        xp_assert_close(recovered_1[0], recovered_1[1], atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(recovered_1[1], recovered_1[2], atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(recovered_1[0], scene, atol=TOLERANCE_ABSOLUTE_TESTS)

        recovered_2 = []
        for L_W in (1000, 2000, 4000):
            gamma = _reference_gamma_BT2100_HLG(L_W)
            F_D = ootf_BT2100_HLG_2(scene, L_W, gamma)
            recovered_2.append(ootf_inverse_BT2100_HLG_2(F_D, L_W, gamma))

        xp_assert_close(recovered_2[0], recovered_2[1], atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(recovered_2[1], recovered_2[2], atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(recovered_2[0], scene, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_inverse_eotf_dispatcher_recovers_signal(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG` definition dispatch against the independently
        evaluated inverse BT.2100-1 and BT.2100-2 photometric equations on a
        black-level monitor.
        """

        L_B, L_W = 0.01, 1000
        gamma = _reference_gamma_BT2100_HLG(L_W)
        E_p = _reference_oetf_BT2100_HLG(_BT2100_HLG_SCENE_SAMPLES)
        F_D_1 = _reference_eotf_BT2100_HLG_1(E_p, L_B, L_W, gamma)
        F_D_2 = _reference_eotf_BT2100_HLG_2(E_p, L_B, L_W, gamma)

        xp_assert_close(
            eotf_inverse_BT2100_HLG(
                xp_as_array(F_D_1, xp=xp),
                L_B,
                L_W,
                gamma,
                method="ITU-R BT.2100-1",
            ),
            _reference_eotf_inverse_BT2100_HLG_1(F_D_1, L_B, L_W, gamma),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT2100_HLG(
                xp_as_array(F_D_2, xp=xp),
                L_B,
                L_W,
                gamma,
                method="ITU-R BT.2100-2",
            ),
            _reference_eotf_inverse_BT2100_HLG_2(F_D_2, L_B, L_W, gamma),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestBT2100HLGMonitoringChainRoundTrip:
    """
    Define unit tests for the complete *Recommendation ITU-R BT.2100*
    *Reference HLG* monitoring chain connecting non-linear signals, absolute
    display light and scene light on coloured imagery.
    """

    def test_signal_display_signal_round_trip(self, xp: ModuleType) -> None:
        """
        Test that the EOTF followed by the inverse EOTF restores the
        non-linear coloured HLG signal on black-level monitors for both
        BT.2100 versions, with expected values derived independently.
        """

        E_p = _reference_oetf_BT2100_HLG(_BT2100_HLG_SCENE_SAMPLES)

        for L_B, L_W in _BT2100_HLG_MONITORS:
            gamma = _reference_gamma_BT2100_HLG(L_W)

            F_D_1 = _reference_eotf_BT2100_HLG_1(E_p, L_B, L_W, gamma)
            xp_assert_close(
                eotf_inverse_BT2100_HLG_1(xp_as_array(F_D_1, xp=xp), L_B, L_W, gamma),
                E_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

            F_D_2 = _reference_eotf_BT2100_HLG_2(E_p, L_B, L_W, gamma)
            xp_assert_close(
                eotf_inverse_BT2100_HLG_2(xp_as_array(F_D_2, xp=xp), L_B, L_W, gamma),
                E_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

    def test_scene_display_scene_round_trip(self, xp: ModuleType) -> None:
        """
        Test that display light produced from a coloured scene is inverted
        back to the identical scene light for both BT.2100 versions and
        across monitor peak luminances.
        """

        scene = _BT2100_HLG_SCENE_SAMPLES

        for L_B, L_W in _BT2100_HLG_MONITORS:
            gamma = _reference_gamma_BT2100_HLG(L_W)

            F_D_1 = _reference_ootf_BT2100_HLG_1(scene, L_B, L_W, gamma)
            xp_assert_close(
                ootf_inverse_BT2100_HLG_1(xp_as_array(F_D_1, xp=xp), L_B, L_W, gamma),
                scene,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

            F_D_2 = _reference_ootf_BT2100_HLG_2(scene, L_W, gamma)
            xp_assert_close(
                ootf_inverse_BT2100_HLG_2(xp_as_array(F_D_2, xp=xp), L_W, gamma),
                scene,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

    def test_full_chain_recovers_scene_light(self, xp: ModuleType) -> None:
        """
        Test the complete monitoring chain for a coloured image: non-linear
        HLG signal to absolute display light, then back through the inverse
        EOTF and inverse OETF to the original relative scene light.
        """

        scene = _BT2100_HLG_SCENE_SAMPLES
        E_p = _reference_oetf_BT2100_HLG(scene)

        for L_B, L_W in ((0.01, 1000), (0.1, 4000)):
            gamma = _reference_gamma_BT2100_HLG(L_W)

            for eotf, eotf_inverse in (
                (
                    eotf_BT2100_HLG_1,
                    eotf_inverse_BT2100_HLG_1,
                ),
                (
                    eotf_BT2100_HLG_2,
                    eotf_inverse_BT2100_HLG_2,
                ),
            ):
                F_D = eotf(xp_as_array(E_p, xp=xp), L_B, L_W, gamma)
                E_p_recovered = eotf_inverse(F_D, L_B, L_W, gamma)
                scene_recovered = oetf_inverse_BT2100_HLG(E_p_recovered)

                xp_assert_close(
                    E_p_recovered,
                    E_p,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
                xp_assert_close(
                    scene_recovered,
                    scene,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
