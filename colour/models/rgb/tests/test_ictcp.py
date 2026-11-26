"""Define the unit tests for the :mod:`colour.models.rgb.ictcp` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb import ICtCp_to_RGB, ICtCp_to_XYZ, RGB_to_ICtCp, XYZ_to_ICtCp
from colour.models.rgb.ictcp import (
    MATRIX_ICTCP_LMS_P_TO_ICTCP,
    MATRIX_ICTCP_LMS_P_TO_ICTCP_BT2100_HLG_2,
)
from colour.models.rgb.transfer_functions import (
    oetf_BT2100_HLG,
    oetf_inverse_BT2100_HLG,
)
from colour.utilities import (
    as_ndarray,
    domain_range_scale,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestRGB_to_ICtCp",
    "TestICtCp_to_RGB",
    "TestXYZ_to_ICtCp",
    "TestICtCp_to_XYZ",
    "TestBT2100DeliverySemantics",
]

_BT2100_PQ_METHOD = "ITU-R BT.2100-2 PQ"
_BT2100_HLG_V1_METHOD = "ITU-R BT.2100-1 HLG"
_BT2100_HLG_V2_METHOD = "ITU-R BT.2100-2 HLG"

_DELIVERY_METHODS = (
    _BT2100_PQ_METHOD,
    _BT2100_HLG_V1_METHOD,
    _BT2100_HLG_V2_METHOD,
)

# A saturated BT.2020 highlight in linear scene/display light.  The values are
# exposure coordinates, not non-linear R'G'B' code values.
_COLOURED_HIGHLIGHT = [1.0, 0.25, 0.05]

# Standard component oracles for that highlight.  Round-tripping these vectors
# is useful but cannot establish that the selected BT.2100 variant was used.
_COLOURED_HIGHLIGHT_REFERENCES = (
    pytest.param(
        _BT2100_PQ_METHOD,
        [0.11223650, -0.04619242, 0.08518316],
        id="bt2100-2-pq",
    ),
    pytest.param(
        _BT2100_HLG_V1_METHOD,
        [0.84679821, -0.36508506, 0.40147240],
        id="bt2100-1-hlg",
    ),
    pytest.param(
        _BT2100_HLG_V2_METHOD,
        [0.84679821, -0.20017940, 0.21270449],
        id="bt2100-2-hlg",
    ),
)

_BT2020_PRIMARY_DIRECTIONS = (
    pytest.param([1.0, 0.0, 0.0], id="bt2020-red"),
    pytest.param([0.0, 1.0, 0.0], id="bt2020-green"),
    pytest.param([0.0, 0.0, 1.0], id="bt2020-blue"),
)


class TestRGB_to_ICtCp:
    """
    Define :func:`colour.models.rgb.ictcp.TestRGB_to_ICtCp` definition unit
    tests methods.
    """

    def test_RGB_to_ICtCp(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.ictcp.RGB_to_ICtCp` definition."""

        xp_assert_close(
            RGB_to_ICtCp(xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp)),
            [0.07351364, 0.00475253, 0.09351596],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_ICtCp(
                xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp), L_p=4000
            ),
            [0.10516931, 0.00514031, 0.12318730],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_ICtCp(
                xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp), L_p=1000
            ),
            [0.17079612, 0.00485580, 0.17431356],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_ICtCp(
                xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp),
                method="ITU-R BT.2100-1 PQ",
            ),
            [0.07351364, 0.00475253, 0.09351596],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_ICtCp(
                xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp),
                method="ITU-R BT.2100-2 PQ",
            ),
            [0.07351364, 0.00475253, 0.09351596],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_ICtCp(
                xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp),
                method="ITU-R BT.2100-1 HLG",
            ),
            [0.62567899, -0.03622422, 0.67786522],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_ICtCp(
                xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp),
                method="ITU-R BT.2100-2 HLG",
            ),
            [0.62567899, -0.01984490, 0.35911259],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_RGB_to_ICtCp(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.RGB_to_ICtCp` definition
        n-dimensional support.
        """

        RGB = xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp)
        ICtCp = as_ndarray(RGB_to_ICtCp(RGB))

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (6, 1))
        ICtCp = xp.tile(xp_as_array(ICtCp, xp=xp), (6, 1))
        xp_assert_close(RGB_to_ICtCp(RGB), ICtCp, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (2, 3, 3), xp=xp)
        ICtCp = xp_reshape(xp_as_array(ICtCp, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(RGB_to_ICtCp(RGB), ICtCp, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_RGB_to_ICtCp(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.RGB_to_ICtCp` definition domain
        and range scale support.
        """

        RGB = xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp)
        ICtCp = as_ndarray(RGB_to_ICtCp(RGB))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    RGB_to_ICtCp(RGB * factor),
                    ICtCp * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_RGB_to_ICtCp(self) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.RGB_to_ICtCp` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_ICtCp(cases)


class TestICtCp_to_RGB:
    """
    Define :func:`colour.models.rgb.ictcp.ICtCp_to_RGB` definition unit tests
    methods.
    """

    def test_ICtCp_to_RGB(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.ictcp.ICtCp_to_RGB` definition."""

        xp_assert_close(
            ICtCp_to_RGB(xp_as_array([0.07351364, 0.00475253, 0.09351596], xp=xp)),
            [0.45620519, 0.03081071, 0.04091952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_RGB(
                xp_as_array([0.10516931, 0.00514031, 0.12318730], xp=xp), L_p=4000
            ),
            [0.45620519, 0.03081071, 0.04091952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_RGB(
                xp_as_array([0.17079612, 0.00485580, 0.17431356], xp=xp), L_p=1000
            ),
            [0.45620519, 0.03081071, 0.04091952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_RGB(
                xp_as_array([0.07351364, 0.00475253, 0.09351596], xp=xp),
                method="ITU-R BT.2100-1 PQ",
            ),
            [0.45620519, 0.03081071, 0.04091952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_RGB(
                xp_as_array([0.07351364, 0.00475253, 0.09351596], xp=xp),
                method="ITU-R BT.2100-2 PQ",
            ),
            [0.45620519, 0.03081071, 0.04091952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_RGB(
                xp_as_array([0.62567899, -0.03622422, 0.67786522], xp=xp),
                method="ITU-R BT.2100-1 HLG",
            ),
            [0.45620519, 0.03081071, 0.04091952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_RGB(
                xp_as_array([0.62567899, -0.01984490, 0.35911259], xp=xp),
                method="ITU-R BT.2100-2 HLG",
            ),
            [0.45620519, 0.03081071, 0.04091952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_ICtCp_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.ICtCp_to_RGB` definition
        n-dimensional support.
        """

        ICtCp = xp_as_array([0.07351364, 0.00475253, 0.09351596], xp=xp)
        RGB = as_ndarray(ICtCp_to_RGB(ICtCp))

        ICtCp = xp.tile(xp_as_array(ICtCp, xp=xp), (6, 1))
        RGB = xp.tile(xp_as_array(RGB, xp=xp), (6, 1))
        xp_assert_close(ICtCp_to_RGB(ICtCp), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

        ICtCp = xp_reshape(xp_as_array(ICtCp, xp=xp), (2, 3, 3), xp=xp)
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(ICtCp_to_RGB(ICtCp), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_ICtCp_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.ICtCp_to_RGB` definition domain
        and range scale support.
        """

        ICtCp = xp_as_array([0.07351364, 0.00475253, 0.09351596], xp=xp)
        RGB = as_ndarray(ICtCp_to_RGB(ICtCp))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    ICtCp_to_RGB(ICtCp * factor),
                    RGB * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_ICtCp_to_RGB(self) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.ICtCp_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        ICtCp_to_RGB(cases)


class TestXYZ_to_ICtCp:
    """
    Define :func:`colour.models.rgb.ictcp.TestXYZ_to_ICtCp` definition unit
    tests methods.
    """

    def test_XYZ_to_ICtCp(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.ictcp.XYZ_to_ICtCp` definition."""

        xp_assert_close(
            XYZ_to_ICtCp(xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)),
            [0.06858097, -0.00283842, 0.06020983],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_ICtCp(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                [0.34570, 0.35850],
            ),
            [0.06792437, 0.00452089, 0.05514480],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_ICtCp(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                [0.34570, 0.35850],
                chromatic_adaptation_transform="Bradford",
            ),
            [0.06783951, 0.00476111, 0.05523093],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_ICtCp(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp), L_p=4000
            ),
            [0.09871102, -0.00447247, 0.07984812],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_ICtCp(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp), L_p=1000
            ),
            [0.16173872, -0.00792543, 0.11409458],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_ICtCp(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                method="ITU-R BT.2100-1 PQ",
            ),
            [0.06858097, -0.00283842, 0.06020983],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_ICtCp(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                method="ITU-R BT.2100-2 PQ",
            ),
            [0.06858097, -0.00283842, 0.06020983],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_ICtCp(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                method="ITU-R BT.2100-1 HLG",
            ),
            [0.59242792, -0.06824263, 0.47421473],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_ICtCp(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                method="ITU-R BT.2100-2 HLG",
            ),
            [0.59242792, -0.03740730, 0.25122675],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_ICtCp(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.XYZ_to_ICtCp` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        ICtCp = as_ndarray(XYZ_to_ICtCp(XYZ))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        ICtCp = xp.tile(xp_as_array(ICtCp, xp=xp), (6, 1))
        xp_assert_close(XYZ_to_ICtCp(XYZ), ICtCp, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        ICtCp = xp_reshape(xp_as_array(ICtCp, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(XYZ_to_ICtCp(XYZ), ICtCp, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_XYZ_to_ICtCp(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.XYZ_to_ICtCp` definition domain
        and range scale support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        ICtCp = as_ndarray(XYZ_to_ICtCp(XYZ))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_ICtCp(XYZ * factor),
                    ICtCp * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_ICtCp(self) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.XYZ_to_ICtCp` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_ICtCp(cases)


class TestICtCp_to_XYZ:
    """
    Define :func:`colour.models.rgb.ictcp.ICtCp_to_XYZ` definition unit tests
    methods.
    """

    def test_ICtCp_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.ictcp.ICtCp_to_XYZ` definition."""

        xp_assert_close(
            ICtCp_to_XYZ(xp_as_array([0.06858097, -0.00283842, 0.06020983], xp=xp)),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_XYZ(
                xp_as_array([0.06792437, 0.00452089, 0.05514480], xp=xp),
                [0.34570, 0.35850],
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_XYZ(
                xp_as_array([0.06783951, 0.00476111, 0.05523093], xp=xp),
                [0.34570, 0.35850],
                chromatic_adaptation_transform="Bradford",
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_XYZ(
                xp_as_array([0.09871102, -0.00447247, 0.07984812], xp=xp), L_p=4000
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_XYZ(
                xp_as_array([0.16173872, -0.00792543, 0.11409458], xp=xp), L_p=1000
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_XYZ(
                xp_as_array([0.06858097, -0.00283842, 0.06020983], xp=xp),
                method="ITU-R BT.2100-1 PQ",
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_XYZ(
                xp_as_array([0.06858097, -0.00283842, 0.06020983], xp=xp),
                method="ITU-R BT.2100-2 PQ",
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_XYZ(
                xp_as_array([0.59242792, -0.06824263, 0.47421473], xp=xp),
                method="ITU-R BT.2100-1 HLG",
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ICtCp_to_XYZ(
                xp_as_array([0.59242792, -0.03740730, 0.25122675], xp=xp),
                method="ITU-R BT.2100-2 HLG",
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_ICtCp_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.ICtCp_to_XYZ` definition
        n-dimensional support.
        """

        ICtCp = xp_as_array([0.06858097, -0.00283842, 0.06020983], xp=xp)
        XYZ = as_ndarray(ICtCp_to_XYZ(ICtCp))

        ICtCp = xp.tile(xp_as_array(ICtCp, xp=xp), (6, 1))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(ICtCp_to_XYZ(ICtCp), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        ICtCp = xp_reshape(xp_as_array(ICtCp, xp=xp), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(ICtCp_to_XYZ(ICtCp), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_ICtCp_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.ICtCp_to_XYZ` definition domain
        and range scale support.
        """

        ICtCp = xp_as_array([0.06858097, -0.00283842, 0.06020983], xp=xp)
        XYZ = as_ndarray(ICtCp_to_XYZ(ICtCp))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    ICtCp_to_XYZ(ICtCp * factor),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_ICtCp_to_XYZ(self) -> None:
        """
        Test :func:`colour.models.rgb.ictcp.ICtCp_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        ICtCp_to_XYZ(cases)


class TestBT2100DeliverySemantics:
    """
    Define standard-delivery and HDR master-exposure conformance tests for
    :mod:`colour.models.rgb.ictcp`.
    """

    @pytest.mark.parametrize(
        ("method", "components"),
        _COLOURED_HIGHLIGHT_REFERENCES,
    )
    def test_highlight_components_match_selected_bt2100_variant(
        self,
        xp: ModuleType,
        method: str,
        components: list[float],
    ) -> None:
        """
        Test that the intensity and two chroma components use the matrices
        prescribed by the selected *BT.2100* variant.
        """

        master = xp_as_array(_COLOURED_HIGHLIGHT, xp=xp)
        delivered = RGB_to_ICtCp(master, method=method)

        xp_assert_close(
            delivered,
            xp_as_array(components, xp=xp),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            ICtCp_to_RGB(delivered, method=method),
            master,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_hlg_component_matrices_match_standard_coefficients(self) -> None:
        """
        Test the version-specific *HLG* intensity and chroma matrix rows.
        """

        xp_assert_close(
            MATRIX_ICTCP_LMS_P_TO_ICTCP,
            np.array(
                [
                    [2048, 2048, 0],
                    [6610, -13613, 7003],
                    [17933, -17390, -543],
                ],
                dtype=float,
            )
            / 4096,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            MATRIX_ICTCP_LMS_P_TO_ICTCP_BT2100_HLG_2,
            np.array(
                [
                    [2048, 2048, 0],
                    [3625, -7465, 3840],
                    [9500, -9212, -288],
                ],
                dtype=float,
            )
            / 4096,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @pytest.mark.parametrize("method", _DELIVERY_METHODS)
    @pytest.mark.parametrize("primary", _BT2020_PRIMARY_DIRECTIONS)
    @pytest.mark.parametrize("exposure", [0.2, 1.0, 1.8])
    def test_wcg_primary_highlights_restore_master_exposure(
        self,
        xp: ModuleType,
        method: str,
        primary: list[float],
        exposure: float,
    ) -> None:
        """
        Test that each *BT.2020* primary direction remains the same linear
        master colour when recovered from its delivered components.
        """

        master = xp_as_array(exposure * np.asarray(primary, dtype=float), xp=xp)

        delivered = RGB_to_ICtCp(master, method=method)
        restored = ICtCp_to_RGB(delivered, method=method)

        xp_assert_close(restored, master, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(
            as_ndarray(restored).max(),
            exposure,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @pytest.mark.parametrize("method", _DELIVERY_METHODS)
    @pytest.mark.parametrize("primary", _BT2020_PRIMARY_DIRECTIONS)
    def test_linear_exposure_scaling_and_signal_component_scaling_differ(
        self,
        xp: ModuleType,
        method: str,
        primary: list[float],
    ) -> None:
        """
        Test that doubling linear exposure doubles recovered master light but
        does not double the non-linear delivered components.
        """

        half_exposure = xp_as_array(0.45 * np.asarray(primary, dtype=float), xp=xp)
        full_exposure = xp_as_array(0.90 * np.asarray(primary, dtype=float), xp=xp)

        half_delivered = RGB_to_ICtCp(half_exposure, method=method)
        full_delivered = RGB_to_ICtCp(full_exposure, method=method)

        half_restored = ICtCp_to_RGB(half_delivered, method=method)
        full_restored = ICtCp_to_RGB(full_delivered, method=method)

        xp_assert_close(
            full_restored,
            2 * half_restored,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        component_nonlinearity = as_ndarray(full_delivered) - 2 * as_ndarray(
            half_delivered
        )
        assert np.linalg.norm(component_nonlinearity) > 1.0e-2

    def test_bt2100_1_hlg_roundtrip_is_not_bt2100_2_conformant(
        self,
        xp: ModuleType,
    ) -> None:
        """
        Test that a self-consistent *BT.2100-1 HLG* codec is rejected by the
        *BT.2100-2 HLG* component oracle and recovers the wrong master when
        decoded as the latter.
        """

        master = xp_as_array(_COLOURED_HIGHLIGHT, xp=xp)
        expected_bt2100_2 = [0.84679821, -0.20017940, 0.21270449]

        bt2100_1 = RGB_to_ICtCp(master, method=_BT2100_HLG_V1_METHOD)
        bt2100_2 = RGB_to_ICtCp(master, method=_BT2100_HLG_V2_METHOD)

        # The old and new HLG matrices share the intensity row but prescribe
        # different C_T and C_P coefficients.
        xp_assert_close(bt2100_2[0], bt2100_1[0], atol=TOLERANCE_ABSOLUTE_TESTS)
        assert not np.allclose(
            as_ndarray(bt2100_2)[1:],
            as_ndarray(bt2100_1)[1:],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # BT.2100-1 is internally round-trip consistent, so a round-trip-only
        # test cannot distinguish it from the required BT.2100-2 delivery.
        xp_assert_close(
            ICtCp_to_RGB(bt2100_1, method=_BT2100_HLG_V1_METHOD),
            master,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        with pytest.raises(AssertionError):
            xp_assert_close(
                bt2100_1,
                xp_as_array(expected_bt2100_2, xp=xp),
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

        xp_assert_close(
            bt2100_2,
            xp_as_array(expected_bt2100_2, xp=xp),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        version_mismatched_master = ICtCp_to_RGB(
            bt2100_2,
            method=_BT2100_HLG_V1_METHOD,
        )
        assert (
            np.max(np.abs(as_ndarray(version_mismatched_master) - as_ndarray(master)))
            > 1.0e-1
        )

    @pytest.mark.parametrize("primary", _BT2020_PRIMARY_DIRECTIONS)
    def test_hlg_primary_codes_are_distinct_from_linear_light_values(
        self,
        xp: ModuleType,
        primary: list[float],
    ) -> None:
        """
        Test that the same *BT.2020* primary direction denotes proportional
        light at the master end but non-linear code values after HLG encoding.
        """

        linear_master = xp_as_array(
            1.8 * np.asarray(primary, dtype=float),
            xp=xp,
        )
        signal_codes = oetf_BT2100_HLG(linear_master)

        assert (
            np.linalg.norm(as_ndarray(signal_codes) - as_ndarray(linear_master))
            > 5.0e-1
        )

        decoded_linear = oetf_inverse_BT2100_HLG(signal_codes)
        xp_assert_close(
            decoded_linear,
            linear_master,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        standard_delivery = RGB_to_ICtCp(
            linear_master,
            method=_BT2100_HLG_V2_METHOD,
        )
        xp_assert_close(
            RGB_to_ICtCp(decoded_linear, method=_BT2100_HLG_V2_METHOD),
            standard_delivery,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Feeding the signal codes back in as linear light applies the
        # encoding a second time and does not produce the standard delivery.
        signal_codes_as_linear = RGB_to_ICtCp(
            signal_codes,
            method=_BT2100_HLG_V2_METHOD,
        )
        assert (
            np.max(
                np.abs(
                    as_ndarray(signal_codes_as_linear) - as_ndarray(standard_delivery)
                )
            )
            > 5.0e-2
        )
