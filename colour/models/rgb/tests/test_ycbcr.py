"""Define the unit tests for the :mod:`colour.models.rgb.ycbcr` module."""

from __future__ import annotations

import typing
from contextlib import nullcontext

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.ycbcr import (
    SCALES_YCBCR,
    WEIGHTS_YCBCR,
    RGB_to_YCbCr,
    RGB_to_YcCbcCrc,
    YCbCr_to_RGB,
    YcCbcCrc_to_RGB,
    matrix_YCbCr,
    offset_YCbCr,
    ranges_YCbCr,
    round_BT2100,
)
from colour.utilities import (
    array_namespace,
    as_ndarray,
    domain_range_scale,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_assert_equal,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Development"

__all__ = [
    "TestRoundBT2100",
    "TestRangeYCbCr",
    "TestMatrixYCbCr",
    "TestOffsetYCbCr",
    "TestRGB_to_YCbCr",
    "TestYCbCr_to_RGB",
    "TestBroadcastYCbCrRoundTrip",
    "TestRGB_to_YcCbcCrc",
    "TestYcCbcCrc_to_RGB",
]


class TestRoundBT2100:
    """
    Define :func:`colour.models.rgb.ycbcr.round_BT2100` definition unit tests
    methods.
    """

    def test_round_BT2100(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.ycbcr.round_BT2100` definition."""

        xp_assert_equal(
            round_BT2100(xp_as_array([-0.6, -0.5, -0.4, 0.4, 0.5, 0.6], xp=xp)),
            [-1.0, -1.0, -0.0, 0.0, 1.0, 1.0],
        )


class TestRangeYCbCr:
    """
    Define :func:`colour.models.rgb.ycbcr.ranges_YCbCr` definition unit tests
    methods.
    """

    def test_ranges_YCbCr(self) -> None:
        """Test :func:`colour.models.rgb.ycbcr.ranges_YCbCr` definition."""

        xp_assert_close(
            ranges_YCbCr(8, True, True),
            [16.00000000, 235.00000000, 16.00000000, 240.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ranges_YCbCr(8, True, False),
            [0.06274510, 0.92156863, 0.06274510, 0.94117647],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ranges_YCbCr(8, False, True),
            [0.00000000, 255.00000000, 0.50000000, 255.50000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ranges_YCbCr(8, False, False),
            [0.00000000, 1.00000000, -0.50000000, 0.50000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ranges_YCbCr(10, True, True),
            [64.00000000, 940.00000000, 64.00000000, 960.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ranges_YCbCr(10, True, False),
            [0.06256109, 0.91886608, 0.06256109, 0.93841642],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ranges_YCbCr(10, False, True),
            [0.00000000, 1023.00000000, 0.50000000, 1023.50000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ranges_YCbCr(10, False, False),
            [0.00000000, 1.00000000, -0.50000000, 0.50000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestMatrixYCbCr:
    """
    Define :func:`colour.models.rgb.ycbcr.matrix_YCbCr` definition unit tests
    methods.
    """

    def test_matrix_YCbCr(self, xp: ModuleType, request: pytest.FixtureRequest) -> None:
        """Test :func:`colour.models.rgb.ycbcr.matrix_YCbCr` definition."""

        xp_assert_close(
            matrix_YCbCr(K=xp_as_array(WEIGHTS_YCBCR["ITU-R BT.709"], xp=xp)),
            [
                [1.00000000, 0.00000000, 1.57480000],
                [1.00000000, -0.18732427, -0.46812427],
                [1.00000000, 1.85560000, 0.00000000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_YCbCr(K=xp_as_array(WEIGHTS_YCBCR["ITU-R BT.601"], xp=xp)),
            [
                [1.00000000, 0.00000000, 1.40200000],
                [1.00000000, -0.34413629, -0.71413629],
                [1.00000000, 1.77200000, -0.00000000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_YCbCr(
                K=xp_as_array(WEIGHTS_YCBCR["ITU-R BT.709"], xp=xp), is_legal=True
            ),
            [
                [1.16438356, 0.00000000, 1.79274107],
                [1.16438356, -0.21324861, -0.53290933],
                [1.16438356, 2.11240179, -0.00000000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_YCbCr(K=xp_as_array(WEIGHTS_YCBCR["ITU-R BT.709"], xp=xp), bits=10),
            [
                [1.00000000, 0.00000000, 1.57480000],
                [1.00000000, -0.18732427, -0.46812427],
                [1.00000000, 1.85560000, 0.00000000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_YCbCr(
                K=xp_as_array(WEIGHTS_YCBCR["ITU-R BT.709"], xp=xp),
                bits=10,
                is_int=True,
            ),
            [
                [0.00097752, 0.00000000, 0.00153939],
                [0.00097752, -0.00018311, -0.00045760],
                [0.00097752, 0.00181388, 0.00000000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_YCbCr(
                K=xp_as_array(WEIGHTS_YCBCR["ITU-R BT.709"], xp=xp),
                S=xp_as_array(SCALES_YCBCR["Y'UV"], xp=xp),
            ),
            [
                [1.00000000, 0.00000000, 1.28032520],
                [1.00000000, -0.21482141, -0.38058884],
                [1.00000000, 2.12798165, 0.00000000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # A native secondary operand must retain its backend, device and graph
        # while every other operand remains a host array.
        host_arguments: dict[str, typing.Any] = {"K": [0.2126, 0.0722], "S": [0.5, 0.5]}
        expected = matrix_YCbCr(**host_arguments)
        for parameter in ("S",):
            native = xp_as_array(host_arguments[parameter], xp=xp)
            arguments = {**host_arguments, parameter: native}
            actual = matrix_YCbCr(**arguments)
            assert array_namespace(actual) is array_namespace(native)
            assert getattr(actual, "device", None) == getattr(native, "device", None)
            xp_assert_close(actual, expected, atol=TOLERANCE_ABSOLUTE_TESTS)
            if xp.__name__ != "numpy":
                autodiff = request.getfixturevalue("autodiff")
                autodiff(
                    lambda value, parameter=parameter: matrix_YCbCr(
                        **{**host_arguments, parameter: value}
                    ),
                    native,
                )

    def test_matrix_YCbCr_autodiff(
        self, xp: ModuleType, autodiff: typing.Callable
    ) -> None:
        """Test numerical gradients with respect to luma coefficients."""

        autodiff(matrix_YCbCr, xp_as_array([0.2126, 0.0722], xp=xp))


class TestOffsetYCbCr:
    """
    Define :func:`colour.models.rgb.ycbcr.offset_YCbCr` definition unit tests
    methods.
    """

    def test_offset_YCbCr(self) -> None:
        """Test :func:`colour.models.rgb.ycbcr.offset_YCbCr` definition."""

        xp_assert_close(
            offset_YCbCr(),
            [0.00000000, 0.00000000, 0.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            offset_YCbCr(is_legal=True),
            [0.06274510, 0.50196078, 0.50196078],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            offset_YCbCr(bits=10),
            [0.00000000, 0.00000000, 0.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            offset_YCbCr(bits=10, is_int=True),
            [0.00000000, 512.00000000, 512.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestRGB_to_YCbCr:
    """
    Define :func:`colour.models.rgb.ycbcr.RGB_to_YCbCr` definition unit tests
    methods.
    """

    def test_RGB_to_YCbCr(self, xp: ModuleType, request: pytest.FixtureRequest) -> None:
        """Test :func:`colour.models.rgb.ycbcr.RGB_to_YCbCr` definition."""

        xp_assert_close(
            RGB_to_YCbCr(xp_as_array([0.75, 0.75, 0.0], xp=xp)),
            [0.66035745, 0.17254902, 0.53216593],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_YCbCr(
                xp_as_array([0.25, 0.5, 0.75], xp=xp),
                K=WEIGHTS_YCBCR["ITU-R BT.601"],
                out_int=True,
                out_legal=True,
                out_bits=10,
            ),
            [461, 662, 382],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_YCbCr(
                xp_as_array([0.0, 0.75, 0.75], xp=xp),
                K=WEIGHTS_YCBCR["ITU-R BT.2020"],
                out_int=False,
                out_legal=False,
            ),
            [0.55297500, 0.10472255, -0.37500000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_YCbCr(
                xp_as_array([0.75, 0.0, 0.75], xp=xp),
                K=WEIGHTS_YCBCR["ITU-R BT.709"],
                out_range=(16 / 255, 235 / 255, 15.5 / 255, 239.5 / 255),
            ),
            [0.24618980, 0.75392897, 0.79920662],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_YCbCr(
                xp_as_array([0.75, 0.5, 0.25], xp=xp),
                S=SCALES_YCBCR["Y'UV"],
                out_legal=False,
                out_int=False,
            ),
            [0.53510000, -0.13397672, 0.16784798],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        host_arguments: dict[str, typing.Any] = {
            "RGB": [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
            "K": [0.2126, 0.0722],
            "S": [0.5, 0.5],
            "in_range": [0.0, 1.0],
            "out_range": [0.0, 1.0, -0.5, 0.5],
        }
        expected = RGB_to_YCbCr(**host_arguments)
        for parameter, host_value in host_arguments.items():
            native = xp_as_array(host_value, xp=xp)
            # Keep the physical input device while factories default to CPU.
            with xp.device("cpu") if xp.__name__ == "torch" else nullcontext():
                actual = RGB_to_YCbCr(**{**host_arguments, parameter: native})
                assert array_namespace(actual) is array_namespace(native)
                assert getattr(actual, "device", None) == getattr(
                    native, "device", None
                )
                xp_assert_close(actual, expected, atol=TOLERANCE_ABSOLUTE_TESTS)
                if xp.__name__ != "numpy":
                    autodiff = request.getfixturevalue("autodiff")
                    autodiff(
                        lambda value, parameter=parameter: RGB_to_YCbCr(
                            **{**host_arguments, parameter: value}
                        ),
                        native,
                    )

    def test_n_dimensional_RGB_to_YCbCr(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ycbcr.RGB_to_YCbCr` definition
        n-dimensional arrays support.
        """

        RGB = xp_as_array([0.75, 0.5, 0.25], xp=xp)
        YCbCr = as_ndarray(RGB_to_YCbCr(RGB))

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 3), xp=xp)
        YCbCr = xp.tile(xp_as_array(YCbCr, xp=xp), (4,))
        YCbCr = xp_reshape(xp_as_array(YCbCr, xp=xp), (4, 3), xp=xp)
        xp_assert_close(
            RGB_to_YCbCr(RGB),
            YCbCr,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 3), xp=xp)
        YCbCr = xp.tile(xp_as_array(YCbCr, xp=xp), (4,))
        YCbCr = xp_reshape(xp_as_array(YCbCr, xp=xp), (4, 4, 3), xp=xp)
        xp_assert_close(
            RGB_to_YCbCr(RGB),
            YCbCr,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 4, 3), xp=xp)
        YCbCr = xp.tile(xp_as_array(YCbCr, xp=xp), (4,))
        YCbCr = xp_reshape(xp_as_array(YCbCr, xp=xp), (4, 4, 4, 3), xp=xp)
        xp_assert_close(
            RGB_to_YCbCr(RGB),
            YCbCr,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_RGB_to_YCbCr(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.prismatic.RGB_to_YCbCr` definition
        domain and range scale support.
        """

        RGB = xp_as_array([0.75, 0.5, 0.25], xp=xp)
        YCbCr = as_ndarray(RGB_to_YCbCr(RGB))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    RGB_to_YCbCr(RGB * factor),
                    YCbCr * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_RGB_to_YCbCr(self) -> None:
        """
        Test :func:`colour.models.rgb.ycbcr.RGB_to_YCbCr` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_YCbCr(cases)


class TestYCbCr_to_RGB:
    """
    Define :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition unit tests
    methods.
    """

    def test_YCbCr_to_RGB(self, xp: ModuleType, request: pytest.FixtureRequest) -> None:
        """Test :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition."""

        xp_assert_close(
            YCbCr_to_RGB(xp_as_array([0.66035745, 0.17254902, 0.53216593], xp=xp)),
            [0.75, 0.75, 0.0],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            YCbCr_to_RGB(
                xp_as_array([471, 650, 390], xp=xp),
                in_bits=10,
                in_legal=True,
                in_int=True,
            ),
            [0.25018598, 0.49950072, 0.75040741],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            YCbCr_to_RGB(
                xp_as_array([150, 99, 175], xp=xp),
                in_bits=8,
                in_legal=False,
                in_int=True,
                out_bits=8,
                out_legal=True,
                out_int=True,
            ),
            [208, 131, 99],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            YCbCr_to_RGB(
                xp_as_array([0.53510000, -0.13397672, 0.16784798], xp=xp),
                S=SCALES_YCBCR["Y'UV"],
                in_legal=False,
                in_int=False,
            ),
            [0.75, 0.5, 0.25],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        host_arguments: dict[str, typing.Any] = {
            "YCbCr": [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
            "K": [0.2126, 0.0722],
            "S": [0.5, 0.5],
            "in_range": [0.0, 1.0, -0.5, 0.5],
            "out_range": [0.0, 1.0],
        }
        expected = YCbCr_to_RGB(**host_arguments)
        for parameter, host_value in host_arguments.items():
            native = xp_as_array(host_value, xp=xp)
            # Keep the physical input device while factories default to CPU.
            with xp.device("cpu") if xp.__name__ == "torch" else nullcontext():
                actual = YCbCr_to_RGB(**{**host_arguments, parameter: native})
                assert array_namespace(actual) is array_namespace(native)
                assert getattr(actual, "device", None) == getattr(
                    native, "device", None
                )
                xp_assert_close(actual, expected, atol=TOLERANCE_ABSOLUTE_TESTS)
                if xp.__name__ != "numpy":
                    autodiff = request.getfixturevalue("autodiff")
                    autodiff(
                        lambda value, parameter=parameter: YCbCr_to_RGB(
                            **{**host_arguments, parameter: value}
                        ),
                        native,
                    )

    def test_roundtrip_YCbCr_YUV(self, xp: ModuleType) -> None:
        """Test *Y'UV* roundtrip with :func:`colour.models.rgb.ycbcr.RGB_to_YCbCr`
        and :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definitions.
        """

        RGB = xp_as_array([0.75, 0.5, 0.25], xp=xp)
        YUV = RGB_to_YCbCr(RGB, S=SCALES_YCBCR["Y'UV"], out_legal=False, out_int=False)
        xp_assert_close(
            YCbCr_to_RGB(YUV, S=SCALES_YCBCR["Y'UV"], in_legal=False, in_int=False),
            [0.75, 0.5, 0.25],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_YCbCr_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition
        n-dimensional arrays support.
        """

        YCbCr = xp_as_array([0.52230157, 0.36699593, 0.62183309], xp=xp)
        RGB = as_ndarray(YCbCr_to_RGB(YCbCr))

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 3), xp=xp)
        YCbCr = xp.tile(xp_as_array(YCbCr, xp=xp), (4,))
        YCbCr = xp_reshape(xp_as_array(YCbCr, xp=xp), (4, 3), xp=xp)
        xp_assert_close(
            YCbCr_to_RGB(YCbCr),
            RGB,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 3), xp=xp)
        YCbCr = xp.tile(xp_as_array(YCbCr, xp=xp), (4,))
        YCbCr = xp_reshape(xp_as_array(YCbCr, xp=xp), (4, 4, 3), xp=xp)
        xp_assert_close(
            YCbCr_to_RGB(YCbCr),
            RGB,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 4, 3), xp=xp)
        YCbCr = xp.tile(xp_as_array(YCbCr, xp=xp), (4,))
        YCbCr = xp_reshape(xp_as_array(YCbCr, xp=xp), (4, 4, 4, 3), xp=xp)
        xp_assert_close(
            YCbCr_to_RGB(YCbCr),
            RGB,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_YCbCr_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.prismatic.YCbCr_to_RGB` definition
        domain and range scale support.
        """

        YCbCr = xp_as_array([0.52230157, 0.36699593, 0.62183309], xp=xp)
        RGB = as_ndarray(YCbCr_to_RGB(YCbCr))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    YCbCr_to_RGB(YCbCr * factor),
                    RGB * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_YCbCr_to_RGB(self) -> None:
        """
        Test :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        YCbCr_to_RGB(cases)


BROADCAST_CODE_WORD_PARAMS = [
    pytest.param(
        8,
        np.array([0.0, 0.0, 0.0]),
        np.array([16, 128, 128]),
        id="8bit-black",
    ),
    pytest.param(
        8,
        np.array([0.5, 0.5, 0.5]),
        np.array([126, 128, 128]),
        id="8bit-half-gray",
    ),
    pytest.param(
        8,
        np.array([1.0, 1.0, 1.0]),
        np.array([235, 128, 128]),
        id="8bit-white",
    ),
    pytest.param(
        8,
        np.array([1.0, 0.0, 0.0]),
        np.array([63, 102, 240]),
        id="8bit-red",
    ),
    pytest.param(
        8,
        np.array([0.0, 1.0, 0.0]),
        np.array([173, 42, 26]),
        id="8bit-green",
    ),
    pytest.param(
        8,
        np.array([0.0, 0.0, 1.0]),
        np.array([32, 240, 118]),
        id="8bit-blue",
    ),
    pytest.param(
        8,
        np.array([0.75, 0.5, 0.25]),
        np.array([133, 94, 159]),
        id="8bit-interior-ramp",
    ),
    pytest.param(
        10,
        np.array([0.0, 0.0, 0.0]),
        np.array([64, 512, 512]),
        id="10bit-black",
    ),
    pytest.param(
        10,
        np.array([0.5, 0.5, 0.5]),
        np.array([502, 512, 512]),
        id="10bit-half-gray",
    ),
    pytest.param(
        10,
        np.array([1.0, 1.0, 1.0]),
        np.array([940, 512, 512]),
        id="10bit-white",
    ),
    pytest.param(
        10,
        np.array([1.0, 0.0, 0.0]),
        np.array([250, 409, 960]),
        id="10bit-red",
    ),
    pytest.param(
        10,
        np.array([0.0, 1.0, 0.0]),
        np.array([691, 167, 105]),
        id="10bit-green",
    ),
    pytest.param(
        10,
        np.array([0.0, 0.0, 1.0]),
        np.array([127, 960, 471]),
        id="10bit-blue",
    ),
    pytest.param(
        10,
        np.array([0.75, 0.5, 0.25]),
        np.array([533, 374, 634]),
        id="10bit-interior-ramp",
    ),
]


class TestBroadcastYCbCrRoundTrip:
    """
    Test the float *R'G'B'* to narrow-range *Y'CbCr* code word workflow and
    the editor decode back to float *R'G'B'*.
    """

    @staticmethod
    def _quantization_atol(bits: int) -> float:
        """
        Return a conservative *R'G'B'* error bound for one rounded code value.
        """

        scale = 2 ** (bits - 8)
        half_luma_level = 0.5 / (219 * scale)
        half_chroma_level = 0.5 / (224 * scale)

        kr, kb = (float(coefficient) for coefficient in WEIGHTS_YCBCR["ITU-R BT.709"])
        cb_scale, cr_scale = (
            float(coefficient) for coefficient in SCALES_YCBCR["Y'CbCr"]
        )

        # The blue channel has the largest direct inverse-matrix chroma gain.
        largest_gain = max(
            (1.0 - kb) / cb_scale,
            (1.0 - kr) / cr_scale,
            (kb * (1.0 - kb) / cb_scale + kr * (1.0 - kr) / cr_scale) / (1.0 - kr - kb),
        )

        return half_luma_level + largest_gain * half_chroma_level + 1e-12

    @pytest.mark.parametrize("bits", [8, 10, 12])
    def test_digital_video_ranges_and_offsets(self, bits: int) -> None:
        """Test legal-range anchors scale with bit depth, not code maximum."""

        depth_scale = 1 << (bits - 8)
        code_maximum = (1 << bits) - 1
        integer_ranges = np.array([16, 235, 16, 240]) * depth_scale

        np.testing.assert_array_equal(
            ranges_YCbCr(bits, is_legal=True, is_int=True),
            integer_ranges,
        )
        np.testing.assert_allclose(
            ranges_YCbCr(bits, is_legal=True, is_int=False),
            integer_ranges / code_maximum,
            atol=1e-14,
            rtol=0.0,
        )
        np.testing.assert_array_equal(
            ranges_YCbCr(bits, is_legal=False, is_int=False),
            [0.0, 1.0, -0.5, 0.5],
        )
        np.testing.assert_array_equal(
            offset_YCbCr(bits=bits, is_legal=True, is_int=True),
            [16 * depth_scale, 128 * depth_scale, 128 * depth_scale],
        )
        np.testing.assert_allclose(
            offset_YCbCr(bits=bits, is_legal=True, is_int=False),
            np.array(
                [16 * depth_scale, 128 * depth_scale, 128 * depth_scale],
                dtype=float,
            )
            / code_maximum,
            atol=1e-14,
            rtol=0.0,
        )

    def test_signed_bt709_luma_and_chroma_axes(self) -> None:
        """Test primary colours against signed +/- 0.5 chroma scales."""

        signed = RGB_to_YCbCr(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            K=WEIGHTS_YCBCR["ITU-R BT.709"],
            out_legal=False,
            out_int=False,
        )

        np.testing.assert_allclose(
            signed,
            [
                [0.2126, -0.1145721061, 0.5000000000],
                [0.7152, -0.3854278939, -0.4541529083],
                [0.0722, 0.5000000000, -0.0458470917],
            ],
            atol=1e-10,
            rtol=0.0,
        )

    @pytest.mark.parametrize(("bits", "rgb", "code_words"), BROADCAST_CODE_WORD_PARAMS)
    def test_standard_integer_code_words(
        self,
        bits: int,
        rgb: np.ndarray,
        code_words: np.ndarray,
    ) -> None:
        """Test exact broadcast code words for both common bit depths."""

        actual = RGB_to_YCbCr(
            rgb,
            K=WEIGHTS_YCBCR["ITU-R BT.709"],
            out_bits=bits,
            out_legal=True,
            out_int=True,
        )

        np.testing.assert_array_equal(actual, code_words)
        assert np.issubdtype(actual.dtype, np.integer)

    @pytest.mark.parametrize(("bits", "rgb", "code_words"), BROADCAST_CODE_WORD_PARAMS)
    def test_integer_and_normalised_code_words_denote_sample_colour(
        self,
        bits: int,
        rgb: np.ndarray,
        code_words: np.ndarray,
    ) -> None:
        """Test integer and normalised code values decode to one colour."""

        decoded_integer = YCbCr_to_RGB(
            code_words,
            in_bits=bits,
            in_legal=True,
            in_int=True,
        )

        code_maximum = (1 << bits) - 1
        normalised = code_words.astype(float) / code_maximum
        decoded_normalised = YCbCr_to_RGB(
            normalised,
            in_bits=bits,
            in_legal=True,
            in_int=False,
        )

        np.testing.assert_allclose(
            decoded_normalised,
            decoded_integer,
            atol=1e-12,
            rtol=0.0,
        )

        # Dividing by 2**bits and supplying the corresponding video ranges is
        # another common normalisation, with neutral chroma at exactly 1 / 2.
        video_normalised = code_words.astype(float) / (1 << bits)
        decoded_video_normalised = YCbCr_to_RGB(
            video_normalised,
            in_range=(16 / 256, 235 / 256, 16 / 256, 240 / 256),
        )

        np.testing.assert_allclose(
            decoded_video_normalised,
            decoded_integer,
            atol=1e-12,
            rtol=0.0,
        )

        np.testing.assert_allclose(
            decoded_integer,
            rgb,
            atol=self._quantization_atol(bits),
            rtol=0.0,
        )

    def test_selected_code_words_have_golden_monitor_values(self) -> None:
        """Test decoded float monitor values independently of round trips."""

        cases = [
            (8, [16, 128, 128], [0.0, 0.0, 0.0]),
            (8, [235, 128, 128], [1.0, 1.0, 1.0]),
            (8, [126, 128, 128], [110 / 219] * 3),
            (
                8,
                [133, 94, 159],
                [0.75218765, 0.49789467, 0.25259300],
            ),
            (
                8,
                [63, 102, 240],
                [1.00201187, 0.00229273, -0.00077027],
            ),
            (
                8,
                [32, 240, 118],
                [0.00275579, 0.00029563, 1.00085936],
            ),
            (10, [64, 512, 512], [0.0, 0.0, 0.0]),
            (10, [940, 512, 512], [1.0, 1.0, 1.0]),
            (10, [502, 512, 512], [0.5, 0.5, 0.5]),
            (
                10,
                [533, 374, 634],
                [0.74981402, 0.50049928, 0.24959259],
            ),
            (
                10,
                [250, 409, 960],
                [0.99972877, -0.00019944, -0.00098239],
            ),
            (
                10,
                [127, 960, 471],
                [-0.00014335, -0.00032346, 0.99971781],
            ),
        ]

        for bits, code_words, expected_rgb in cases:
            np.testing.assert_allclose(
                YCbCr_to_RGB(
                    np.array(code_words),
                    in_bits=bits,
                    in_legal=True,
                    in_int=True,
                ),
                expected_rgb,
                atol=1e-8,
                rtol=0.0,
            )

    @pytest.mark.parametrize(("bits", "rgb", "code_words"), BROADCAST_CODE_WORD_PARAMS)
    def test_legal_float_range_itself_does_not_change_colour(
        self,
        bits: int,
        rgb: np.ndarray,
        code_words: np.ndarray,
    ) -> None:
        """Test unquantized legal-range conversion is numerically reversible."""

        legal_float = RGB_to_YCbCr(
            rgb,
            K=WEIGHTS_YCBCR["ITU-R BT.709"],
            out_bits=bits,
            out_legal=True,
            out_int=False,
        )
        code_maximum = (1 << bits) - 1

        np.testing.assert_allclose(
            legal_float,
            code_words.astype(float) / code_maximum,
            atol=0.5 / code_maximum + 1e-12,
            rtol=0.0,
        )

        np.testing.assert_allclose(
            YCbCr_to_RGB(
                legal_float,
                in_bits=bits,
                in_legal=True,
                in_int=False,
            ),
            rgb,
            atol=TOLERANCE_ABSOLUTE_TESTS,
            rtol=0.0,
        )

    @pytest.mark.parametrize(("bits", "rgb", "code_words"), BROADCAST_CODE_WORD_PARAMS)
    def test_range_misinterpretation_is_larger_than_quantization(
        self,
        bits: int,
        rgb: np.ndarray,
        code_words: np.ndarray,
    ) -> None:
        """
        Test quantization error and a colour-changing range mismatch are
        distinguishable.
        """

        quantized_rgb = YCbCr_to_RGB(
            code_words,
            in_bits=bits,
            in_legal=True,
            in_int=True,
        )
        quantization_error = np.max(np.abs(quantized_rgb - rgb))
        quantization_atol = self._quantization_atol(bits)

        assert quantization_error <= quantization_atol

        normalised = code_words.astype(float) / ((1 << bits) - 1)
        full_swing_rgb = YCbCr_to_RGB(
            normalised,
            in_bits=bits,
            # Unsigned [0, 1] chroma span centered on 1 / 2, the range
            # conversion a legal-range tag must prevent.
            in_range=(0.0, 1.0, 0.0, 1.0),
        )
        range_error = np.max(np.abs(full_swing_rgb - rgb))

        assert range_error > quantization_atol

    @pytest.mark.parametrize("bits", [8, 10, 12])
    def test_guard_band_inputs_keep_extended_colour(self, bits: int) -> None:
        """Test super-black and super-white broadcast guard bands survive."""

        depth_scale = 1 << (bits - 8)
        foot_room_rgb = np.array([-16 / 219] * 3)
        head_room_rgb = np.array([239 / 219] * 3)
        cases = [
            (foot_room_rgb, [0, 128 * depth_scale, 128 * depth_scale]),
            (
                head_room_rgb,
                [255 * depth_scale, 128 * depth_scale, 128 * depth_scale],
            ),
        ]

        for rgb, expected_codes in cases:
            code_words = RGB_to_YCbCr(
                rgb,
                out_bits=bits,
                out_legal=True,
                out_int=True,
            )

            np.testing.assert_array_equal(code_words, expected_codes)
            np.testing.assert_allclose(
                YCbCr_to_RGB(
                    code_words,
                    in_bits=bits,
                    in_legal=True,
                    in_int=True,
                ),
                rgb,
                atol=1e-12,
                rtol=0.0,
            )


class TestRGB_to_YcCbcCrc:
    """
    Define :func:`colour.models.rgb.ycbcr.RGB_to_YcCbcCrc` definition unit
    tests methods.
    """

    def test_RGB_to_YcCbcCrc(
        self, xp: ModuleType, request: pytest.FixtureRequest
    ) -> None:
        """Test :func:`colour.models.rgb.ycbcr.RGB_to_YcCbcCrc` definition."""

        xp_assert_close(
            RGB_to_YcCbcCrc(xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp)),
            [0.37020379, 0.41137200, 0.77704674],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_YcCbcCrc(
                xp_as_array([0.18, 0.18, 0.18], xp=xp),
                out_bits=10,
                out_legal=True,
                out_int=True,
                is_12_bits_system=False,
            ),
            [422, 512, 512],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        host_arguments: dict[str, typing.Any] = {
            "RGB": [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
            "out_range": [0.0, 1.0, -0.5, 0.5],
        }
        expected = RGB_to_YcCbcCrc(**host_arguments)
        for parameter, host_value in host_arguments.items():
            native = xp_as_array(host_value, xp=xp)
            # Keep the physical input device while factories default to CPU.
            with xp.device("cpu") if xp.__name__ == "torch" else nullcontext():
                actual = RGB_to_YcCbcCrc(**{**host_arguments, parameter: native})
                assert array_namespace(actual) is array_namespace(native)
                assert getattr(actual, "device", None) == getattr(
                    native, "device", None
                )
                xp_assert_close(actual, expected, atol=TOLERANCE_ABSOLUTE_TESTS)
                if xp.__name__ != "numpy":
                    autodiff = request.getfixturevalue("autodiff")
                    autodiff(
                        lambda value, parameter=parameter: RGB_to_YcCbcCrc(
                            **{**host_arguments, parameter: value}
                        ),
                        native,
                    )

    def test_n_dimensional_RGB_to_YcCbcCrc(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ycbcr.RGB_to_YcCbcCrc` definition
        n-dimensional arrays support.
        """

        RGB = xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp)
        YcCbcCrc = as_ndarray(RGB_to_YcCbcCrc(RGB))

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 3), xp=xp)
        YcCbcCrc = xp.tile(xp_as_array(YcCbcCrc, xp=xp), (4,))
        YcCbcCrc = xp_reshape(xp_as_array(YcCbcCrc, xp=xp), (4, 3), xp=xp)
        xp_assert_close(RGB_to_YcCbcCrc(RGB), YcCbcCrc, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 3), xp=xp)
        YcCbcCrc = xp.tile(xp_as_array(YcCbcCrc, xp=xp), (4,))
        YcCbcCrc = xp_reshape(xp_as_array(YcCbcCrc, xp=xp), (4, 4, 3), xp=xp)
        xp_assert_close(RGB_to_YcCbcCrc(RGB), YcCbcCrc, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 4, 3), xp=xp)
        YcCbcCrc = xp.tile(xp_as_array(YcCbcCrc, xp=xp), (4,))
        YcCbcCrc = xp_reshape(xp_as_array(YcCbcCrc, xp=xp), (4, 4, 4, 3), xp=xp)
        xp_assert_close(RGB_to_YcCbcCrc(RGB), YcCbcCrc, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_RGB_to_YcCbcCrc(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.prismatic.RGB_to_YcCbcCrc` definition
        domain and range scale support.
        """

        RGB = xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp)
        YcCbcCrc = as_ndarray(RGB_to_YcCbcCrc(RGB))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    RGB_to_YcCbcCrc(RGB * factor),
                    YcCbcCrc * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_RGB_to_YcCbcCrc(self) -> None:
        """
        Test :func:`colour.models.rgb.ycbcr.RGB_to_YcCbcCrc` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_YcCbcCrc(cases)


class TestYcCbcCrc_to_RGB:
    """
    Define :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition unit tests
    methods.
    """

    def test_YcCbcCrc_to_RGB(
        self, xp: ModuleType, request: pytest.FixtureRequest
    ) -> None:
        """Test :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition."""

        xp_assert_close(
            YcCbcCrc_to_RGB(xp_as_array([0.37020379, 0.41137200, 0.77704674], xp=xp)),
            [0.45620519, 0.03081071, 0.04091952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            YcCbcCrc_to_RGB(
                xp_as_array([1689, 2048, 2048], xp=xp),
                in_bits=12,
                in_legal=True,
                in_int=True,
                is_12_bits_system=True,
            ),
            [0.18009037, 0.18009037, 0.18009037],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        host_arguments: dict[str, typing.Any] = {
            "YcCbcCrc": [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
            "in_range": [0.0, 1.0, -0.5, 0.5],
        }
        expected = YcCbcCrc_to_RGB(**host_arguments)
        for parameter, host_value in host_arguments.items():
            native = xp_as_array(host_value, xp=xp)
            # Keep the physical input device while factories default to CPU.
            with xp.device("cpu") if xp.__name__ == "torch" else nullcontext():
                actual = YcCbcCrc_to_RGB(**{**host_arguments, parameter: native})
                assert array_namespace(actual) is array_namespace(native)
                assert getattr(actual, "device", None) == getattr(
                    native, "device", None
                )
                xp_assert_close(actual, expected, atol=TOLERANCE_ABSOLUTE_TESTS)
                if xp.__name__ != "numpy":
                    autodiff = request.getfixturevalue("autodiff")
                    autodiff(
                        lambda value, parameter=parameter: YcCbcCrc_to_RGB(
                            **{**host_arguments, parameter: value}
                        ),
                        native,
                    )

    def test_n_dimensional_YcCbcCrc_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ycbcr.YcCbcCrc_to_RGB` definition
        n-dimensional arrays support.
        """

        YcCbcCrc = xp_as_array([0.37020379, 0.41137200, 0.77704674], xp=xp)
        RGB = as_ndarray(YcCbcCrc_to_RGB(YcCbcCrc))

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 3), xp=xp)
        YcCbcCrc = xp.tile(xp_as_array(YcCbcCrc, xp=xp), (4,))
        YcCbcCrc = xp_reshape(xp_as_array(YcCbcCrc, xp=xp), (4, 3), xp=xp)
        xp_assert_close(YcCbcCrc_to_RGB(YcCbcCrc), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 3), xp=xp)
        YcCbcCrc = xp.tile(xp_as_array(YcCbcCrc, xp=xp), (4,))
        YcCbcCrc = xp_reshape(xp_as_array(YcCbcCrc, xp=xp), (4, 4, 3), xp=xp)
        xp_assert_close(YcCbcCrc_to_RGB(YcCbcCrc), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 4, 3), xp=xp)
        YcCbcCrc = xp.tile(xp_as_array(YcCbcCrc, xp=xp), (4,))
        YcCbcCrc = xp_reshape(xp_as_array(YcCbcCrc, xp=xp), (4, 4, 4, 3), xp=xp)
        xp_assert_close(YcCbcCrc_to_RGB(YcCbcCrc), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_YcCbcCrc_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.prismatic.YcCbcCrc_to_RGB` definition
        domain and range scale support.
        """

        YcCbcCrc = xp_as_array([0.69943807, 0.38814348, 0.61264549], xp=xp)
        RGB = as_ndarray(YcCbcCrc_to_RGB(YcCbcCrc))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    YcCbcCrc_to_RGB(YcCbcCrc * factor),
                    RGB * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_YcCbcCrc_to_RGB(self) -> None:
        """
        Test :func:`colour.models.rgb.ycbcr.YcCbcCrc_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        YcCbcCrc_to_RGB(cases)
