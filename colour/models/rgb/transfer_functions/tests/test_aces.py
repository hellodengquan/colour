"""
Define the unit tests for the :mod:`colour.models.rgb.transfer_functions.aces`
module.
"""

from __future__ import annotations

import math
import typing

import numpy as np

from colour.constants import (
    TOLERANCE_ABSOLUTE_TESTS,
    TOLERANCE_RELATIVE_TESTS,
)
from colour.models.rgb.transfer_functions import (
    cctf_decoding,
    log_decoding,
    log_decoding_ACEScc,
    log_decoding_ACEScct,
    log_decoding_ACESproxy,
    log_encoding_ACEScc,
    log_encoding_ACEScct,
    log_encoding_ACESproxy,
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
    import pytest

    from colour.hints import Literal, ModuleType

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLogEncoding_ACESproxy",
    "TestLogDecoding_ACESproxy",
    "TestLogEncoding_ACEScc",
    "TestLogDecoding_ACEScc",
    "TestLogDecoding_ACEScct",
]


class TestLogEncoding_ACESproxy:
    """
    Define :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy`
    definition unit tests methods.
    """

    def test_log_encoding_ACESproxy(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy` definition.
        """

        xp_assert_close(
            log_encoding_ACESproxy(xp_as_array(0.0, xp=xp)),
            0.062561094819159,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_ACESproxy(xp_as_array(0.18, xp=xp)),
            0.416422287390029,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_ACESproxy(xp_as_array(0.18, xp=xp), 12),
            0.416361416361416,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_ACESproxy(xp_as_array(1.0, xp=xp)),
            0.537634408602151,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert (
            as_ndarray(log_encoding_ACESproxy(xp_as_array(0.18, xp=xp), out_int=True))
            == 426
        )

    def test_n_dimensional_log_encoding_ACESproxy(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy` definition n-dimensional arrays support.
        """

        lin_AP1 = 0.18
        ACESproxy = as_ndarray(log_encoding_ACESproxy(xp_as_array(lin_AP1, xp=xp)))

        lin_AP1 = xp.tile(xp_as_array(lin_AP1, xp=xp), (6,))
        ACESproxy = xp.tile(xp_as_array(ACESproxy, xp=xp), (6,))
        xp_assert_close(
            log_encoding_ACESproxy(lin_AP1),
            ACESproxy,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3), xp=xp)
        ACESproxy = xp_reshape(xp_as_array(ACESproxy, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_encoding_ACESproxy(lin_AP1),
            ACESproxy,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3, 1), xp=xp)
        ACESproxy = xp_reshape(xp_as_array(ACESproxy, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_encoding_ACESproxy(lin_AP1),
            ACESproxy,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_encoding_ACESproxy(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy` definition domain and range scale support.
        """

        lin_AP1 = 0.18
        ACESproxy = as_ndarray(log_encoding_ACESproxy(xp_as_array(lin_AP1, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_ACESproxy(xp_as_array(lin_AP1 * factor, xp=xp)),
                    ACESproxy * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_ACESproxy(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy` definition nan support.
        """

        log_encoding_ACESproxy(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_ACESproxy:
    """
    Define :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACESproxy`
    definition unit tests methods.
    """

    def test_log_decoding_ACESproxy(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACESproxy` definition.
        """

        xp_assert_close(
            log_decoding_ACESproxy(xp_as_array(0.062561094819159, xp=xp)),
            0.0,
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.01),
        )

        xp_assert_close(
            log_decoding_ACESproxy(xp_as_array(0.416422287390029, xp=xp)),
            0.18,
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.01),
        )

        xp_assert_close(
            log_decoding_ACESproxy(xp_as_array(0.416361416361416, xp=xp), 12),
            0.18,
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.01),
        )

        xp_assert_close(
            log_decoding_ACESproxy(xp_as_array(0.537634408602151, xp=xp)),
            1.0,
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.01),
        )

        xp_assert_close(
            log_decoding_ACESproxy(xp_as_array(426, xp=xp), in_int=True),
            0.18,
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.01),
        )

    def test_n_dimensional_log_decoding_ACESproxy(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACESproxy` definition n-dimensional arrays support.
        """

        ACESproxy = 0.416422287390029
        lin_AP1 = as_ndarray(log_decoding_ACESproxy(xp_as_array(ACESproxy, xp=xp)))

        ACESproxy = xp.tile(xp_as_array(ACESproxy, xp=xp), (6,))
        lin_AP1 = xp.tile(xp_as_array(lin_AP1, xp=xp), (6,))
        xp_assert_close(
            log_decoding_ACESproxy(ACESproxy),
            lin_AP1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        ACESproxy = xp_reshape(xp_as_array(ACESproxy, xp=xp), (2, 3), xp=xp)
        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_decoding_ACESproxy(ACESproxy),
            lin_AP1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        ACESproxy = xp_reshape(xp_as_array(ACESproxy, xp=xp), (2, 3, 1), xp=xp)
        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_decoding_ACESproxy(ACESproxy),
            lin_AP1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_decoding_ACESproxy(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACESproxy` definition domain and range scale support.
        """

        ACESproxy = 426 / (2**10 - 1)
        lin_AP1 = as_ndarray(log_decoding_ACESproxy(xp_as_array(ACESproxy, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_ACESproxy(xp_as_array(ACESproxy * factor, xp=xp)),
                    lin_AP1 * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

        cases: tuple[tuple[Literal[10, 12], list[int]], ...] = (
            (10, [64, 512, 940]),
            (12, [256, 2048, 3760]),
        )
        for bit_depth, codes in cases:
            ACESproxy_int = xp_as_array(codes, xp=xp)
            lin_AP1_int = log_decoding_ACESproxy(
                ACESproxy_int / (2**bit_depth - 1), bit_depth
            )
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    xp_assert_close(
                        log_decoding_ACESproxy(ACESproxy_int, bit_depth, in_int=True),
                        lin_AP1_int * factor,
                        atol=TOLERANCE_ABSOLUTE_TESTS,
                    )

    @ignore_numpy_errors
    def test_nan_log_decoding_ACESproxy(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACESproxy` definition nan support.
        """

        log_decoding_ACESproxy(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_ACEScc:
    """
    Define :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScc` definition unit tests methods.
    """

    def test_log_encoding_ACEScc(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScc` definition.
        """

        xp_assert_close(
            log_encoding_ACEScc(xp_as_array(0.0, xp=xp)),
            -0.358447488584475,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_ACEScc(xp_as_array(0.18, xp=xp)),
            0.413588402492442,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_ACEScc(xp_as_array(1.0, xp=xp)),
            0.554794520547945,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_ACEScc(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScc` definition n-dimensional arrays support.
        """

        lin_AP1 = 0.18
        ACEScc = as_ndarray(log_encoding_ACEScc(xp_as_array(lin_AP1, xp=xp)))

        lin_AP1 = xp.tile(xp_as_array(lin_AP1, xp=xp), (6,))
        ACEScc = xp.tile(xp_as_array(ACEScc, xp=xp), (6,))
        xp_assert_close(
            log_encoding_ACEScc(lin_AP1),
            ACEScc,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3), xp=xp)
        ACEScc = xp_reshape(xp_as_array(ACEScc, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_encoding_ACEScc(lin_AP1),
            ACEScc,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3, 1), xp=xp)
        ACEScc = xp_reshape(xp_as_array(ACEScc, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_encoding_ACEScc(lin_AP1),
            ACEScc,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_encoding_ACEScc(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScc` definition domain and range scale support.
        """

        lin_AP1 = 0.18
        ACEScc = as_ndarray(log_encoding_ACEScc(xp_as_array(lin_AP1, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_ACEScc(xp_as_array(lin_AP1 * factor, xp=xp)),
                    ACEScc * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_ACEScc(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScc` definition nan support.
        """

        log_encoding_ACEScc(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))

    def test_negative_exposure_spec_landmarks(self, xp: ModuleType) -> None:
        """
        Test the negative exposure landmarks against *ACEScc* specification
        S-2014-003.

        Negative *AP1* light is pinned to the floor code
        ``(log2(2^-16) + 9.72) / 17.52``, and the shadow toe joins the
        regular logarithmic branch at ``x_brk = 2^-15``.
        """

        code_floor = (math.log2(2**-16) + 9.72) / 17.52
        x_break = 2**-15
        code_break = (9.72 - 15) / 17.52

        xp_assert_close(
            code_floor,
            -0.358447488584475,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            log_encoding_ACEScc(xp_as_array(0.0, xp=xp)),
            code_floor,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            log_encoding_ACEScc(xp_as_array(x_break, xp=xp)),
            code_break,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_negative_exposure_levels_merged_at_encoding(self, xp: ModuleType) -> None:
        """
        Distinct negative exposure stops are merged onto one floor code.

        Every strictly negative linear exposure, however slight, is encoded
        as the same S-2014-003 floor value: those exposure levels are merged
        while encoding and cannot be recovered from the code value.
        """

        code_floor = (math.log2(2**-16) + 9.72) / 17.52

        negative_stops = -(2.0 ** np.arange(0, 17))
        codes = as_ndarray(log_encoding_ACEScc(xp_as_array(negative_stops, xp=xp)))

        assert codes.shape == negative_stops.shape
        assert len(np.unique(codes)) == 1
        assert np.all(codes == code_floor)

        tiny_negatives = np.array([-1.0e-12, -1.0e-30, -1.0e-300])
        tiny_codes = as_ndarray(log_encoding_ACEScc(xp_as_array(tiny_negatives, xp=xp)))
        assert np.all(tiny_codes == code_floor)

        # Black shares the exact same floor code.
        black_code = float(as_ndarray(log_encoding_ACEScc(xp_as_array(0.0, xp=xp))))
        assert black_code == code_floor

    def test_shadow_toe_exposure_levels_remain_distinct(self, xp: ModuleType) -> None:
        """
        Sub-breakpoint non-negative shadow levels stay recoverable.

        Inside the ``0 <= lin_AP1 < 2^-15`` toe branch, exposure stops map
        to pairwise distinct, monotonically increasing code values and
        round-trip to linear light at relative test precision, contrasting
        with the merged negative exposures.
        """

        code_floor = (math.log2(2**-16) + 9.72) / 17.52
        shadow_stops = np.array([2.0**exponent for exponent in range(-24, -15)])

        codes = as_ndarray(log_encoding_ACEScc(xp_as_array(shadow_stops, xp=xp)))

        assert len(np.unique(codes)) == shadow_stops.size
        assert np.all(np.diff(codes) > 0.0)
        assert np.all(codes > code_floor)

        restored = log_decoding_ACEScc(xp_as_array(codes, xp=xp))
        xp_assert_close(
            restored,
            xp_as_array(shadow_stops, xp=xp),
            rtol=TOLERANCE_RELATIVE_TESTS,
            atol=0.0,
        )

    def test_negative_exposure_junction_follows_acescc_definition(
        self, xp: ModuleType
    ) -> None:
        """
        Test the shadow-to-normal exposure join of the *ACEScc* curve.

        At ``lin_AP1 = 2^-15`` the two branches share one code value (C0
        continuity), but S-2014-003 defines a corner rather than a smooth
        join: the logarithmic branch slope is twice the toe branch slope.
        """

        x_break = 2**-15
        code_break = (9.72 - 15) / 17.52

        xp_assert_close(
            log_encoding_ACEScc(xp_as_array(x_break, xp=xp)),
            code_break,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        step = x_break * 1.0e-3
        code_at_break = float(
            as_ndarray(log_encoding_ACEScc(xp_as_array(x_break, xp=xp)))
        )
        code_before = float(
            as_ndarray(log_encoding_ACEScc(xp_as_array(x_break - step, xp=xp)))
        )
        code_after = float(
            as_ndarray(log_encoding_ACEScc(xp_as_array(x_break + step, xp=xp)))
        )
        slope_before = (code_at_break - code_before) / step
        slope_after = (code_after - code_at_break) / step

        # Finite differences recover the spec 2:1 slope change.
        assert 1.99 < slope_after / slope_before < 2.01

        slope_toe_spec = 0.5 / (17.52 * math.log(2.0) * (2**-16 + 0.5 * x_break))
        slope_log_spec = 1.0 / (17.52 * math.log(2.0) * x_break)
        assert abs(slope_log_spec / slope_toe_spec - 2.0) < 1.0e-12


class TestLogDecoding_ACEScc:
    """
    Define :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScc` definition unit tests methods.
    """

    def test_log_decoding_ACEScc(
        self, xp: ModuleType, request: pytest.FixtureRequest
    ) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScc` definition.
        """

        xp_assert_close(
            log_decoding_ACEScc(xp_as_array(-0.358447488584475, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_ACEScc(xp_as_array(0.413588402492442, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_ACEScc(xp_as_array(0.554794520547945, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_ACEScc(
                xp_as_array(
                    [(9.72 - 15) / 17.52, (np.log2(65504) + 9.72) / 17.52],
                    xp=xp,
                )
            ),
            [2**-15, 65504],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        saturated = xp_as_array([10.0, 100.0], xp=xp)
        xp_assert_close(
            log_decoding_ACEScc(saturated),
            [65504, 65504],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        for decoder in (log_decoding, cctf_decoding):
            xp_assert_close(
                decoder(saturated, function="ACEScc"),
                [65504, 65504],
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )
            if xp.__name__ != "numpy":
                _result, (gradient,), _inputs = request.getfixturevalue("autodiff")(
                    lambda value, decoder=decoder: decoder(value, function="ACEScc"),
                    xp_as_array([0.18, 0.5, 10.0, 100.0], xp=xp),
                )
                xp_assert_close(
                    gradient[-2:], [0.0, 0.0], atol=TOLERANCE_ABSOLUTE_TESTS
                )

    def test_n_dimensional_log_decoding_ACEScc(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScc` definition n-dimensional arrays support.
        """

        ACEScc = 0.413588402492442
        lin_AP1 = as_ndarray(log_decoding_ACEScc(xp_as_array(ACEScc, xp=xp)))

        ACEScc = xp.tile(xp_as_array(ACEScc, xp=xp), (6,))
        lin_AP1 = xp.tile(xp_as_array(lin_AP1, xp=xp), (6,))
        xp_assert_close(
            log_decoding_ACEScc(ACEScc),
            lin_AP1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        ACEScc = xp_reshape(xp_as_array(ACEScc, xp=xp), (2, 3), xp=xp)
        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_decoding_ACEScc(ACEScc),
            lin_AP1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        ACEScc = xp_reshape(xp_as_array(ACEScc, xp=xp), (2, 3, 1), xp=xp)
        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_decoding_ACEScc(ACEScc),
            lin_AP1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_decoding_ACEScc(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScc` definition domain and range scale support.
        """

        ACEScc = 0.413588402492442
        lin_AP1 = as_ndarray(log_decoding_ACEScc(xp_as_array(ACEScc, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_ACEScc(xp_as_array(ACEScc * factor, xp=xp)),
                    lin_AP1 * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_ACEScc(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScc` definition nan support.
        """

        log_decoding_ACEScc(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))

    def test_negative_exposure_merged_codes_decode_to_black(
        self, xp: ModuleType
    ) -> None:
        """
        Decoding the merged floor code returns black for every stop.

        A negative linear exposure encoded through the S-2014-003 clamp is
        restored as zero linear light, not as a negative value: the encoding
        merge cannot be undone by decoding.
        """

        code_floor = (math.log2(2**-16) + 9.72) / 17.52
        negative_stops = -(2.0 ** np.arange(0, 17))

        codes = log_encoding_ACEScc(xp_as_array(negative_stops, xp=xp))
        restored = as_ndarray(log_decoding_ACEScc(codes))

        xp_assert_close(
            restored,
            np.zeros(negative_stops.size),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            log_decoding_ACEScc(xp_as_array(code_floor, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_negative_exposure_sub_floor_codes_decode_to_distinct_stops(
        self, xp: ModuleType
    ) -> None:
        """
        Sub-floor codes retained after black subtraction decode to distinct
        negative exposure stops.

        Grading in the *ACEScc* workspace can push values below the encoding
        floor code without clipping them.  The inverse toe maps a code
        ``n`` logarithmic stops below the floor to
        ``2^-15 * (2^-n - 1)`` linear light: the levels stay pairwise
        distinct and ordered, remain above the ``-2^-15`` inverse limit and
        their linear spacing halves per stop.
        """

        code_floor = (math.log2(2**-16) + 9.72) / 17.52
        x_break = 2**-15

        stops_below_floor = np.arange(0, 9)
        codes = code_floor - stops_below_floor / 17.52
        expected_linear = x_break * (2.0 ** (-stops_below_floor) - 1.0)

        linear = as_ndarray(log_decoding_ACEScc(xp_as_array(codes, xp=xp)))

        xp_assert_close(linear, expected_linear, atol=TOLERANCE_ABSOLUTE_TESTS)
        assert linear[0] == 0.0
        assert np.all(np.diff(linear) < 0.0)
        assert len(np.unique(linear)) == stops_below_floor.size
        assert np.all(linear > -x_break)

        gaps = -np.diff(linear)
        xp_assert_close(
            gaps[:-1] / gaps[1:],
            np.full(gaps.size - 1, 2.0),
            rtol=TOLERANCE_RELATIVE_TESTS,
            atol=0.0,
        )

    def test_negative_exposure_deep_codes_approach_toe_asymptote(
        self, xp: ModuleType
    ) -> None:
        """
        Very deep sub-floor codes converge onto the inverse toe limit.

        The extrapolated *ACEScc* toe is bounded: as the code is pushed
        arbitrarily below the floor, restored linear light approaches
        ``-2^-15``, so deeper codes cease to add separable negative range.
        """

        code_floor = (math.log2(2**-16) + 9.72) / 17.52

        codes = code_floor - np.array([1.0, 10.0, 100.0]) / 17.52
        linear = as_ndarray(log_decoding_ACEScc(xp_as_array(codes, xp=xp)))

        assert linear[0] > linear[1] > linear[2]
        xp_assert_close(linear[-1], -(2**-15), atol=1.0e-12)

    def test_negative_exposure_junction_follows_acescc_definition(
        self, xp: ModuleType
    ) -> None:
        """
        Test the inverse junction at ``ACEScc = (9.72 - 15) / 17.52``.

        Decoding is continuous at the shadow/log threshold and reproduces
        ``lin_AP1 = 2^-15``; the inverse branches keep the spec corner,
        with the low-code slope twice the high-code slope.
        """

        code_break = (9.72 - 15) / 17.52
        x_break = 2**-15

        xp_assert_close(
            log_decoding_ACEScc(xp_as_array(code_break, xp=xp)),
            x_break,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        step = 1.0e-4
        linear_at_break = float(
            as_ndarray(log_decoding_ACEScc(xp_as_array(code_break, xp=xp)))
        )
        linear_before = float(
            as_ndarray(log_decoding_ACEScc(xp_as_array(code_break - step, xp=xp)))
        )
        linear_after = float(
            as_ndarray(log_decoding_ACEScc(xp_as_array(code_break + step, xp=xp)))
        )
        slope_before = (linear_at_break - linear_before) / step
        slope_after = (linear_after - linear_at_break) / step

        assert 1.98 < slope_before / slope_after < 2.02


class TestLogEncoding_ACEScct:
    """
    Define :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScct` definition unit tests methods.
    """

    def test_log_encoding_ACEScct(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScct` definition.
        """

        xp_assert_close(
            log_encoding_ACEScct(xp_as_array(0.0, xp=xp)),
            0.072905534195835495,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_ACEScct(xp_as_array(0.18, xp=xp)),
            0.413588402492442,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_ACEScct(xp_as_array(1.0, xp=xp)),
            0.554794520547945,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_ACEScct(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScct` definition n-dimensional arrays support.
        """

        lin_AP1 = 0.18
        ACEScct = as_ndarray(log_encoding_ACEScct(xp_as_array(lin_AP1, xp=xp)))

        lin_AP1 = xp.tile(xp_as_array(lin_AP1, xp=xp), (6,))
        ACEScct = xp.tile(xp_as_array(ACEScct, xp=xp), (6,))
        xp_assert_close(
            log_encoding_ACEScct(lin_AP1),
            ACEScct,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3), xp=xp)
        ACEScct = xp_reshape(xp_as_array(ACEScct, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_encoding_ACEScct(lin_AP1),
            ACEScct,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3, 1), xp=xp)
        ACEScct = xp_reshape(xp_as_array(ACEScct, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_encoding_ACEScct(lin_AP1),
            ACEScct,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_encoding_ACEScct(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScct` definition domain and range scale support.
        """

        lin_AP1 = 0.18
        ACEScct = as_ndarray(log_encoding_ACEScct(xp_as_array(lin_AP1, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_ACEScct(xp_as_array(lin_AP1 * factor, xp=xp)),
                    ACEScct * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    def test_ACEScc_equivalency_log_encoding_ACEScct(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScct` definition ACEScc equivalency, and explicit requirement
        specified by AMPAS ACES specification S-2016-001
        (https://github.com/ampas/aces-dev/blob/v1.0.3/documents/LaTeX/\
S-2016-001/introduction.tex#L14)
        """

        equiv = np.linspace(0.0078125, 222.86094420380761, 100)
        xp_assert_close(
            log_encoding_ACEScct(xp_as_array(equiv, xp=xp)),
            as_ndarray(log_encoding_ACEScc(xp_as_array(equiv, xp=xp))),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_log_encoding_ACEScct(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScct` definition nan support.
        """

        log_encoding_ACEScct(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))

    def test_log_encoding_ACEScct_autodiff(
        self, xp: ModuleType, autodiff: typing.Callable
    ) -> None:
        """Test numerical gradients through the piecewise transfer curve."""

        autodiff(log_encoding_ACEScct, xp_as_array([0.0, 0.18], xp=xp))

    def test_negative_exposure_spec_landmarks(self, xp: ModuleType) -> None:
        """
        Test the *ACEScct* linear-toe landmarks from specification
        S-2016-001.

        The toe is ``y = A * x + B`` up to ``X_BRK = 0.0078125 = 2^-7`` and
        ``Y_BRK = 0.155251141552511``; unlike *ACEScc* the linear branch is
        defined for every value below the breakpoint, negatives included.
        """

        x_break = 0.0078125
        y_break = 0.155251141552511
        slope = 10.5402377416545
        offset = 0.0729055341958355

        assert x_break == 2**-7
        xp_assert_close(slope * x_break + offset, y_break, atol=1.0e-12)
        xp_assert_close(
            log_encoding_ACEScct(xp_as_array(0.0, xp=xp)),
            offset,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            log_encoding_ACEScct(xp_as_array(x_break, xp=xp)),
            y_break,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_negative_exposure_levels_remain_distinct(self, xp: ModuleType) -> None:
        """
        Negative exposure stops keep distinct *ACEScct* codes.

        The linear toe extends over every ``lin_AP1 <= X_BRK`` value, so
        all negative stops follow ``y = A * x + B`` exactly: none are
        merged by encoding, in contrast to the *ACEScc* negative clamp.
        """

        y_break = 0.155251141552511
        slope = 10.5402377416545
        offset = 0.0729055341958355

        negative_stops = -(2.0 ** np.arange(0, 17))
        codes = as_ndarray(log_encoding_ACEScct(xp_as_array(negative_stops, xp=xp)))

        assert len(np.unique(codes)) == negative_stops.size
        assert np.all(np.diff(codes) < 0.0)
        assert np.all(codes <= y_break)
        xp_assert_close(
            codes,
            slope * negative_stops + offset,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        shadow_negatives = np.array([-1.0e-3, -1.0e-6, -1.0e-12])
        shadow_codes = as_ndarray(
            log_encoding_ACEScct(xp_as_array(shadow_negatives, xp=xp))
        )
        assert len(np.unique(shadow_codes)) == shadow_negatives.size
        xp_assert_close(
            shadow_codes,
            slope * shadow_negatives + offset,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_negative_exposure_junction_is_c1_continuous(self, xp: ModuleType) -> None:
        """
        Test the shadow-to-normal exposure join of the *ACEScct* curve.

        S-2016-001 fits ``A`` and ``B`` so the linear toe joins the
        logarithmic branch at ``X_BRK`` with matching value and matching
        derivative (C1 continuity), unlike the deliberate *ACEScc* corner.
        """

        x_break = 0.0078125
        y_break = 0.155251141552511
        slope = 10.5402377416545
        offset = 0.0729055341958355

        xp_assert_close(slope * x_break + offset, y_break, atol=1.0e-12)
        xp_assert_close(
            log_encoding_ACEScct(xp_as_array(x_break, xp=xp)),
            y_break,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        log_slope_at_break = 1.0 / (17.52 * math.log(2.0) * x_break)
        xp_assert_close(
            log_slope_at_break,
            slope,
            rtol=TOLERANCE_RELATIVE_TESTS,
            atol=0.0,
        )

        step = x_break * 1.0e-3
        code_at_break = float(
            as_ndarray(log_encoding_ACEScct(xp_as_array(x_break, xp=xp)))
        )
        code_before = float(
            as_ndarray(log_encoding_ACEScct(xp_as_array(x_break - step, xp=xp)))
        )
        code_after = float(
            as_ndarray(log_encoding_ACEScct(xp_as_array(x_break + step, xp=xp)))
        )
        finite_slope_before = (code_at_break - code_before) / step
        finite_slope_after = (code_after - code_at_break) / step

        assert 0.99 < finite_slope_after / finite_slope_before < 1.01


class TestLogDecoding_ACEScct:
    """
    Define :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScct` definition unit tests methods.
    """

    def test_log_decoding_ACEScct(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScct` definition.
        """

        xp_assert_close(
            log_decoding_ACEScct(xp_as_array(0.072905534195835495, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_ACEScct(xp_as_array(0.41358840249244228, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_ACEScct(xp_as_array(0.554794520547945, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_ACEScct(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScct` definition n-dimensional arrays support.
        """

        ACEScct = 0.413588402492442
        lin_AP1 = as_ndarray(log_decoding_ACEScct(xp_as_array(ACEScct, xp=xp)))

        ACEScct = xp.tile(xp_as_array(ACEScct, xp=xp), (6,))
        lin_AP1 = xp.tile(xp_as_array(lin_AP1, xp=xp), (6,))
        xp_assert_close(
            log_decoding_ACEScct(ACEScct),
            lin_AP1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        ACEScct = xp_reshape(xp_as_array(ACEScct, xp=xp), (2, 3), xp=xp)
        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_decoding_ACEScct(ACEScct),
            lin_AP1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        ACEScct = xp_reshape(xp_as_array(ACEScct, xp=xp), (2, 3, 1), xp=xp)
        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_decoding_ACEScct(ACEScct),
            lin_AP1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_decoding_ACEScct(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScct` definition domain and range scale support.
        """

        ACEScc = 0.413588402492442
        lin_AP1 = as_ndarray(log_decoding_ACEScct(xp_as_array(ACEScc, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_ACEScct(xp_as_array(ACEScc * factor, xp=xp)),
                    lin_AP1 * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    def test_ACEScc_equivalency_log_decoding_ACEScct(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScct` definition ACEScc equivalency, and explicit requirement
        specified by AMPAS ACES specification S-2016-001
        (https://github.com/ampas/aces-dev/blob/v1.0.3/documents/LaTeX/\
S-2016-001/introduction.tex#L14)
        """

        equiv = np.linspace(0.15525114155251146, 1.0, 100)
        xp_assert_close(
            log_decoding_ACEScct(xp_as_array(equiv, xp=xp)),
            as_ndarray(log_decoding_ACEScc(xp_as_array(equiv, xp=xp))),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_log_decoding_ACEScct(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScct` definition nan support.
        """

        log_decoding_ACEScct(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))

    def test_negative_exposure_sub_black_codes_decode_through_toe(
        self, xp: ModuleType
    ) -> None:
        """
        Codes at and below ``B`` decode through the linear toe.

        Subtracting black in the grading workspace produces codes below the
        zero-light code ``B``, including negative codes.  They decode via
        ``lin_AP1 = (ACEScct - B) / A`` into pairwise distinct, ordered
        negative exposures and re-encode to the exact same codes.
        """

        slope = 10.5402377416545
        offset = 0.0729055341958355

        codes = np.array([offset, 0.05, 0.0, -0.1, -0.5, -2.0])
        expected_linear = (codes - offset) / slope

        linear = as_ndarray(log_decoding_ACEScct(xp_as_array(codes, xp=xp)))

        xp_assert_close(linear, expected_linear, atol=TOLERANCE_ABSOLUTE_TESTS)
        assert abs(linear[0]) < TOLERANCE_ABSOLUTE_TESTS
        assert np.all(np.diff(linear) < 0.0)
        assert len(np.unique(linear)) == codes.size

        xp_assert_close(
            log_encoding_ACEScct(xp_as_array(linear, xp=xp)),
            xp_as_array(codes, xp=xp),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_negative_exposure_toe_keeps_uniform_spacing(self, xp: ModuleType) -> None:
        """
        Equal code steps below ``B`` restore equal linear steps.

        The unbounded linear toe preserves a uniform exposure spacing with
        no asymptotic merging, contrasting with the bounded *ACEScc*
        extrapolated toe.
        """

        slope = 10.5402377416545
        offset = 0.0729055341958355

        stops = np.arange(0, 8)
        codes = offset - 0.05 * stops

        linear = as_ndarray(log_decoding_ACEScct(xp_as_array(codes, xp=xp)))
        gaps = np.diff(linear)

        xp_assert_close(
            gaps,
            np.full(stops.size - 1, -0.05 / slope),
            rtol=TOLERANCE_RELATIVE_TESTS,
            atol=0.0,
        )

    def test_negative_exposure_round_trip(self, xp: ModuleType) -> None:
        """
        Negative exposure stops survive an *ACEScct* encode/decode round
        trip at relative test precision.
        """

        negative_stops = -(2.0 ** np.arange(0, 17))

        encoded = log_encoding_ACEScct(xp_as_array(negative_stops, xp=xp))
        restored = log_decoding_ACEScct(encoded)

        xp_assert_close(
            restored,
            xp_as_array(negative_stops, xp=xp),
            rtol=TOLERANCE_RELATIVE_TESTS,
            atol=0.0,
        )

    def test_negative_exposure_junction_is_c1_continuous(self, xp: ModuleType) -> None:
        """
        Test the inverse junction at ``ACEScct = Y_BRK``.

        Decoding reproduces ``lin_AP1 = X_BRK`` and the inverse linear and
        logarithmic branches meet with matching slopes (C1 continuity),
        as required by S-2016-001.
        """

        x_break = 0.0078125
        y_break = 0.155251141552511

        xp_assert_close(
            log_decoding_ACEScct(xp_as_array(y_break, xp=xp)),
            x_break,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        step = 1.0e-4
        linear_at_break = float(
            as_ndarray(log_decoding_ACEScct(xp_as_array(y_break, xp=xp)))
        )
        linear_before = float(
            as_ndarray(log_decoding_ACEScct(xp_as_array(y_break - step, xp=xp)))
        )
        linear_after = float(
            as_ndarray(log_decoding_ACEScct(xp_as_array(y_break + step, xp=xp)))
        )
        slope_before = (linear_at_break - linear_before) / step
        slope_after = (linear_after - linear_at_break) / step

        assert 0.99 < slope_before / slope_after < 1.01
