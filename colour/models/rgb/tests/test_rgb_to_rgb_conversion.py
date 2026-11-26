"""
Define colour conversion unit tests for different source white points.

The tests exercise :func:`colour.models.RGB_to_RGB` and
:func:`colour.models.matrix_RGB_to_RGB` around two workflows encountered when
bringing plates from different sources into one working *RGB* colourspace:

*   The landing coordinates of a neutral grey are pinned for each selected
    chromatic adaptation transform, while coloured swatches are checked
    against the explicit channel contributions of the new primary basis.
*   The input and output encoding states are declared independently with
    ``apply_cctf_decoding`` and ``apply_cctf_encoding``.  Every combination is
    compared to a value derived independently from a literal conversion
    matrix and simple transfer functions, and the adapted conversions are
    distinguished from the ``chromatic_adaptation_transform=None`` result.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike

import numpy as np
import pytest
from numpy.testing import assert_allclose

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import (
    RGB_COLOURSPACES,
    RGB_Colourspace,
    RGB_to_RGB,
    matrix_RGB_to_RGB,
    xy_to_XYZ,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestNeutralGreyWhitePointLanding",
    "TestPrimaryBasisChannelContributions",
    "TestDeclaredEncodingStates",
]

_SRGB = RGB_COLOURSPACES["sRGB"]
_ACES2065_1 = RGB_COLOURSPACES["ACES2065-1"]
_ACESCG = RGB_COLOURSPACES["ACEScg"]

# Reference conversion matrices from *sRGB* (D65) to *ACES2065-1* (D60).
# Each literal was derived independently by composing
# ``M_XYZ_to_RGB(output) @ M_CAT @ M_RGB_to_XYZ(input)`` and is pinned here so
# that an adaptation or primary matrix regression is detected directly.
M_SRGB_TO_ACES2065_1_CAT02 = np.array(
    [
        [0.4395856442, 0.3839294030, 0.1765327364],
        [0.0895395735, 0.8147498351, 0.0956836061],
        [0.0173871832, 0.1087391143, 0.8738205876],
    ]
)

M_SRGB_TO_ACES2065_1_BRADFORD = np.array(
    [
        [0.4396430040, 0.3830054714, 0.1773993089],
        [0.0897157319, 0.8134750538, 0.0967822524],
        [0.0175127205, 0.1115514385, 0.8708827930],
    ]
)

M_SRGB_TO_ACES2065_1_NO_CAT = np.array(
    [
        [0.4329401822, 0.3754007997, 0.1893982296],
        [0.0893519795, 0.8165687816, 0.1030198606],
        [0.0191311640, 0.1181572406, 0.9421850433],
    ]
)


def _synthetic_colourspaces() -> tuple[RGB_Colourspace, RGB_Colourspace, np.ndarray]:
    """
    Build a controlled source (D65) / target (D50) pair with exact matrices.

    The returned source ``RGB_Colourspace`` has an explicit *RGB* to *XYZ*
    matrix whose rows sum to the source white tristimulus values, while the
    target has a diagonal *XYZ* to *RGB* matrix normalising by its white.  The
    third returned value is the source *XYZ* white point, allowing the
    conversion matrices to be derived by simple row normalisation.
    """

    source_white_xy = np.array([0.31270, 0.32900])
    target_white_xy = np.array([0.34570, 0.35850])
    source_white_xyz = xy_to_XYZ(source_white_xy)
    target_white_xyz = xy_to_XYZ(target_white_xy)

    # Off-diagonal contributions are fixed; diagonals make the row sums equal
    # to the source white point so that ``M_RGB_to_XYZ @ (1, 1, 1)`` is exact.
    off_diagonal = np.array(
        [
            [0.00, 0.20, 0.15],
            [0.10, 0.00, 0.20],
            [0.05, 0.15, 0.00],
        ]
    )
    matrix_rgb_to_xyz = off_diagonal.copy()
    for i in range(3):
        matrix_rgb_to_xyz[i, i] = source_white_xyz[i] - off_diagonal[i].sum()

    # ``M_XYZ_to_RGB @ target_white_XYZ == (1, 1, 1)`` exactly.
    matrix_xyz_to_rgb = np.diag(1.0 / target_white_xyz)

    primaries = np.array(
        [
            [0.6400, 0.3300],
            [0.3000, 0.6000],
            [0.1500, 0.0600],
        ]
    )

    # Exact inverse transfer pair: decoding squares, encoding square-roots.
    def decoding(values: ArrayLike) -> np.ndarray:
        """Decode a declared encoded sample by squaring it."""

        return np.square(np.asarray(values, dtype=float))

    def encoding(values: ArrayLike) -> np.ndarray:
        """Encode a linear sample by taking its square root."""

        return np.sqrt(np.asarray(values, dtype=float))

    source = RGB_Colourspace(
        "Unit Test Source - D65",
        primaries,
        source_white_xy,
        "D65",
        matrix_RGB_to_XYZ=matrix_rgb_to_xyz,
        cctf_encoding=encoding,
        cctf_decoding=decoding,
    )
    target = RGB_Colourspace(
        "Unit Test Target - D50",
        primaries,
        target_white_xy,
        "D50",
        matrix_XYZ_to_RGB=matrix_xyz_to_rgb,
        cctf_encoding=encoding,
        cctf_decoding=decoding,
    )

    return source, target, source_white_xyz


class TestNeutralGreyWhitePointLanding:
    """
    Test the neutral grey landing coordinates under white point adaptation.
    """

    @pytest.mark.parametrize(
        ("cat", "expected"),
        [
            pytest.param(
                "CAT02",
                [0.1800086010, 0.1799951426, 0.1799904393],
                id="cat02",
            ),
            pytest.param(
                "Bradford",
                [0.1800086012, 0.1799951469, 0.1799904514],
                id="bradford",
            ),
            pytest.param(
                None,
                [0.1795930581, 0.1816093119, 0.1943052206],
                id="no-adaptation",
            ),
        ],
    )
    def test_grey_landing_is_pinned_for_selected_adaptation(
        self, cat: str | None, expected: list[float]
    ) -> None:
        """
        Pin the *sRGB* to *ACES2065-1* 18% grey landing for each CAT.

        The adapted methods keep the three channels within floating point
        neighbourhood of the same value, whereas omitting the adaptation
        shifts the grey visibly toward the destination primary basis.
        """

        RGB = np.array([0.18, 0.18, 0.18])

        assert_allclose(
            RGB_to_RGB(RGB, _SRGB, _ACES2065_1, cat),
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @pytest.mark.parametrize("cat", ["CAT02", "Bradford", "XYZ Scaling"])
    @pytest.mark.parametrize("level", [0.0, 0.18, 0.5, 1.0])
    def test_adapted_grey_maps_onto_target_white_exactly(
        self, cat: str, level: float
    ) -> None:
        """
        Assert that a coherent source / target pair maps neutral grey onto
        equal destination channels whenever white point adaptation is enabled.
        """

        source, target, _ = _synthetic_colourspaces()

        M = matrix_RGB_to_RGB(source, target, cat)

        assert_allclose(M @ np.ones(3), np.ones(3), atol=1e-12)
        assert_allclose(
            RGB_to_RGB(np.full(3, level), source, target, cat),
            np.full(3, level),
            atol=1e-12,
        )

    def test_unadapted_grey_follows_source_to_target_white_ratio(self) -> None:
        """
        Derive the no-CAT grey landing independently as the source / target
        white tristimulus ratio and show it is not a neutral destination.
        """

        source, target, source_white_xyz = _synthetic_colourspaces()
        target_white_xyz = xy_to_XYZ(np.array([0.34570, 0.35850]))

        M = matrix_RGB_to_RGB(source, target, None)
        expected_white_ratio = source_white_xyz / target_white_xyz

        assert_allclose(
            M @ np.ones(3),
            [0.9856478156, 1.0000000000, 1.3199026492],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        assert_allclose(M @ np.ones(3), expected_white_ratio, atol=1e-12)

        grey = 0.18
        result = RGB_to_RGB(np.full(3, grey), source, target, None)

        assert_allclose(result, grey * expected_white_ratio, atol=1e-12)
        # The blue channel carries the largest residual cast because D50 is
        # warmer than the D65 source.
        assert result[0] < result[1] < result[2]


class TestPrimaryBasisChannelContributions:
    """
    Test coloured swatches against the explicit primary basis contributions.
    """

    @pytest.mark.parametrize(
        ("cat", "literal"),
        [
            pytest.param("CAT02", M_SRGB_TO_ACES2065_1_CAT02, id="cat02"),
            pytest.param("Bradford", M_SRGB_TO_ACES2065_1_BRADFORD, id="bradford"),
            pytest.param(None, M_SRGB_TO_ACES2065_1_NO_CAT, id="no-adaptation"),
        ],
    )
    def test_conversion_matrix_matches_pinned_literal(
        self, cat: str | None, literal: np.ndarray
    ) -> None:
        """Pin the full conversion matrix for each declared CAT method."""

        assert_allclose(
            matrix_RGB_to_RGB(_SRGB, _ACES2065_1, cat),
            literal,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @pytest.mark.parametrize(
        ("swatch", "column"),
        [
            pytest.param([1.0, 0.0, 0.0], 0, id="red-column"),
            pytest.param([0.0, 1.0, 0.0], 1, id="green-column"),
            pytest.param([0.0, 0.0, 1.0], 2, id="blue-column"),
        ],
    )
    def test_primary_swatches_equal_matrix_columns(
        self, swatch: list[float], column: int
    ) -> None:
        """
        A unit primary isolates one matrix column, exposing exactly how much
        that source primary contributes to each working-space channel.
        """

        result = RGB_to_RGB(np.array(swatch), _SRGB, _ACES2065_1, "CAT02")

        assert_allclose(
            result,
            M_SRGB_TO_ACES2065_1_CAT02[:, column],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_mixed_swatch_channel_contributions_are_explicit(self) -> None:
        """
        Recompute a mixed swatch channel by channel from the literal matrix
        rows and pin the result, documenting the basis change contributions.
        """

        r, g, b = 0.8, 0.4, 0.1

        # Explicit row dot products of the pinned CAT02 matrix.
        expected = np.array(
            [
                r * 0.4395856442 + g * 0.3839294030 + b * 0.1765327364,
                r * 0.0895395735 + g * 0.8147498351 + b * 0.0956836061,
                r * 0.0173871832 + g * 0.1087391143 + b * 0.8738205876,
            ]
        )
        assert_allclose(
            expected,
            [0.5228935502, 0.4070999535, 0.1447874511],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        result = RGB_to_RGB(np.array([r, g, b]), _SRGB, _ACES2065_1, "CAT02")

        assert_allclose(result, expected, atol=TOLERANCE_ABSOLUTE_TESTS)

    @pytest.mark.parametrize(
        ("cat", "expected"),
        [
            pytest.param(
                "CAT02",
                [0.5228935502, 0.4070999535, 0.1447874511],
                id="cat02",
            ),
            pytest.param(
                "Bradford",
                [0.5226565227, 0.4068408322, 0.1457190311],
                id="bradford",
            ),
            pytest.param(
                None,
                [0.5154522886, 0.4084110823, 0.1567863317],
                id="no-adaptation",
            ),
        ],
    )
    def test_coloured_swatch_is_distinct_per_adaptation(
        self, cat: str | None, expected: list[float]
    ) -> None:
        """Pin the mixed swatch for CAT02, Bradford and no adaptation."""

        assert_allclose(
            RGB_to_RGB(np.array([0.8, 0.4, 0.1]), _SRGB, _ACES2065_1, cat),
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_adapted_methods_and_no_adaptation_are_distinguishable(self) -> None:
        """Ensure the pinned methods do not collapse to the same result."""

        swatch = np.array([0.8, 0.4, 0.1])
        cat02 = RGB_to_RGB(swatch, _SRGB, _ACES2065_1, "CAT02")
        bradford = RGB_to_RGB(swatch, _SRGB, _ACES2065_1, "Bradford")
        no_cat = RGB_to_RGB(swatch, _SRGB, _ACES2065_1, None)

        assert not np.allclose(cat02, bradford, atol=1e-6)
        assert not np.allclose(cat02, no_cat, atol=1e-6)
        assert not np.allclose(bradford, no_cat, atol=1e-6)


M_SYNTHETIC_XYZ_SCALING = np.array(
    [
        [0.6317556764, 0.2104253278, 0.1578189958],
        [0.1000000000, 0.7000000000, 0.2000000000],
        [0.0459112476, 0.1377337427, 0.8163550098],
    ]
)

M_SYNTHETIC_NO_CAT = np.array(
    [
        [0.6226886024, 0.2074052647, 0.1555539485],
        [0.1000000000, 0.7000000000, 0.2000000000],
        [0.0605983773, 0.1817951318, 1.0775091401],
    ]
)

ENCODED_SYNTHETIC_INPUT = np.array([0.5, 0.8, 0.7])


class TestDeclaredEncodingStates:
    """
    Test independently declared input / output encoding states.
    """

    @pytest.mark.parametrize(
        ("decode", "encode", "expected"),
        [
            pytest.param(
                False,
                False,
                [0.5946913975, 0.7500000000, 0.7045911248],
                id="linear-in-linear-out",
            ),
            pytest.param(
                True,
                False,
                [0.3699424368, 0.5710000000, 0.4996413620],
                id="encoded-in-working-linear-out",
            ),
            pytest.param(
                False,
                True,
                [0.7711623678, 0.8660254038, 0.8393992642],
                id="linear-in-encoded-out",
            ),
            pytest.param(
                True,
                True,
                [0.6082289346, 0.7556454195, 0.7068531403],
                id="encoded-in-encoded-out",
            ),
        ],
    )
    def test_xyz_scaling_encoding_combinations(
        self, decode: bool, encode: bool, expected: list[float]
    ) -> None:
        """
        Derive every encoding combination from the literal matrix and the
        exact ``square`` / ``square-root`` transfer pair.
        """

        source, target, _ = _synthetic_colourspaces()
        c = ENCODED_SYNTHETIC_INPUT

        assert_allclose(
            matrix_RGB_to_RGB(source, target, "XYZ Scaling"),
            M_SYNTHETIC_XYZ_SCALING,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Independent derivation of the declared encoding states.
        linear_input = c**2 if decode else c
        working_linear = M_SYNTHETIC_XYZ_SCALING @ linear_input
        independently_derived = np.sqrt(working_linear) if encode else working_linear

        assert_allclose(
            independently_derived,
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        assert_allclose(
            RGB_to_RGB(
                c,
                source,
                target,
                "XYZ Scaling",
                apply_cctf_decoding=decode,
                apply_cctf_encoding=encode,
            ),
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @pytest.mark.parametrize(
        ("decode", "encode", "expected"),
        [
            pytest.param(
                False,
                False,
                [0.5861562769, 0.7500000000, 0.9299916922],
                id="linear-in-linear-out",
            ),
            pytest.param(
                True,
                False,
                [0.3646329548, 0.5710000000, 0.6594779574],
                id="encoded-in-working-linear-out",
            ),
            pytest.param(
                False,
                True,
                [0.7656084358, 0.8660254038, 0.9643607687],
                id="linear-in-encoded-out",
            ),
            pytest.param(
                True,
                True,
                [0.6038484535, 0.7556454195, 0.8120824819],
                id="encoded-in-encoded-out",
            ),
        ],
    )
    def test_no_adaptation_encoding_combinations(
        self, decode: bool, encode: bool, expected: list[float]
    ) -> None:
        """Derive the same four states with white point adaptation disabled."""

        source, target, _ = _synthetic_colourspaces()
        c = ENCODED_SYNTHETIC_INPUT

        assert_allclose(
            matrix_RGB_to_RGB(source, target, None),
            M_SYNTHETIC_NO_CAT,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        linear_input = c**2 if decode else c
        working_linear = M_SYNTHETIC_NO_CAT @ linear_input
        independently_derived = np.sqrt(working_linear) if encode else working_linear

        assert_allclose(
            RGB_to_RGB(
                c,
                source,
                target,
                None,
                apply_cctf_decoding=decode,
                apply_cctf_encoding=encode,
            ),
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        assert_allclose(
            independently_derived,
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @pytest.mark.parametrize("decode", [False, True])
    @pytest.mark.parametrize("encode", [False, True])
    def test_adapted_and_unadapted_results_differ_in_every_state(
        self, decode: bool, encode: bool
    ) -> None:
        """
        For every pair of declared encoding states, enabling the XYZ Scaling
        adaptation must be distinguishable from disabling adaptation.
        """

        source, target, _ = _synthetic_colourspaces()
        c = ENCODED_SYNTHETIC_INPUT

        common_arguments = {
            "apply_cctf_decoding": decode,
            "apply_cctf_encoding": encode,
        }
        adapted = RGB_to_RGB(c, source, target, "XYZ Scaling", **common_arguments)
        unadapted = RGB_to_RGB(c, source, target, None, **common_arguments)

        assert not np.allclose(adapted, unadapted, atol=1e-6)

    def test_linearised_and_transfer_encoded_grey_differ(self) -> None:
        """
        Compare already linearised footage with footage still carrying the
        *sRGB* transfer curve at the same nominal 0.5 code value.
        """

        code_value = np.full(3, 0.5)

        # Independently written *sRGB* EOTF for the 0.5 > 0.04045 branch.
        linear_value = ((0.5 + 0.055) / 1.055) ** 2.4

        assert_allclose(linear_value, 0.2140411405, atol=1e-10)

        expected_linear_input = [
            0.5000238918,
            0.4999865074,
            0.4999734426,
        ]
        expected_decoded_input = M_SRGB_TO_ACES2065_1_CAT02 @ np.full(3, linear_value)

        assert_allclose(
            expected_decoded_input,
            [0.214051368124, 0.214035364518, 0.214029771725],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        linearised = RGB_to_RGB(code_value, _SRGB, _ACES2065_1, "CAT02")
        transfer_encoded = RGB_to_RGB(
            code_value,
            _SRGB,
            _ACES2065_1,
            "CAT02",
            apply_cctf_decoding=True,
        )

        assert_allclose(
            linearised,
            expected_linear_input,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        assert_allclose(
            transfer_encoded,
            expected_decoded_input,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        assert not np.allclose(linearised, transfer_encoded, atol=1e-3)

    def test_output_encoding_state_applies_srgb_oetf(self) -> None:
        """
        Check the output-side declared encoding state against an explicit
        *sRGB* OETF applied to independently pinned working-linear values.
        """

        # ACEScg (AP1, D60) to sRGB (BT.709 primaries, D65) CAT02 matrix.
        m_acescg_to_srgb = np.array(
            [
                [1.7048873310, -0.6241572745, -0.0808867739],
                [-0.1295209353, 1.1383993260, -0.0087792418],
                [-0.0241270599, -0.1246206123, 1.1488221099],
            ]
        )
        acescg_linear = np.array([0.4, 0.2, 0.1])
        working_linear = m_acescg_to_srgb @ acescg_linear

        assert_allclose(
            matrix_RGB_to_RGB(_ACESCG, _SRGB, "CAT02"),
            m_acescg_to_srgb,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        assert_allclose(
            working_linear,
            [0.5490348001, 0.1749935669, 0.0803072646],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Explicit piecewise *sRGB* OETF; all channels are above the linear
        # segment threshold here.
        encoded_expected = 1.055 * np.power(working_linear, 1.0 / 2.4) - 0.055

        assert_allclose(
            encoded_expected,
            [0.7667740190, 0.4553228263, 0.3138929087],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        raw = RGB_to_RGB(acescg_linear, _ACESCG, _SRGB, "CAT02")
        encoded = RGB_to_RGB(
            acescg_linear,
            _ACESCG,
            _SRGB,
            "CAT02",
            apply_cctf_encoding=True,
        )

        assert_allclose(raw, working_linear, atol=TOLERANCE_ABSOLUTE_TESTS)
        assert_allclose(encoded, encoded_expected, atol=TOLERANCE_ABSOLUTE_TESTS)
        assert not np.allclose(raw, encoded, atol=1e-3)
