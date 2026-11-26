"""
Define integration tests for the :mod:`colour.io.luts.resolve_cube` module.

The tests model a *Resolve* *.cube* file carrying a 1D shaper and a 3D colour
cube together, as a grading house would hand both stages over to a downstream
facility. Two properties are exercised:

-   Each stage acts on the lens colour with its own declared input range, and
    the graded channel values follow the *Resolve* format conventions.
-   Both directions of the exchange - importing an externally authored file
    and writing a sequence built in the project - are judged on the colour
    mapping actually encoded by the file, using an independent reader of the
    file bytes. A reader and a writer sharing the same misreading round-trip
    each other perfectly while encoding a different transformation than
    *DaVinci Resolve* applies, hence equality of the read and written objects
    is deliberately not treated as evidence.
"""

from __future__ import annotations

import dataclasses
import typing
from itertools import product
from pathlib import Path

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.hints import cast

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, Literal, NDArrayFloat

from colour.io import (
    LUT3D,
    LUT3x1D,
    LUTSequence,
    read_LUT,
    read_LUT_ResolveCube,
    write_LUT,
    write_LUT_ResolveCube,
)
from colour.utilities import xp_assert_close, xp_assert_equal

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_LUTS",
    "EXCHANGE_CUBE",
    "THIRD_PARTY_CUBE",
    "ResolveCubeDocument",
    "DEFAULT_INPUT_RANGE",
    "EXCHANGE_PROBES",
    "EXCHANGE_SHAPER_OUTPUT",
    "EXCHANGE_MAPPING",
    "THIRD_PARTY_PROBES",
    "THIRD_PARTY_MAPPING",
    "PROJECT_SHAPER_TABLE",
    "PROJECT_SHAPER_DOMAIN",
    "PROJECT_CUBE_TABLE",
    "PROJECT_CUBE_DOMAIN",
    "PROJECT_PROBES",
    "PROJECT_SHAPER_OUTPUT",
    "PROJECT_MAPPING",
    "UNITY_SHAPER_PROBES",
    "UNITY_SHAPER_MAPPING",
    "UNITY_CUBE_MAPPING",
    "TOLERANCE_DISCRIMINATION",
    "TestResolveCubeExchangeDecoding",
    "TestResolveCubeExchangeEncoding",
    "TestResolveCubeExchangeSelfConsistency",
]

ROOT_LUTS: Path = Path(__file__).parent / "resources" / "resolve_cube"

EXCHANGE_CUBE: Path = ROOT_LUTS / "Camera_Log_To_Grade_Exchange.cube"
"""Externally authored file carrying a shaper and a cube with distinct ranges."""

THIRD_PARTY_CUBE: Path = ROOT_LUTS / "Three_Dimensional_Table_With_Shaper.cube"
"""Third-party authored file used as a second decoding reference."""

DEFAULT_INPUT_RANGE: tuple[float, float] = (0.0, 1.0)
"""Input range assumed by the format when a range line is absent."""

TOLERANCE_DISCRIMINATION: float = 1e-2
"""
Minimum mapping deviation considered meaningful when discriminating between
two candidate interpretations of a file. Every alternative interpretation
checked below is wrong by at least this much, while the accepted
interpretation matches its reference to floating point precision.
"""


@dataclasses.dataclass(frozen=True)
class ResolveCubeDocument:
    """
    Raw content of a *Resolve* *.cube* file, read independently of the library.

    The reader below is a second, deliberately simple implementation of the
    format: it only understands the declaration lines and the row data, which
    is exactly what a downstream application would consume.

    Parameters
    ----------
    title
        Declared title, empty when absent.
    comments
        Comment lines, in order.
    size_1D
        Declared 1D *LUT* size, *None* when the file has no shaper.
    size_3D
        Declared 3D *LUT* size, *None* when the file has no cube.
    input_range_1D
        Declared 1D input range, *None* when the line is absent.
    input_range_3D
        Declared 3D input range, *None* when the line is absent.
    rows
        Table rows as written, in file order.
    """

    title: str
    comments: tuple[str, ...]
    size_1D: int | None
    size_3D: int | None
    input_range_1D: tuple[float, float] | None
    input_range_3D: tuple[float, float] | None
    rows: NDArrayFloat

    def shaper_table(self) -> NDArrayFloat:
        """
        Return the shaper rows, i.e. the first ``size_1D`` rows.

        Returns
        -------
        :class:`numpy.ndarray`
            Shaper table with ``size_1D`` rows and three columns.
        """

        return self.rows[: self.required_size_1D()]

    def cube_table(self, order: Literal["F", "C"] = "F") -> NDArrayFloat:
        """
        Return the cube rows reshaped to the declared 3D size.

        Parameters
        ----------
        order
            Reshaping order: ``"F"`` implements the *Resolve* convention where
            the red index changes fastest, ``"C"`` implements the alternative
            reading used to model a mis-specified file.

        Returns
        -------
        :class:`numpy.ndarray`
            Cube table with ``size_3D**3`` nodes.
        """

        size = self.required_size_3D()

        return np.reshape(
            self.rows[self.required_size_1D() :], (size,) * 3 + (3,), order=order
        )

    def required_size_1D(self) -> int:
        """
        Return the declared shaper size, asserting that the file has a shaper.

        Returns
        -------
        :class:`int`
            Declared 1D *LUT* size.
        """

        assert self.size_1D is not None, "The document has no 1D shaper!"

        return self.size_1D

    def required_size_3D(self) -> int:
        """
        Return the declared cube size, asserting that the file has a cube.

        Returns
        -------
        :class:`int`
            Declared 3D *LUT* size.
        """

        assert self.size_3D is not None, "The document has no 3D cube!"

        return self.size_3D

    def required_input_range_1D(self) -> NDArrayFloat:
        """
        Return the shaper input range as written, i.e. without the default.

        Returns
        -------
        :class:`numpy.ndarray`
            Lower and upper bounds declared for the shaper.
        """

        assert self.input_range_1D is not None, "No 1D input range is declared!"

        return np.array(self.input_range_1D, dtype=float)

    def required_input_range_3D(self) -> NDArrayFloat:
        """
        Return the cube input range as written, i.e. without the default.

        Returns
        -------
        :class:`numpy.ndarray`
            Lower and upper bounds declared for the cube.
        """

        assert self.input_range_3D is not None, "No 3D input range is declared!"

        return np.array(self.input_range_3D, dtype=float)


def _parse_resolve_cube(path: str | Path) -> ResolveCubeDocument:
    """
    Read a *Resolve* *.cube* file with an implementation independent of the
    library under test.

    Parameters
    ----------
    path
        Path to the file to read.

    Returns
    -------
    :class:`ResolveCubeDocument`
        Parsed document.
    """

    title = ""
    comments: list[str] = []
    size_1D: int | None = None
    size_3D: int | None = None
    input_range_1D: tuple[float, float] | None = None
    input_range_3D: tuple[float, float] | None = None
    rows: list[list[float]] = []

    with open(path) as cube_file:
        for raw_line in cube_file:
            line = raw_line.strip()

            if len(line) == 0:
                continue

            if line.startswith("#"):
                comments.append(line[1:].strip())
                continue

            tokens = line.split()
            keyword = tokens[0]

            if keyword == "TITLE":
                title = line[len("TITLE") :].strip().strip('"')
            elif keyword == "LUT_1D_SIZE":
                size_1D = int(tokens[1])
            elif keyword == "LUT_3D_SIZE":
                size_3D = int(tokens[1])
            elif keyword == "LUT_1D_INPUT_RANGE":
                input_range_1D = (float(tokens[1]), float(tokens[2]))
            elif keyword == "LUT_3D_INPUT_RANGE":
                input_range_3D = (float(tokens[1]), float(tokens[2]))
            else:
                rows.append([float(value) for value in tokens])

    return ResolveCubeDocument(
        title,
        tuple(comments),
        size_1D,
        size_3D,
        input_range_1D,
        input_range_3D,
        np.array(rows),
    )


def _resolve_input_range(
    document: ResolveCubeDocument,
    dimension: Literal[1, 3],
    override: ArrayLike | None = None,
) -> NDArrayFloat:
    """
    Return the input range a stage is expected to act on.

    Parameters
    ----------
    document
        Parsed document.
    dimension
        LUT dimensionality of the stage, i.e. 1 for the shaper and 3 for the
        cube.
    override
        Optional range replacing the declared one, used to model a file whose
        ranges are read as attached to the wrong stage.

    Returns
    -------
    :class:`numpy.ndarray`
        Lower and upper bounds of the stage input range.
    """

    if override is not None:
        return np.asarray(override, dtype=float)

    declared = document.input_range_1D if dimension == 1 else document.input_range_3D

    return np.array(
        declared if declared is not None else DEFAULT_INPUT_RANGE, dtype=float
    )


def _apply_shaper(
    RGB: ArrayLike, table: ArrayLike, input_range: ArrayLike
) -> NDArrayFloat:
    """
    Apply a 1D shaper stage acting on ``input_range``.

    The shaper interpolates each channel independently with a straight line
    between neighbouring samples, the samples being spread evenly over the
    declared input range. Callers restrict their samples to that range so
    that interpolation, and not the out-of-range behaviour, decides the
    result.

    Parameters
    ----------
    RGB
        Lens colour values to grade.
    table
        Shaper rows, one per sample.
    input_range
        Comma separated lower and upper bounds the shaper acts on.

    Returns
    -------
    :class:`numpy.ndarray`
        Graded values, the intermediate handed to the cube stage.
    """

    RGB = np.asarray(RGB, dtype=float)
    table = np.asarray(table, dtype=float)
    lower, upper = np.asarray(input_range, dtype=float)

    samples = np.linspace(lower, upper, table.shape[0])

    return np.stack(
        [
            np.interp(RGB[..., channel], samples, table[:, channel])
            for channel in range(3)
        ],
        axis=-1,
    )


def _apply_cube(
    RGB: ArrayLike, table: ArrayLike, input_range: ArrayLike
) -> NDArrayFloat:
    """
    Apply a 3D cube stage acting on ``input_range``.

    The cube normalises the incoming values over its declared input range,
    clamps them to the unit cell as *Resolve* does for out-of-range values and
    interpolates trilinearly between the eight surrounding nodes.

    Parameters
    ----------
    RGB
        Graded values produced by the shaper stage.
    table
        Cube nodes indexed by red, green and blue index.
    input_range
        Comma separated lower and upper bounds the cube acts on.

    Returns
    -------
    :class:`numpy.ndarray`
        Graded output values, allowed to fall outside the unit interval.
    """

    RGB = np.asarray(RGB, dtype=float)
    table = np.asarray(table, dtype=float)
    lower, upper = np.asarray(input_range, dtype=float)

    shape = RGB.shape
    coordinates = np.clip((RGB - lower) / (upper - lower), 0, 1)
    coordinates = np.reshape(coordinates, (-1, 3))

    nodes = np.array(table.shape[:-1]) - 1
    scaled = coordinates * nodes
    index = np.clip(np.floor(scaled).astype(int), 0, nodes - 1)
    fraction = scaled - index

    output = np.zeros((coordinates.shape[0], 3))
    for offset in product((0, 1), repeat=3):
        weight = np.ones(coordinates.shape[0])
        for axis, delta in enumerate(offset):
            weight *= fraction[:, axis] if delta else 1 - fraction[:, axis]
        output += (
            weight[:, None]
            * table[
                index[:, 0] + offset[0],
                index[:, 1] + offset[1],
                index[:, 2] + offset[2],
            ]
        )

    return np.reshape(output, shape)


def _evaluate_document(
    document: ResolveCubeDocument,
    RGB: ArrayLike,
    *,
    shaper_input_range: ArrayLike | None = None,
    cube_input_range: ArrayLike | None = None,
    cube_row_order: Literal["F", "C"] = "F",
    stage_order: Literal["ShaperCube", "CubeShaper"] = "ShaperCube",
) -> NDArrayFloat:
    """
    Evaluate the colour mapping a file encodes, independently of the library.

    Parameters
    ----------
    document
        Parsed document.
    RGB
        Lens colour values to map.
    shaper_input_range
        Optional override of the shaper input range.
    cube_input_range
        Optional override of the cube input range.
    cube_row_order
        Order used to lay the cube rows out over the node grid.
    stage_order
        Order in which the two stages are chained.

    Returns
    -------
    :class:`numpy.ndarray`
        Mapped colour values.
    """

    shaper_range = _resolve_input_range(document, 1, shaper_input_range)
    cube_range = _resolve_input_range(document, 3, cube_input_range)
    shaper = document.shaper_table()
    cube = document.cube_table(cube_row_order)

    if stage_order == "CubeShaper":
        return _apply_shaper(_apply_cube(RGB, cube, cube_range), shaper, shaper_range)

    return _apply_cube(_apply_shaper(RGB, shaper, shaper_range), cube, cube_range)


def _max_deviation(a: ArrayLike, b: ArrayLike) -> float:
    """
    Return the largest absolute deviation between two arrays.

    Parameters
    ----------
    a
        First array.
    b
        Second array.

    Returns
    -------
    :class:`float`
        Largest absolute deviation.
    """

    return float(
        np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
    )


def _domain(input_range: ArrayLike) -> NDArrayFloat:
    """
    Return the implicit *LUT* domain corresponding to an input range.

    Parameters
    ----------
    input_range
        Lower and upper bounds shared by the three channels.

    Returns
    -------
    :class:`numpy.ndarray`
        Domain with shape (2, 3).
    """

    bounds = np.asarray(input_range, dtype=float)

    return np.tile(bounds[:, None], (1, 3))


EXCHANGE_PROBES: NDArrayFloat = np.array(
    [
        [-0.25, -0.25, -0.25],
        [3.75, 3.75, 3.75],
        [1.75, 1.75, 1.75],
        [0.75, 0.75, 0.75],
        [2.75, 2.75, 2.75],
        [0.25, 1.25, 3.25],
        [3.25, 2.25, 0.25],
        [1.25, 2.25, 2.75],
    ]
)
"""Lens colour values spanning the shaper declared range of the exchange file."""

EXCHANGE_SHAPER_OUTPUT: NDArrayFloat = np.array(
    [
        [0.10, 0.15, 0.10],
        [0.90, 0.85, 0.80],
        [0.50, 0.50, 0.50],
        [0.30, 0.35, 0.25],
        [0.70, 0.65, 0.60],
        [0.20, 0.425, 0.70],
        [0.80, 0.575, 0.175],
        [0.40, 0.575, 0.60],
    ]
)
"""Expected intermediate after the exchange file shaper stage."""

EXCHANGE_MAPPING: NDArrayFloat = np.array(
    [
        [-0.020, 0.040, 0.100],
        [0.930, 0.940, 1.100],
        [0.455, 0.490, 0.600],
        [0.1425, 0.235833333333, 0.200],
        [0.755, 0.706666666667, 0.7875],
        [0.059166666667, 0.4275, 0.827083333333],
        [0.8425, 0.5275, 0.23125],
        [0.3425, 0.594166666667, 0.73125],
    ]
)
"""Expected colour mapping of the exchange file, both stages included."""

THIRD_PARTY_PROBES: NDArrayFloat = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.25, 0.75],
        [3.0, 3.0, 3.0],
    ]
)
"""Lens colour values spanning the declared range of the third-party file."""

THIRD_PARTY_MAPPING: NDArrayFloat = np.array(
    [
        [0.926638000046, 0.901576086360, 0.897331902447],
        [1.165728756514, 0.852926863331, 0.854411851710],
        [1.022938768361, 0.541711455520, 0.727455424626],
        [1.647681600000, 1.647681600000, 1.647681600000],
    ]
)
"""Expected colour mapping of the third-party file, both stages included."""

PROJECT_SHAPER_TABLE: NDArrayFloat = np.array(
    [
        [0.15, 0.20, 0.10],
        [0.55, 0.40, 0.60],
        [0.85, 0.75, 0.90],
    ]
)
"""Shaper authored in the project: a three-sample camera log tone curve."""

PROJECT_SHAPER_DOMAIN: NDArrayFloat = np.array([[-0.5, -0.5, -0.5], [5.0, 5.0, 5.0]])
"""Input range the project shaper is authored for."""

PROJECT_CUBE_TABLE: NDArrayFloat = np.array(
    [
        [
            [[0.020, 0.050, -0.020], [0.020, 0.150, 0.930]],
            [[0.070, 0.900, 0.010], [0.070, 1.000, 0.960]],
        ],
        [
            [[0.920, 0.070, -0.020], [0.920, 0.170, 0.930]],
            [[0.970, 0.920, 0.010], [0.970, 1.020, 0.960]],
        ],
    ]
)
"""Creative grade authored in the project, indexed by red, green and blue."""

PROJECT_CUBE_DOMAIN: NDArrayFloat = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
"""Input range the project cube is authored for, distinct from the shaper one."""

PROJECT_PROBES: NDArrayFloat = np.array(
    [
        [-0.5, -0.5, -0.5],
        [5.0, 5.0, 5.0],
        [2.0, 1.0, 3.5],
        [0.25, 0.75, 1.25],
        [3.6, -0.1, 4.9],
    ]
)
"""Lens colour values spanning the project shaper input range."""

PROJECT_SHAPER_OUTPUT: NDArrayFloat = np.array(
    [
        [0.15, 0.20, 0.10],
        [0.85, 0.75, 0.90],
        [0.513636363636, 0.309090909091, 0.736363636364],
        [0.259090909091, 0.290909090909, 0.418181818182],
        [0.697272727273, 0.229090909091, 0.889090909091],
    ]
)
"""Expected intermediate after the project shaper stage."""

PROJECT_MAPPING: NDArrayFloat = np.array(
    [
        [0.0825, 0.1575, -0.01625],
        [0.904375, 0.859375, 0.954375],
        [0.498409090909, 0.362045454545, 0.743522727273],
        [0.210909090909, 0.296590909091, 0.365],
        [0.700, 0.300727272727, 0.921886363636],
    ]
)
"""Expected colour mapping of the project sequence, both stages included."""

UNITY_SHAPER_PROBES: NDArrayFloat = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.25, 0.5, 0.75],
        [0.6, 0.3, 0.9],
        [0.0625, 0.9375, 0.5],
    ]
)
"""Lens colour values spanning the default shaper input range."""

UNITY_SHAPER_MAPPING: NDArrayFloat = np.array(
    [
        [0.0825, 0.1575, -0.01625],
        [0.904375, 0.859375, 0.954375],
        [0.320, 0.45625, 0.763125],
        [0.6075, 0.389, 0.867],
        [0.170390625, 0.759140625, 0.596484375],
    ]
)
"""Expected mapping when the shaper keeps the default input range."""

UNITY_CUBE_MAPPING: NDArrayFloat = np.array(
    [
        [0.165, 0.233, 0.081],
        [0.8225, 0.7945, 0.8575],
        [0.497727272727, 0.396636363636, 0.688818181818],
        [0.267727272727, 0.344272727273, 0.386],
        [0.659, 0.347581818182, 0.831509090909],
    ]
)
"""Expected mapping when the cube keeps the default input range."""


def _project_sequence() -> LUTSequence:
    """
    Build the shaper and cube pair authored in the project.

    Returns
    -------
    :class:`colour.LUTSequence`
        Sequence handing the shaper and the cube over together.
    """

    return LUTSequence(
        LUT3x1D(
            PROJECT_SHAPER_TABLE,
            "Camera Log Shaper",
            PROJECT_SHAPER_DOMAIN,
            comments=["1D shaper authored by the originating facility."],
        ),
        LUT3D(
            PROJECT_CUBE_TABLE,
            "Creative Grade",
            PROJECT_CUBE_DOMAIN,
            comments=["3D cube authored by the originating facility."],
        ),
    )


def _swap_stage_domains(
    LUT: LUTSequence,
) -> LUTSequence:
    """
    Return a copy of the sequence with the two stages' input ranges exchanged.

    Parameters
    ----------
    LUT
        Sequence to copy and modify.

    Returns
    -------
    :class:`colour.LUTSequence`
        Sequence holding the shaper range on the cube and vice versa.
    """

    swapped = LUT.copy()
    swapped[0].domain, swapped[1].domain = swapped[1].domain, swapped[0].domain

    return swapped


def _reorder_cube_rows(
    LUT: LUTSequence,
) -> LUTSequence:
    """
    Return a copy of the sequence with the cube nodes laid out in ``"C"`` order.

    The nodes are reordered while keeping the same axis meaning, which is the
    step a writer makes when it flattens the cube with the blue index changing
    fastest instead of the red one.

    Parameters
    ----------
    LUT
        Sequence to copy and modify.

    Returns
    -------
    :class:`colour.LUTSequence`
        Sequence holding the cube nodes in the alternative row order.
    """

    reordered = LUT.copy()
    table = reordered[1].table
    reordered[1].table = np.reshape(
        np.reshape(table, (-1, 3), order="F"), table.shape, order="C"
    )

    return reordered


class TestResolveCubeExchangeDecoding:
    """
    Define the tests importing a shaper and cube handover file and judging the
    colour mapping it encodes.
    """

    def test_stage_structure_and_declared_ranges(self) -> None:
        """
        Test that both stages are recovered and that each one keeps the input
        range declared for it in the file.
        """

        document = _parse_resolve_cube(EXCHANGE_CUBE)
        sequence = cast("LUTSequence", read_LUT_ResolveCube(EXCHANGE_CUBE))

        assert document.title == "Camera Log to Grade Exchange"
        assert document.size_1D == 5
        assert document.size_3D == 3
        assert len(document.comments) == 3

        assert isinstance(sequence, LUTSequence)
        assert len(sequence) == 2
        assert isinstance(sequence[0], LUT3x1D)
        assert isinstance(sequence[1], LUT3D)
        assert sequence[0].size == document.required_size_1D()
        assert sequence[1].size == document.required_size_3D()

        xp_assert_equal(sequence[0].domain, _domain(document.required_input_range_1D()))
        xp_assert_equal(sequence[1].domain, _domain(document.required_input_range_3D()))
        assert document.input_range_1D != document.input_range_3D
        assert document.input_range_1D == (-0.25, 3.75)
        assert document.input_range_3D == (0.2, 0.8)

        assert sequence[0].name == "Camera Log to Grade Exchange - Shaper"
        assert sequence[1].name == "Camera Log to Grade Exchange - Cube"
        assert sequence[0].comments == []
        assert sequence[1].comments == list(document.comments)

    def test_declared_tables_match_file_rows(self) -> None:
        """
        Test that the library tables and an independent reader of the same
        bytes agree on every node.
        """

        document = _parse_resolve_cube(EXCHANGE_CUBE)
        sequence = cast("LUTSequence", read_LUT_ResolveCube(EXCHANGE_CUBE))

        xp_assert_equal(sequence[0].table, document.shaper_table())
        xp_assert_equal(sequence[1].table, document.cube_table())

        assert document.rows.shape == (
            document.required_size_1D() + document.required_size_3D() ** 3,
            3,
        )

    def test_shaper_stage_uses_its_own_input_range(self) -> None:
        """
        Test that the shaper acts on its declared range, per channel, and that
        it agrees with an independent evaluation of the same rows.
        """

        document = _parse_resolve_cube(EXCHANGE_CUBE)
        sequence = cast("LUTSequence", read_LUT_ResolveCube(EXCHANGE_CUBE))
        shaper = cast("LUT3x1D", sequence[0])

        expected = _apply_shaper(
            EXCHANGE_PROBES,
            document.shaper_table(),
            document.required_input_range_1D(),
        )

        xp_assert_close(expected, EXCHANGE_SHAPER_OUTPUT, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(
            shaper.apply(EXCHANGE_PROBES),
            EXCHANGE_SHAPER_OUTPUT,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            shaper.apply(EXCHANGE_PROBES), expected, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        mixed = shaper.apply(np.array([[1.75, 0.75, 2.75]]))
        xp_assert_close(
            mixed,
            np.array(
                [
                    [
                        document.shaper_table()[2, 0],
                        document.shaper_table()[1, 1],
                        document.shaper_table()[3, 2],
                    ]
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert (
            _max_deviation(
                _apply_shaper(
                    EXCHANGE_PROBES,
                    document.shaper_table(),
                    document.input_range_3D,
                ),
                expected,
            )
            > TOLERANCE_DISCRIMINATION
        )
        assert (
            _max_deviation(
                _apply_shaper(
                    EXCHANGE_PROBES,
                    document.shaper_table(),
                    DEFAULT_INPUT_RANGE,
                ),
                expected,
            )
            > TOLERANCE_DISCRIMINATION
        )

    def test_cube_stage_uses_its_own_input_range(self) -> None:
        """
        Test that the cube acts on its declared range: the declared range maps
        onto the corner nodes exactly, out-of-range values clamp onto them and
        the node grid is read in *Resolve* row order.
        """

        document = _parse_resolve_cube(EXCHANGE_CUBE)
        sequence = cast("LUTSequence", read_LUT_ResolveCube(EXCHANGE_CUBE))
        cube = cast("LUT3D", sequence[1])
        table = document.cube_table()
        lower, upper = document.required_input_range_3D()
        size = document.required_size_3D()

        for indices in product(range(size), repeat=3):
            position = np.array(indices, dtype=float) / (size - 1)
            coordinates = lower + (upper - lower) * position
            xp_assert_close(
                cube.apply(coordinates),
                table[indices],
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

        xp_assert_close(
            cube.apply(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])),
            np.array([table[0, 0, 0], table[2, 2, 2]]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            cube.apply(np.array([[lower, lower, lower]])),
            table[0, 0, 0][None, ...],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            cube.apply(EXCHANGE_SHAPER_OUTPUT),
            EXCHANGE_MAPPING,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert table[1, 0, 0].tolist() == [0.38, 0.065, 0.15]
        assert table[0, 1, 0].tolist() == [0.03, 0.39, 0.125]
        assert table[0, 0, 1].tolist() == [0.005, 0.115, 0.525]

    def test_combined_mapping(self) -> None:
        """
        Test that chaining both stages reproduces the expected colour mapping,
        and that the auto-detecting reader imports the same mapping.
        """

        sequence = cast("LUTSequence", read_LUT_ResolveCube(EXCHANGE_CUBE))

        xp_assert_close(
            sequence.apply(EXCHANGE_PROBES),
            EXCHANGE_MAPPING,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        document = _parse_resolve_cube(EXCHANGE_CUBE)
        xp_assert_close(
            _evaluate_document(document, EXCHANGE_PROBES),
            EXCHANGE_MAPPING,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        imported = cast("LUTSequence", read_LUT(EXCHANGE_CUBE))
        assert imported == sequence
        xp_assert_close(
            imported.apply(EXCHANGE_PROBES),
            EXCHANGE_MAPPING,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert EXCHANGE_MAPPING.max() > 1 and EXCHANGE_MAPPING.min() < 0

    def test_mapping_discriminates_alternative_interpretations(self) -> None:
        """
        Test that the expected mapping is distinguishable from the mappings of
        plausible misreadings of the same file.
        """

        document = _parse_resolve_cube(EXCHANGE_CUBE)

        alternative_interpretations = {
            "ranges attached to the wrong stage": dict(
                shaper_input_range=document.required_input_range_3D(),
                cube_input_range=document.required_input_range_1D(),
            ),
            "shaper range assumed default": dict(
                shaper_input_range=DEFAULT_INPUT_RANGE
            ),
            "cube range assumed default": dict(cube_input_range=DEFAULT_INPUT_RANGE),
            "cube rows read with blue changing fastest": dict(cube_row_order="C"),
            "stages chained in reverse order": dict(stage_order="CubeShaper"),
        }

        for description, overrides in alternative_interpretations.items():
            mapping = _evaluate_document(document, EXCHANGE_PROBES, **overrides)
            assert (
                _max_deviation(mapping, EXCHANGE_MAPPING) > TOLERANCE_DISCRIMINATION
            ), description

    def test_third_party_file_mapping(self) -> None:
        """
        Test that an independently authored file using the same handover is
        decoded with the same mapping by the library and by the independent
        reader.
        """

        document = _parse_resolve_cube(THIRD_PARTY_CUBE)
        sequence = cast("LUTSequence", read_LUT_ResolveCube(THIRD_PARTY_CUBE))

        assert document.input_range_1D == (-0.1, 3.0)
        assert document.input_range_3D == (-0.1, 3.0)
        assert document.size_1D == 10
        assert document.size_3D == 3

        xp_assert_close(
            _evaluate_document(document, THIRD_PARTY_PROBES),
            THIRD_PARTY_MAPPING,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            sequence.apply(THIRD_PARTY_PROBES),
            THIRD_PARTY_MAPPING,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestResolveCubeExchangeEncoding:
    """
    Define the tests writing a shaper and cube pair composed in the project
    and judging the colour mapping the written file encodes.
    """

    def test_written_declarations(self, tmp_path: Path) -> None:
        """
        Test that the written file declares the title, comments, sizes and the
        input range of each stage.
        """

        sequence = _project_sequence()
        path = tmp_path / "Camera_Log_To_Grade_Exchange.cube"

        assert write_LUT_ResolveCube(sequence, path) is True

        document = _parse_resolve_cube(path)

        assert document.title == "Camera Log Shaper - Creative Grade"
        assert document.comments == (
            "1D shaper authored by the originating facility.",
            "3D cube authored by the originating facility.",
        )
        assert document.size_1D == PROJECT_SHAPER_TABLE.shape[0]
        assert document.size_3D == PROJECT_CUBE_TABLE.shape[0]
        assert document.input_range_1D == (-0.5, 5.0)
        assert document.input_range_3D == (0.1, 0.9)

        text = path.read_text()
        assert "LUT_1D_INPUT_RANGE -0.5000000 5.0000000" in text
        assert "LUT_3D_INPUT_RANGE 0.1000000 0.9000000" in text

        xp_assert_equal(document.shaper_table(), PROJECT_SHAPER_TABLE)
        xp_assert_equal(document.cube_table(), PROJECT_CUBE_TABLE)

    def test_written_channel_values(self, tmp_path: Path) -> None:
        """
        Test that the written table rows follow the *Resolve* conventions: the
        cube is written with the red index changing fastest, and graded values
        are written verbatim without being clamped to the unit interval.
        """

        path = tmp_path / "Camera_Log_To_Grade_Exchange.cube"
        write_LUT_ResolveCube(_project_sequence(), path)

        document = _parse_resolve_cube(path)

        size_1D = document.required_size_1D()
        size_3D = document.required_size_3D()
        assert document.rows.shape[0] == size_1D + size_3D**3

        for indices in product(range(size_3D), repeat=3):
            flat_index = indices[0] + size_3D * indices[1] + size_3D**2 * indices[2]
            xp_assert_equal(
                document.rows[size_1D + flat_index],
                PROJECT_CUBE_TABLE[indices],
            )

        assert document.rows.max() > 1
        assert document.rows.min() < 0
        assert 1.02 in document.rows
        assert -0.02 in document.rows

    def test_written_mapping(self, tmp_path: Path) -> None:
        """
        Test that the written file encodes the colour mapping computed in the
        project, as judged by the independent reader and by reading the file
        back.
        """

        sequence = _project_sequence()

        xp_assert_close(
            sequence.apply(PROJECT_PROBES),
            PROJECT_MAPPING,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            sequence[0].apply(PROJECT_PROBES),
            PROJECT_SHAPER_OUTPUT,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        path = tmp_path / "Camera_Log_To_Grade_Exchange.cube"
        write_LUT_ResolveCube(sequence, path)

        document = _parse_resolve_cube(path)
        xp_assert_close(
            _evaluate_document(document, PROJECT_PROBES),
            PROJECT_MAPPING,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            read_LUT_ResolveCube(path).apply(PROJECT_PROBES),
            PROJECT_MAPPING,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        dispatched = tmp_path / "Camera_Log_To_Grade_Exchange_Dispatched.cube"
        write_LUT(sequence, dispatched)
        xp_assert_close(
            read_LUT(dispatched).apply(PROJECT_PROBES),
            PROJECT_MAPPING,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_written_file_discriminates_alternative_interpretations(
        self, tmp_path: Path
    ) -> None:
        """
        Test that the mapping encoded by the written file is distinguishable
        from the mappings of plausible misreadings of it.
        """

        path = tmp_path / "Camera_Log_To_Grade_Exchange.cube"
        write_LUT_ResolveCube(_project_sequence(), path)

        document = _parse_resolve_cube(path)

        alternative_interpretations = {
            "ranges attached to the wrong stage": dict(
                shaper_input_range=document.required_input_range_3D(),
                cube_input_range=document.required_input_range_1D(),
            ),
            "shaper range assumed default": dict(
                shaper_input_range=DEFAULT_INPUT_RANGE
            ),
            "cube range assumed default": dict(cube_input_range=DEFAULT_INPUT_RANGE),
            "cube rows read with blue changing fastest": dict(cube_row_order="C"),
            "stages chained in reverse order": dict(stage_order="CubeShaper"),
        }

        for description, overrides in alternative_interpretations.items():
            mapping = _evaluate_document(document, PROJECT_PROBES, **overrides)
            assert (
                _max_deviation(mapping, PROJECT_MAPPING) > TOLERANCE_DISCRIMINATION
            ), description

    def test_round_trip_preserves_structure_and_mapping(self, tmp_path: Path) -> None:
        """
        Test that reading the written file back recovers both stages with
        their own ranges and preserves the colour mapping.
        """

        sequence = _project_sequence()
        path = tmp_path / "Camera_Log_To_Grade_Exchange.cube"
        write_LUT_ResolveCube(sequence, path)

        recovered = read_LUT_ResolveCube(path)

        assert isinstance(recovered, LUTSequence)
        assert recovered == sequence
        assert isinstance(recovered[0], LUT3x1D)
        assert isinstance(recovered[1], LUT3D)
        xp_assert_equal(recovered[0].domain, PROJECT_SHAPER_DOMAIN)
        xp_assert_equal(recovered[1].domain, PROJECT_CUBE_DOMAIN)
        xp_assert_equal(recovered[0].table, PROJECT_SHAPER_TABLE)
        xp_assert_equal(recovered[1].table, PROJECT_CUBE_TABLE)

        xp_assert_close(
            recovered.apply(PROJECT_PROBES),
            PROJECT_MAPPING,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_default_ranges_are_omitted_and_interpreted_as_unit_interval(
        self, tmp_path: Path
    ) -> None:
        """
        Test that a stage keeping the default input range omits its range
        declaration and is read back as acting on the unit interval, not on
        the range of the other stage.
        """

        unity_shaper = LUTSequence(
            LUT3x1D(PROJECT_SHAPER_TABLE, "Unity Shaper"),
            LUT3D(PROJECT_CUBE_TABLE, "Creative Grade", PROJECT_CUBE_DOMAIN),
        )
        shaper_path = tmp_path / "Unity_Shaper.cube"
        write_LUT_ResolveCube(unity_shaper, shaper_path)

        shaper_document = _parse_resolve_cube(shaper_path)
        assert shaper_document.input_range_1D is None
        assert shaper_document.input_range_3D == (0.1, 0.9)
        assert "LUT_1D_INPUT_RANGE" not in shaper_path.read_text()

        xp_assert_close(
            _evaluate_document(shaper_document, UNITY_SHAPER_PROBES),
            UNITY_SHAPER_MAPPING,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            read_LUT_ResolveCube(shaper_path).apply(UNITY_SHAPER_PROBES),
            UNITY_SHAPER_MAPPING,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        assert (
            _max_deviation(
                _evaluate_document(
                    shaper_document,
                    UNITY_SHAPER_PROBES,
                    shaper_input_range=shaper_document.required_input_range_3D(),
                ),
                UNITY_SHAPER_MAPPING,
            )
            > TOLERANCE_DISCRIMINATION
        )

        unity_cube = LUTSequence(
            LUT3x1D(PROJECT_SHAPER_TABLE, "Camera Log Shaper", PROJECT_SHAPER_DOMAIN),
            LUT3D(PROJECT_CUBE_TABLE, "Creative Grade"),
        )
        cube_path = tmp_path / "Unity_Cube.cube"
        write_LUT_ResolveCube(unity_cube, cube_path)

        cube_document = _parse_resolve_cube(cube_path)
        assert cube_document.input_range_1D == (-0.5, 5.0)
        assert cube_document.input_range_3D is None
        assert "LUT_3D_INPUT_RANGE" not in cube_path.read_text()

        xp_assert_close(
            _evaluate_document(cube_document, PROJECT_PROBES),
            UNITY_CUBE_MAPPING,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            read_LUT_ResolveCube(cube_path).apply(PROJECT_PROBES),
            UNITY_CUBE_MAPPING,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        assert (
            _max_deviation(
                _evaluate_document(
                    cube_document,
                    PROJECT_PROBES,
                    cube_input_range=cube_document.required_input_range_1D(),
                ),
                UNITY_CUBE_MAPPING,
            )
            > TOLERANCE_DISCRIMINATION
        )


class TestResolveCubeExchangeSelfConsistency:
    """
    Define the tests ensuring that a reader and a writer agreeing only with
    each other are still caught by judging the colour mapping.
    """

    def test_range_swap_pair_is_self_consistent_but_wrong(self, tmp_path: Path) -> None:
        """
        Test that a writer attaching each range to the wrong stage and a
        reader undoing the swap round-trip each other perfectly, while the
        file as *Resolve* reads it maps the lens colour differently.
        """

        sequence = _project_sequence()
        path = tmp_path / "Swapped_Ranges.cube"
        write_LUT_ResolveCube(_swap_stage_domains(sequence), path)

        recovered = _swap_stage_domains(read_LUT_ResolveCube(path))
        assert recovered == sequence
        xp_assert_close(
            recovered.apply(PROJECT_PROBES),
            PROJECT_MAPPING,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        document = _parse_resolve_cube(path)
        assert document.input_range_1D == (0.1, 0.9)
        assert document.input_range_3D == (-0.5, 5.0)

        conforming = read_LUT_ResolveCube(path).apply(PROJECT_PROBES)
        assert _max_deviation(conforming, PROJECT_MAPPING) > TOLERANCE_DISCRIMINATION

        clamped = _evaluate_document(document, PROJECT_PROBES)
        assert _max_deviation(clamped, PROJECT_MAPPING) > TOLERANCE_DISCRIMINATION

    def test_row_order_pair_is_self_consistent_but_wrong(self, tmp_path: Path) -> None:
        """
        Test that a writer laying the cube rows out with the blue index
        changing fastest and a reader undoing the reordering round-trip each
        other perfectly, while the file as *Resolve* reads it maps the lens
        colour differently.
        """

        sequence = _project_sequence()
        path = tmp_path / "Reordered_Cube.cube"
        write_LUT_ResolveCube(_reorder_cube_rows(sequence), path)

        recovered = _reorder_cube_rows(read_LUT_ResolveCube(path))
        assert recovered == sequence
        xp_assert_close(
            recovered.apply(PROJECT_PROBES),
            PROJECT_MAPPING,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        document = _parse_resolve_cube(path)
        assert document.input_range_1D == (-0.5, 5.0)
        assert document.input_range_3D == (0.1, 0.9)

        conforming = read_LUT_ResolveCube(path).apply(PROJECT_PROBES)
        assert _max_deviation(conforming, PROJECT_MAPPING) > TOLERANCE_DISCRIMINATION

        assert (
            _max_deviation(
                _evaluate_document(document, PROJECT_PROBES),
                PROJECT_MAPPING,
            )
            > TOLERANCE_DISCRIMINATION
        )
