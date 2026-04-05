# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from collections import OrderedDict, abc
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Generic, Union

import numpy as np
import pandas

from hypothesis import strategies as st
from hypothesis.control import reject
from hypothesis.errors import InvalidArgument
from hypothesis.extra import numpy as npst
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.coverage import check, check_function
from hypothesis.internal.validation import (
    check_type,
    check_valid_interval,
    check_valid_size,
    try_convert,
)
from hypothesis.strategies._internal.strategies import Ex, check_strategy
from hypothesis.strategies._internal.utils import cacheable, defines_strategy
from hypothesis.utils.deprecation import note_deprecation

try:
    from pandas.core.arrays.integer import IntegerDtype
except ImportError:
    IntegerDtype = ()


def dtype_for_elements_strategy(s):
    return st.shared(
        s.map(lambda x: pandas.Series([x]).dtype),
        key=("hypothesis.extra.pandas.dtype_for_elements_strategy", s),
    )


def infer_dtype_if_necessary(dtype, values, elements, draw):
    if dtype is None and not values:
        return draw(dtype_for_elements_strategy(elements))
    return dtype


@check_function
def elements_and_dtype(elements, dtype, source=None):
    pass


class ValueIndexStrategy(st.SearchStrategy):
    def __init__(self, elements, dtype, min_size, max_size, unique, name):
        super().__init__()
        self.elements = elements
        self.dtype = dtype
        self.min_size = min_size
        self.max_size = max_size
        self.unique = unique
        self.name = name

    def do_draw(self, data):
        result = []
        seen = set()

        iterator = cu.many(
            data,
            min_size=self.min_size,
            max_size=self.max_size,
            average_size=(self.min_size + self.max_size) / 2,
        )

        while iterator.more():
            elt = data.draw(self.elements)

            if self.unique:
                if elt in seen:
                    iterator.reject()
                    continue
                seen.add(elt)
            result.append(elt)

        dtype = infer_dtype_if_necessary(
            dtype=self.dtype, values=result, elements=self.elements, draw=data.draw
        )
        return pandas.Index(
            result, dtype=dtype, tupleize_cols=False, name=data.draw(self.name)
        )


DEFAULT_MAX_SIZE = 10


@cacheable
@defines_strategy()
def range_indexes(
    min_size: int = 0,
    max_size: int | None = None,
    name: st.SearchStrategy[str | None] = st.none(),
) -> st.SearchStrategy[pandas.RangeIndex]:
    """Provides a strategy which generates an :class:`~pandas.Index` whose
    values are 0, 1, ..., n for some n.

    Arguments:

    * min_size is the smallest number of elements the index can have.
    * max_size is the largest number of elements the index can have. If None
      it will default to some suitable value based on min_size.
    * name is the name of the index. If st.none(), the index will have no name.
    """
    pass


@cacheable
@defines_strategy()
def indexes(
    *,
    elements: st.SearchStrategy[Ex] | None = None,
    dtype: Any = None,
    min_size: int = 0,
    max_size: int | None = None,
    unique: bool = True,
    name: st.SearchStrategy[str | None] = st.none(),
) -> st.SearchStrategy[pandas.Index]:
    """Provides a strategy for producing a :class:`pandas.Index`.

    Arguments:

    * elements is a strategy which will be used to generate the individual
      values of the index. If None, it will be inferred from the dtype. Note:
      even if the elements strategy produces tuples, the generated value
      will not be a MultiIndex, but instead be a normal index whose elements
      are tuples.
    * dtype is the dtype of the resulting index. If None, it will be inferred
      from the elements strategy. At least one of dtype or elements must be
      provided.
    * min_size is the minimum number of elements in the index.
    * max_size is the maximum number of elements in the index. If None then it
      will default to a suitable small size. If you want larger indexes you
      should pass a max_size explicitly.
    * unique specifies whether all of the elements in the resulting index
      should be distinct.
    * name is a strategy for strings or ``None``, which will be passed to
      the :class:`pandas.Index` constructor.
    """
    pass


@defines_strategy()
def series(
    *,
    elements: st.SearchStrategy[Ex] | None = None,
    dtype: Any = None,
    # new-style unions hit https://github.com/sphinx-doc/sphinx/issues/11211 during
    # doc builds. See related comment in django/_fields.py. Quote to prevent
    # shed/pyupgrade from changing it.
    index: (
        st.SearchStrategy["Union[Sequence, pandas.Index]"] | None  # noqa: UP007
    ) = None,
    fill: st.SearchStrategy[Ex] | None = None,
    unique: bool = False,
    name: st.SearchStrategy[str | None] = st.none(),
) -> st.SearchStrategy[pandas.Series]:
    """Provides a strategy for producing a :class:`pandas.Series`.

    Arguments:

    * elements: a strategy that will be used to generate the individual
      values in the series. If None, we will attempt to infer a suitable
      default from the dtype.

    * dtype: the dtype of the resulting series and may be any value
      that can be passed to :class:`numpy.dtype`. If None, will use
      pandas's standard behaviour to infer it from the type of the elements
      values. Note that if the type of values that comes out of your
      elements strategy varies, then so will the resulting dtype of the
      series.

    * index: If not None, a strategy for generating indexes for the
      resulting Series. This can generate either :class:`pandas.Index`
      objects or any sequence of values (which will be passed to the
      Index constructor).

      You will probably find it most convenient to use the
      :func:`~hypothesis.extra.pandas.indexes` or
      :func:`~hypothesis.extra.pandas.range_indexes` function to produce
      values for this argument.

    * name: is a strategy for strings or ``None``, which will be passed to
      the :class:`pandas.Series` constructor.

    Usage:

    .. code-block:: pycon

        >>> series(dtype=int).example()
        0   -2001747478
        1    1153062837
    """
    pass


@dataclass(slots=True, frozen=False)
class column(Generic[Ex]):
    """Data object for describing a column in a DataFrame.

    Arguments:

    * name: the column name, or None to default to the column position. Must
      be hashable, but can otherwise be any value supported as a pandas column
      name.
    * elements: the strategy for generating values in this column, or None
      to infer it from the dtype.
    * dtype: the dtype of the column, or None to infer it from the element
      strategy. At least one of dtype or elements must be provided.
    * fill: A default value for elements of the column. See
      :func:`~hypothesis.extra.numpy.arrays` for a full explanation.
    * unique: If all values in this column should be distinct.
    """

    name: str | int | None = None
    elements: st.SearchStrategy[Ex] | None = None
    dtype: Any = None
    fill: st.SearchStrategy[Ex] | None = None
    unique: bool = False


def columns(
    names_or_number: int | Sequence[str],
    *,
    dtype: Any = None,
    elements: st.SearchStrategy[Ex] | None = None,
    fill: st.SearchStrategy[Ex] | None = None,
    unique: bool = False,
) -> list[column[Ex]]:
    """A convenience function for producing a list of :class:`column` objects
    of the same general shape.

    The names_or_number argument is either a sequence of values, the
    elements of which will be used as the name for individual column
    objects, or a number, in which case that many unnamed columns will
    be created. All other arguments are passed through verbatim to
    create the columns.
    """
    pass


@defines_strategy()
def data_frames(
    columns: Sequence[column] | None = None,
    *,
    rows: st.SearchStrategy[dict | Sequence[Any]] | None = None,
    index: st.SearchStrategy[Ex] | None = None,
) -> st.SearchStrategy[pandas.DataFrame]:
    """Provides a strategy for producing a :class:`pandas.DataFrame`.

    Arguments:

    * columns: An iterable of :class:`column` objects describing the shape
      of the generated DataFrame.

    * rows: A strategy for generating a row object. Should generate
      either dicts mapping column names to values or a sequence mapping
      column position to the value in that position (note that unlike the
      :class:`pandas.DataFrame` constructor, single values are not allowed
      here. Passing e.g. an integer is an error, even if there is only one
      column).

      At least one of rows and columns must be provided. If both are
      provided then the generated rows will be validated against the
      columns and an error will be raised if they don't match.

      Caveats on using rows:

      * In general you should prefer using columns to rows, and only use
        rows if the columns interface is insufficiently flexible to
        describe what you need - you will get better performance and
        example quality that way.
      * If you provide rows and not columns, then the shape and dtype of
        the resulting DataFrame may vary. e.g. if you have a mix of int
        and float in the values for one column in your row entries, the
        column will sometimes have an integral dtype and sometimes a float.

    * index: If not None, a strategy for generating indexes for the
      resulting DataFrame. This can generate either :class:`pandas.Index`
      objects or any sequence of values (which will be passed to the
      Index constructor).

      You will probably find it most convenient to use the
      :func:`~hypothesis.extra.pandas.indexes` or
      :func:`~hypothesis.extra.pandas.range_indexes` function to produce
      values for this argument.

    Usage:

    The expected usage pattern is that you use :class:`column` and
    :func:`columns` to specify a fixed shape of the DataFrame you want as
    follows. For example the following gives a two column data frame:

    .. code-block:: pycon

        >>> from hypothesis.extra.pandas import column, data_frames
        >>> data_frames([
        ... column('A', dtype=int), column('B', dtype=float)]).example()
                    A              B
        0  2021915903  1.793898e+232
        1  1146643993            inf
        2 -2096165693   1.000000e+07

    If you want the values in different columns to interact in some way you
    can use the rows argument. For example the following gives a two column
    DataFrame where the value in the first column is always at most the value
    in the second:

    .. code-block:: pycon

        >>> from hypothesis.extra.pandas import column, data_frames
        >>> import hypothesis.strategies as st
        >>> data_frames(
        ...     rows=st.tuples(st.floats(allow_nan=False),
        ...                    st.floats(allow_nan=False)).map(sorted)
        ... ).example()
                       0             1
        0  -3.402823e+38  9.007199e+15
        1 -1.562796e-298  5.000000e-01

    You can also combine the two:

    .. code-block:: pycon

        >>> from hypothesis.extra.pandas import columns, data_frames
        >>> import hypothesis.strategies as st
        >>> data_frames(
        ...     columns=columns(["lo", "hi"], dtype=float),
        ...     rows=st.tuples(st.floats(allow_nan=False),
        ...                    st.floats(allow_nan=False)).map(sorted)
        ... ).example()
                 lo            hi
        0   9.314723e-49  4.353037e+45
        1  -9.999900e-01  1.000000e+07
        2 -2.152861e+134 -1.069317e-73

    (Note that the column dtype must still be specified and will not be
    inferred from the rows. This restriction may be lifted in future).

    Combining rows and columns has the following behaviour:

    * The column names and dtypes will be used.
    * If the column is required to be unique, this will be enforced.
    * Any values missing from the generated rows will be provided using the
      column's fill.
    * Any values in the row not present in the column specification (if
      dicts are passed, if there are keys with no corresponding column name,
      if sequences are passed if there are too many items) will result in
      InvalidArgument being raised.
    """
    pass
