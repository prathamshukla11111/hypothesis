# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tools for understanding predicates, to satisfy them by construction.

For example::

    integers().filter(lambda x: x >= 0) -> integers(min_value=0)

This is intractable in general, but reasonably easy for simple cases involving
numeric bounds, strings with length or regex constraints, and collection lengths -
and those are precisely the most common cases.  When they arise in e.g. Pandas
dataframes, it's also pretty painful to do the constructive version by hand in
a library; so we prefer to share all the implementation effort here.
See https://github.com/HypothesisWorks/hypothesis/issues/2701 for details.
"""

import ast
import inspect
import math
import operator
import sys
from collections.abc import Callable, Collection
from decimal import Decimal
from fractions import Fraction
from functools import partial
from typing import Any, NamedTuple, TypeVar

from hypothesis.internal.compat import ceil, floor
from hypothesis.internal.floats import next_down, next_up
from hypothesis.internal.lambda_sources import lambda_description
from hypothesis.internal.reflection import get_pretty_function_description

if sys.version_info[:2] >= (3, 14):
    from functools import Placeholder
else:  # pragma: no cover
    Placeholder = object()

Ex = TypeVar("Ex")
Predicate = Callable[[Ex], bool]


class ConstructivePredicate(NamedTuple):
    """Return constraints to the appropriate strategy, and the predicate if needed.

    For example::

        integers().filter(lambda x: x >= 0)
        -> {"min_value": 0"}, None

        integers().filter(lambda x: x >= 0 and x % 7)
        -> {"min_value": 0}, lambda x: x % 7

    At least in principle - for now we usually return the predicate unchanged
    if needed.

    We have a separate get-predicate frontend for each "group" of strategies; e.g.
    for each numeric type, for strings, for bytes, for collection sizes, etc.
    """

    constraints: dict[str, Any]
    predicate: Predicate | None

    @classmethod
    def unchanged(cls, predicate: Predicate) -> "ConstructivePredicate":
        pass

    def __repr__(self) -> str:
        fn = get_pretty_function_description(self.predicate)
        return f"{self.__class__.__name__}(constraints={self.constraints!r}, predicate={fn})"


ARG = object()


def convert(node: ast.AST, argname: str) -> object:
    if isinstance(node, ast.Name):
        if node.id != argname:
            raise ValueError("Non-local variable")
        return ARG
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
        ):
            # error unless comparison is to the len *of the lambda arg*
            return convert(node.args[0], argname)
    return ast.literal_eval(node)


def comp_to_constraints(x: ast.AST, op: ast.AST, y: ast.AST, *, argname: str) -> dict:
    pass


def merge_preds(*con_predicates: ConstructivePredicate) -> ConstructivePredicate:
    # This function is just kinda messy.  Unfortunately the neatest way
    # to do this is just to roll out each case and handle them in turn.
    pass


def numeric_bounds_from_ast(
    tree: ast.AST, argname: str, fallback: ConstructivePredicate
) -> ConstructivePredicate:
    """Take an AST; return a ConstructivePredicate.

    >>> lambda x: x >= 0
    {"min_value": 0}, None
    >>> lambda x: x < 10
    {"max_value": 10, "exclude_max": True}, None
    >>> lambda x: len(x) >= 5
    {"min_value": 5, "len": True}, None
    >>> lambda x: x >= y
    {}, lambda x: x >= y

    See also https://greentreesnakes.readthedocs.io/en/latest/
    """
    pass


def get_numeric_predicate_bounds(predicate: Predicate) -> ConstructivePredicate:
    """Shared logic for understanding numeric bounds.

    We then specialise this in the other functions below, to ensure that e.g.
    all the values are representable in the types that we're planning to generate
    so that the strategy validation doesn't complain.
    """
    pass


def get_integer_predicate_bounds(predicate: Predicate) -> ConstructivePredicate:
    pass


def get_float_predicate_bounds(predicate: Predicate) -> ConstructivePredicate:
    pass


def max_len(size: int, element: Collection[object]) -> bool:
    pass


def min_len(size: int, element: Collection[object]) -> bool:
    pass
