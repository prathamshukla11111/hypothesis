# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import ast
import hashlib
import inspect
import math
import sys
from ast import Constant, Expr, NodeVisitor, UnaryOp, USub
from collections.abc import Iterator, MutableSet
from functools import lru_cache
from itertools import chain
from pathlib import Path
from types import ModuleType
from typing import TypeAlias

import hypothesis
from hypothesis.configuration import storage_directory
from hypothesis.internal.conjecture.choice import ChoiceTypeT
from hypothesis.internal.escalation import is_hypothesis_file

ConstantT: TypeAlias = int | float | bytes | str

# unfortunate collision with builtin. I don't want to name the init arg bytes_.
bytesT = bytes


class Constants:
    def __init__(
        self,
        *,
        integers: MutableSet[int] | None = None,
        floats: MutableSet[float] | None = None,
        bytes: MutableSet[bytes] | None = None,
        strings: MutableSet[str] | None = None,
    ):
        self.integers: MutableSet[int] = set() if integers is None else integers
        self.floats: MutableSet[float] = set() if floats is None else floats
        self.bytes: MutableSet[bytesT] = set() if bytes is None else bytes
        self.strings: MutableSet[str] = set() if strings is None else strings

    def set_for_type(
        self, constant_type: type[ConstantT] | ChoiceTypeT
    ) -> MutableSet[int] | MutableSet[float] | MutableSet[bytes] | MutableSet[str]:
        if constant_type is int or constant_type == "integer":
            return self.integers
        elif constant_type is float or constant_type == "float":
            return self.floats
        elif constant_type is bytes or constant_type == "bytes":
            return self.bytes
        elif constant_type is str or constant_type == "string":
            return self.strings
        raise ValueError(f"unknown constant_type {constant_type}")

    def add(self, constant: ConstantT) -> None:
        self.set_for_type(type(constant)).add(constant)  # type: ignore

    def __contains__(self, constant: ConstantT) -> bool:
        return constant in self.set_for_type(type(constant))

    def __or__(self, other: "Constants") -> "Constants":
        return Constants(
            integers=self.integers | other.integers,  # type: ignore
            floats=self.floats | other.floats,  # type: ignore
            bytes=self.bytes | other.bytes,  # type: ignore
            strings=self.strings | other.strings,  # type: ignore
        )

    def __iter__(self) -> Iterator[ConstantT]:
        return iter(chain(self.integers, self.floats, self.bytes, self.strings))

    def __len__(self) -> int:
        return (
            len(self.integers) + len(self.floats) + len(self.bytes) + len(self.strings)
        )

    def __repr__(self) -> str:
        return f"Constants({self.integers=}, {self.floats=}, {self.bytes=}, {self.strings=})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Constants):
            return False
        return (
            self.integers == other.integers
            and self.floats == other.floats
            and self.bytes == other.bytes
            and self.strings == other.strings
        )


class TooManyConstants(Exception):
    # a control flow exception which we raise in ConstantsVisitor when the
    # number of constants in a module gets too large.
    pass


class ConstantVisitor(NodeVisitor):
    CONSTANTS_LIMIT: int = 1024

    def __init__(self, *, limit: bool):
        super().__init__()
        self.constants = Constants()
        self.limit = limit

    def _add_constant(self, value: object) -> None:
        pass

    def visit_UnaryOp(self, node: UnaryOp) -> None:
        # `a = -1` is actually a combination of a USub and the constant 1.
        pass

    def visit_Expr(self, node: Expr) -> None:
        pass

    def visit_JoinedStr(self, node):
        # dont recurse on JoinedStr, i.e. f strings. Constants that appear *only*
        # in f strings are unlikely to be helpful.
        pass

    def visit_Constant(self, node):
        pass


def _constants_from_source(source: str | bytes, *, limit: bool) -> Constants:
    pass


def _constants_file_str(constants: Constants) -> str:
    pass


@lru_cache(4096)
def constants_from_module(module: ModuleType, *, limit: bool = True) -> Constants:
    pass


@lru_cache(4096)
def is_local_module_file(path: str) -> bool:
    pass
