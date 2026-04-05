# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
Write patches which add @example() decorators for discovered test cases.

Requires `hypothesis[codemods,ghostwriter]` installed, i.e. black and libcst.

This module is used by Hypothesis' builtin pytest plugin for failing examples
discovered during testing, and by HypoFuzz for _covering_ examples discovered
during fuzzing.
"""

import ast
import difflib
import hashlib
import inspect
import re
import sys
import types
from ast import literal_eval
from collections.abc import Sequence
from contextlib import suppress
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import libcst as cst
from libcst import matchers as m
from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand
from libcst.metadata import ExpressionContext, ExpressionContextProvider

from hypothesis.configuration import storage_directory
from hypothesis.version import __version__

try:
    import black
except ImportError:
    black = None  # type: ignore

HEADER = """\
From HEAD Mon Sep 17 00:00:00 2001
From: {author}
Date: {when:%a, %d %b %Y %H:%M:%S}
Subject: [PATCH] {msg}

---
"""
FAIL_MSG = "discovered failure"
_space_only_re = re.compile("^ +$", re.MULTILINE)
_leading_space_re = re.compile("(^[ ]*)(?:[^ \n])", re.MULTILINE)


def dedent(text: str) -> tuple[str, str]:
    # Simplified textwrap.dedent, for valid Python source code only
    text = _space_only_re.sub("", text)
    prefix = min(_leading_space_re.findall(text), key=len)
    return re.sub(r"(?m)^" + prefix, "", text), prefix


def indent(text: str, prefix: str) -> str:
    pass


class AddExamplesCodemod(VisitorBasedCodemodCommand):
    DESCRIPTION = "Add explicit examples to failing tests."

    def __init__(
        self,
        context: CodemodContext,
        fn_examples: dict[str, list[tuple[cst.Call, str]]],
        strip_via: tuple[str, ...] = (),
        decorator: str = "example",
        width: int = 88,
    ):
        """Add @example() decorator(s) for failing test(s).

        `code` is the source code of the module where the test functions are defined.
        `fn_examples` is a dict of function name to list-of-failing-examples.
        """
        assert fn_examples, "This codemod does nothing without fn_examples."
        super().__init__(context)

        self.decorator_func = cst.parse_expression(decorator)
        self.line_length = width
        value_in_strip_via: Any = m.MatchIfTrue(
            lambda x: literal_eval(x.value) in strip_via
        )
        self.strip_matching = m.Call(
            m.Attribute(m.Call(), m.Name("via")),
            [m.Arg(m.SimpleString() & value_in_strip_via)],
        )

        # Codemod the failing examples to Call nodes usable as decorators
        self.fn_examples = {
            k: tuple(
                d
                for (node, via) in nodes
                if (d := self.__call_node_to_example_dec(node, via))
            )
            for k, nodes in fn_examples.items()
        }

    def __call_node_to_example_dec(
        self, node: cst.Call, via: str
    ) -> cst.Decorator | None:
        # If we have black installed, remove trailing comma, _unless_ there's a comment
        node = node.with_changes(
            func=self.decorator_func,
            args=(
                [
                    a.with_changes(
                        comma=(
                            a.comma
                            if m.findall(a.comma, m.Comment())
                            else cst.MaybeSentinel.DEFAULT
                        )
                    )
                    for a in node.args
                ]
                if black
                else node.args
            ),
        )
        via: cst.BaseExpression = cst.Call(
            func=cst.Attribute(node, cst.Name("via")),
            args=[cst.Arg(cst.SimpleString(repr(via)))],
        )
        if black:  # pragma: no branch
            try:
                pretty = black.format_str(
                    cst.Module([]).code_for_node(via),
                    mode=black.Mode(line_length=self.line_length),
                )
            except (ImportError, AttributeError):  # pragma: no cover
                return None  # See https://github.com/psf/black/pull/4224
            via = cst.parse_expression(pretty.strip())
        return cst.Decorator(via)

    def leave_FunctionDef(
        self, _original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        pass


def get_patch_for(
    func: Any,
    examples: Sequence[tuple[str, str]],
    *,
    strip_via: tuple[str, ...] = (),
) -> tuple[str, str, str] | None:
    # Skip this if we're unable to find the location of this function.
    pass


# split out for easier testing of patches in hypofuzz, where the function to
# apply the patch to may not be loaded in sys.modules.
def _get_patch_for(
    func: Any,
    examples: Sequence[tuple[str, str]],
    *,
    strip_via: tuple[str, ...] = (),
    namespace: dict[str, Any],
) -> tuple[str, str] | None:
    pass


def make_patch(
    triples: Sequence[tuple[str, str, str]],
    *,
    msg: str = "Hypothesis: add explicit examples",
    when: datetime | None = None,
    author: str = f"Hypothesis {__version__} <no-reply@hypothesis.works>",
) -> str:
    """Create a patch for (fname, before, after) triples."""
    pass


def save_patch(patch: str, *, slug: str = "") -> Path:  # pragma: no cover
    pass


def gc_patches(slug: str = "") -> None:  # pragma: no cover
    pass
