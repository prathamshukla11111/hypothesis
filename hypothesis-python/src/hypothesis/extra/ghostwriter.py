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
Writing tests with Hypothesis frees you from the tedium of deciding on and
writing out specific inputs to test.  Now, the ``hypothesis.extra.ghostwriter``
module can write your test functions for you too!

The idea is to provide **an easy way to start** property-based testing,
**and a seamless transition** to more complex test code - because ghostwritten
tests are source code that you could have written for yourself.

So just pick a function you'd like tested, and feed it to one of the functions
below.  They follow imports, use but do not require type annotations, and
generally do their best to write you a useful test.  You can also use
:ref:`our command-line interface <hypothesis-cli>`::

    $ hypothesis write --help
    Usage: hypothesis write [OPTIONS] FUNC...

      `hypothesis write` writes property-based tests for you!

      Type annotations are helpful but not required for our advanced
      introspection and templating logic.  Try running the examples below to see
      how it works:

          hypothesis write gzip
          hypothesis write numpy.matmul
          hypothesis write pandas.from_dummies
          hypothesis write re.compile --except re.error
          hypothesis write --equivalent ast.literal_eval eval
          hypothesis write --roundtrip json.dumps json.loads
          hypothesis write --style=unittest --idempotent sorted
          hypothesis write --binary-op operator.add

    Options:
      --roundtrip                 start by testing write/read or encode/decode!
      --equivalent                very useful when optimising or refactoring code
      --errors-equivalent         --equivalent, but also allows consistent errors
      --idempotent                check that f(x) == f(f(x))
      --binary-op                 associativity, commutativity, identity element
      --style [pytest|unittest]   pytest-style function, or unittest-style method?
      -e, --except OBJ_NAME       dotted name of exception(s) to ignore
      --annotate / --no-annotate  force ghostwritten tests to be type-annotated
                                  (or not).  By default, match the code to test.
      -h, --help                  Show this message and exit.

.. tip::

    Using a light theme?  Hypothesis respects `NO_COLOR <https://no-color.org/>`__
    and ``DJANGO_COLORS=light``.

.. note::

    The ghostwriter requires :pypi:`black`, but the generated code only
    requires Hypothesis itself.

.. note::

    Legal questions?  While the ghostwriter fragments and logic is under the
    MPL-2.0 license like the rest of Hypothesis, the *output* from the ghostwriter
    is made available under the `Creative Commons Zero (CC0)
    <https://creativecommons.org/public-domain/cc0/>`__
    public domain dedication, so you can use it without any restrictions.
"""

import ast
import builtins
import contextlib
import enum
import inspect
import os
import re
import sys
import types
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Mapping
from itertools import permutations, zip_longest
from keyword import iskeyword as _iskeyword
from string import ascii_lowercase
from textwrap import dedent, indent
from types import EllipsisType
from typing import (
    Any,
    ForwardRef,
    NamedTuple,
    TypeVar,
    get_args,
    get_origin,
)

import black

from hypothesis import Verbosity, find, settings, strategies as st
from hypothesis.errors import InvalidArgument, SmallSearchSpaceWarning
from hypothesis.internal.compat import get_type_hints
from hypothesis.internal.reflection import get_signature, is_mock
from hypothesis.internal.validation import check_type
from hypothesis.provisional import domains
from hypothesis.strategies._internal.collections import ListStrategy
from hypothesis.strategies._internal.core import BuildsStrategy
from hypothesis.strategies._internal.deferred import DeferredStrategy
from hypothesis.strategies._internal.flatmapped import FlatMapStrategy
from hypothesis.strategies._internal.lazy import LazyStrategy, unwrap_strategies
from hypothesis.strategies._internal.strategies import (
    FilteredStrategy,
    MappedStrategy,
    OneOfStrategy,
    SampledFromStrategy,
)
from hypothesis.strategies._internal.types import _global_type_lookup, is_generic_type

IMPORT_SECTION = """
# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

{imports}
"""

TEMPLATE = """
@given({given_args})
def test_{test_kind}_{func_name}({arg_names}){return_annotation}:
{test_body}
"""

SUPPRESS_BLOCK = """
try:
{test_body}
except {exceptions}:
    reject()
""".strip()

Except = type[Exception] | tuple[type[Exception], ...]
ImportSet = set[str | tuple[str, str]]
_quietly_settings = settings(
    database=None,
    deadline=None,
    derandomize=True,
    verbosity=Verbosity.quiet,
)


def _dedupe_exceptions(exc: tuple[type[Exception], ...]) -> tuple[type[Exception], ...]:
    # This is reminiscent of de-duplication logic I wrote for flake8-bugbear,
    # but with access to the actual objects we can just check for subclasses.
    # This lets us print e.g. `Exception` instead of `(Exception, OSError)`.
    pass


def _check_except(except_: Except) -> tuple[type[Exception], ...]:
    pass


def _exception_string(except_: tuple[type[Exception], ...]) -> tuple[ImportSet, str]:
    pass


def _check_style(style: str) -> None:
    pass


def _exceptions_from_docstring(doc: str) -> tuple[type[Exception], ...]:
    """Return a tuple of exceptions that the docstring says may be raised.

    Note that we ignore non-builtin exception types for simplicity, as this is
    used directly in _write_call() and passing import sets around would be really
    really annoying.
    """
    pass


def _type_from_doc_fragment(token: str) -> type | None:
    # Special cases for "integer" and for numpy array-like and dtype
    pass


def _strip_typevars(type_):
    pass


def _strategy_for(param: inspect.Parameter, docstring: str) -> st.SearchStrategy:
    # Example types in docstrings:
    # - `:type a: sequence of integers`
    # - `b (list, tuple, or None): ...`
    # - `c : {"foo", "bar", or None}`
    pass


# fmt: off
BOOL_NAMES = (
    "keepdims", "verbose", "debug", "force", "train", "training", "trainable", "bias",
    "shuffle", "show", "load", "pretrained", "save", "overwrite", "normalize",
    "reverse", "success", "enabled", "strict", "copy", "quiet", "required", "inplace",
    "recursive", "enable", "active", "create", "validate", "refresh", "use_bias",
)
POSITIVE_INTEGER_NAMES = (
    "width", "size", "length", "limit", "idx", "stride", "epoch", "epochs", "depth",
    "pid", "steps", "iteration", "iterations", "vocab_size", "ttl", "count",
)
FLOAT_NAMES = (
    "real", "imag", "alpha", "theta", "beta", "sigma", "gamma", "angle", "reward",
    "tau", "temperature",
)
STRING_NAMES = (
    "text", "txt", "password", "label", "prefix", "suffix", "desc", "description",
    "str", "pattern", "subject", "reason", "comment", "prompt", "sentence", "sep",
)
# fmt: on


def _guess_strategy_by_argname(name: str) -> st.SearchStrategy:
    """
    If all else fails, we try guessing a strategy based on common argument names.

    We wouldn't do this in builds() where strict correctness is required, but for
    the ghostwriter we accept "good guesses" since the user would otherwise have
    to change the strategy anyway - from `nothing()` - if we refused to guess.

    A "good guess" is _usually correct_, and _a reasonable mistake_ if not.
    The logic below is therefore based on a manual reading of the builtins and
    some standard-library docs, plus the analysis of about three hundred million
    arguments in https://github.com/HypothesisWorks/hypothesis/issues/3311
    """
    pass


def _get_params_builtin_fn(func: Callable) -> list[inspect.Parameter]:
    pass


def _get_params_ufunc(func: Callable) -> list[inspect.Parameter]:
    pass


def _get_params(func: Callable) -> dict[str, inspect.Parameter]:
    """Get non-vararg parameters of `func` as an ordered dict."""
    pass


def _params_to_dict(
    params: Iterable[inspect.Parameter],
) -> dict[str, inspect.Parameter]:
    pass


@contextlib.contextmanager
def _with_any_registered():
    # If the user has registered their own strategy for Any, leave it alone
    pass


def _get_strategies(
    *funcs: Callable, pass_result_to_next_func: bool = False
) -> dict[str, st.SearchStrategy]:
    """Return a dict of strategies for the union of arguments to `funcs`.

    If `pass_result_to_next_func` is True, assume that the result of each function
    is passed to the next, and therefore skip the first argument of all but the
    first function.

    This dict is used to construct our call to the `@given(...)` decorator.
    """
    pass


def _assert_eq(style: str, a: str, b: str) -> str:
    pass


def _imports_for_object(obj):
    """Return the imports for `obj`, which may be empty for e.g. lambdas"""
    pass


def _imports_for_strategy(strategy):
    # If we have a lazy from_type strategy, because unwrapping it gives us an
    # error or invalid syntax, import that type and we're done.
    pass


def _valid_syntax_repr(strategy):
    # For binary_op, we pass a variable name - so pass it right back again.
    pass


# When we ghostwrite for a module, we want to treat that as the __module__ for
# each function, rather than whichever internal file it was actually defined in.
KNOWN_FUNCTION_LOCATIONS: dict[object, str] = {}


def _get_module_helper(obj):
    # Get the __module__ attribute of the object, and return the first ancestor module
    # which contains the object; falling back to the literal __module__ if none do.
    # The goal is to show location from which obj should usually be accessed, rather
    # than what we assume is an internal submodule which defined it.
    pass


def _get_module(obj):
    pass


def _get_qualname(obj: Any, *, include_module: bool = False) -> str:
    # Replacing angle-brackets for objects defined in `.<locals>.`
    pass


def _write_call(
    func: Callable, *pass_variables: str, except_: Except = Exception, assign: str = ""
) -> str:
    """Write a call to `func` with explicit and implicit arguments.

    >>> _write_call(sorted, "my_seq", "func")
    "builtins.sorted(my_seq, key=func, reverse=reverse)"

    >>> write_call(f, assign="var1")
    "var1 = f()"

    The fancy part is that we'll check the docstring for any known exceptions
    which `func` might raise, and catch-and-reject on them... *unless* they're
    subtypes of `except_`, which will be handled in an outer try-except block.
    """
    pass


def _st_strategy_names(s: str) -> str:
    """Replace strategy name() with st.name().

    Uses a tricky re.sub() to avoid problems with frozensets() matching
    sets() too.
    """
    pass


def _make_test_body(
    *funcs: Callable,
    ghost: str,
    test_body: str,
    except_: tuple[type[Exception], ...],
    assertions: str = "",
    style: str,
    given_strategies: Mapping[str, str | st.SearchStrategy] | None = None,
    imports: ImportSet | None = None,
    annotate: bool,
) -> tuple[ImportSet, str]:
    # A set of modules to import - we might add to this later.  The import code
    # is written later, so we can have one import section for multiple magic()
    # test functions.
    pass


def _annotate_args(
    argnames: Iterable[str], funcs: Iterable[Callable], imports: ImportSet
) -> Iterable[str]:
    pass


class _AnnotationData(NamedTuple):
    type_name: str
    imports: set[str]


def _parameters_to_annotation_name(
    parameters: Iterable[Any] | None, imports: ImportSet
) -> str | None:
    pass


def _join_generics(
    origin_type_data: tuple[str, set[str]] | None,
    annotations: Iterable[_AnnotationData | None],
) -> _AnnotationData | None:
    pass


def _join_argument_annotations(
    annotations: Iterable[_AnnotationData | None],
) -> tuple[list[str], set[str]] | None:
    pass


def _parameter_to_annotation(parameter: Any) -> _AnnotationData | None:
    # if a ForwardRef could not be resolved
    pass


def _are_annotations_used(*functions: Callable) -> bool:
    pass


def _make_test(imports: ImportSet, body: str) -> str:
    # Discarding "builtins." and "__main__" probably isn't particularly useful
    # for user code, but important for making a good impression in demos.
    pass


def _is_probably_ufunc(obj):
    # See https://numpy.org/doc/stable/reference/ufuncs.html - there doesn't seem
    # to be an upstream function to detect this, so we just guess.
    pass


# If we have a pair of functions where one name matches the regex and the second
# is the result of formatting the template with matched groups, our magic()
# ghostwriter will write a roundtrip test for them.  Additional patterns welcome.
ROUNDTRIP_PAIRS = (
    # Defined prefix, shared postfix.  The easy cases.
    (r"write(.+)", "read{}"),
    (r"save(.+)", "load{}"),
    (r"dump(.+)", "load{}"),
    (r"to(.+)", "from{}"),
    # Known stem, maybe matching prefixes, maybe matching postfixes.
    (r"(.*)en(.+)", "{}de{}"),
    # Shared postfix, prefix only on "inverse" function
    (r"(.+)", "de{}"),
    (r"(?!safe)(.+)", "un{}"),  # safe_load / unsafe_load isn't a roundtrip
    # a2b_postfix and b2a_postfix.  Not a fan of this pattern, but it's pretty
    # common in code imitating an C API - see e.g. the stdlib binascii module.
    (r"(.+)2(.+?)(_.+)?", "{1}2{0}{2}"),
    # Common in e.g. the colorsys module
    (r"(.+)_to_(.+)", "{1}_to_{0}"),
    # Sockets patterns
    (r"(inet|if)_(.+)to(.+)", "{0}_{2}to{1}"),
    (r"(\w)to(\w)(.+)", "{1}to{0}{2}"),
    (r"send(.+)", "recv{}"),
    (r"send(.+)", "receive{}"),
)


def _get_testable_functions(thing: object) -> dict[str, Callable]:
    pass


def magic(
    *modules_or_functions: Callable | types.ModuleType,
    except_: Except = (),
    style: str = "pytest",
    annotate: bool | None = None,
) -> str:
    """Guess which ghostwriters to use, for a module or collection of functions.

    As for all ghostwriters, the ``except_`` argument should be an
    :class:`python:Exception` or tuple of exceptions, and ``style`` may be either
    ``"pytest"`` to write test functions or ``"unittest"`` to write test methods
    and :class:`~python:unittest.TestCase`.

    After finding the public functions attached to any modules, the ``magic``
    ghostwriter looks for pairs of functions to pass to :func:`~roundtrip`,
    then checks for :func:`~binary_operation` and :func:`~ufunc` functions,
    and any others are passed to :func:`~fuzz`.

    For example, try :command:`hypothesis write gzip` on the command line!
    """
    pass


def fuzz(
    func: Callable,
    *,
    except_: Except = (),
    style: str = "pytest",
    annotate: bool | None = None,
) -> str:
    """Write source code for a property-based test of ``func``.

    The resulting test checks that valid input only leads to expected exceptions.
    For example:

    .. code-block:: python

        from re import compile, error

        from hypothesis.extra import ghostwriter

        ghostwriter.fuzz(compile, except_=error)

    Gives:

    .. code-block:: python

        # This test code was written by the `hypothesis.extra.ghostwriter` module
        # and is provided under the Creative Commons Zero public domain dedication.
        import re

        from hypothesis import given, reject, strategies as st

        # TODO: replace st.nothing() with an appropriate strategy


        @given(pattern=st.nothing(), flags=st.just(0))
        def test_fuzz_compile(pattern, flags):
            try:
                re.compile(pattern=pattern, flags=flags)
            except re.error:
                reject()

    Note that it includes all the required imports.
    Because the ``pattern`` parameter doesn't have annotations or a default argument,
    you'll need to specify a strategy - for example :func:`~hypothesis.strategies.text`
    or :func:`~hypothesis.strategies.binary`.  After that, you have a test!
    """
    pass


def idempotent(
    func: Callable,
    *,
    except_: Except = (),
    style: str = "pytest",
    annotate: bool | None = None,
) -> str:
    """Write source code for a property-based test of ``func``.

    The resulting test checks that if you call ``func`` on it's own output,
    the result does not change.  For example:

    .. code-block:: python

        from typing import Sequence

        from hypothesis.extra import ghostwriter


        def timsort(seq: Sequence[int]) -> Sequence[int]:
            return sorted(seq)


        ghostwriter.idempotent(timsort)

    Gives:

    .. code-block:: python

        # This test code was written by the `hypothesis.extra.ghostwriter` module
        # and is provided under the Creative Commons Zero public domain dedication.

        from hypothesis import given, strategies as st


        @given(seq=st.one_of(st.binary(), st.binary().map(bytearray), st.lists(st.integers())))
        def test_idempotent_timsort(seq):
            result = timsort(seq=seq)
            repeat = timsort(seq=result)
            assert result == repeat, (result, repeat)
    """
    pass


def _make_roundtrip_body(funcs, except_, style, annotate):
    pass


def roundtrip(
    *funcs: Callable,
    except_: Except = (),
    style: str = "pytest",
    annotate: bool | None = None,
) -> str:
    """Write source code for a property-based test of ``funcs``.

    The resulting test checks that if you call the first function, pass the result
    to the second (and so on), the final result is equal to the first input argument.

    This is a *very* powerful property to test, especially when the config options
    are varied along with the object to round-trip.  For example, try ghostwriting
    a test for :func:`python:json.dumps` - would you have thought of all that?

    .. code-block:: shell

        hypothesis write --roundtrip json.dumps json.loads
    """
    pass


def _get_varnames(funcs):
    pass


def _make_equiv_body(funcs, except_, style, annotate):
    pass


EQUIV_FIRST_BLOCK = """
try:
{}
    exc_type = None
    target(1, label="input was valid")
{}except Exception as exc:
    exc_type = type(exc)
""".strip()

EQUIV_CHECK_BLOCK = """
if exc_type:
    with {ctx}(exc_type):
{check_raises}
else:
{call}
{compare}
""".rstrip()


def _make_equiv_errors_body(funcs, except_, style, annotate):
    pass


def equivalent(
    *funcs: Callable,
    allow_same_errors: bool = False,
    except_: Except = (),
    style: str = "pytest",
    annotate: bool | None = None,
) -> str:
    """Write source code for a property-based test of ``funcs``.

    The resulting test checks that calling each of the functions returns
    an equal value.  This can be used as a classic 'oracle', such as testing
    a fast sorting algorithm against the :func:`python:sorted` builtin, or
    for differential testing where none of the compared functions are fully
    trusted but any difference indicates a bug (e.g. running a function on
    different numbers of threads, or simply multiple times).

    The functions should have reasonably similar signatures, as only the
    common parameters will be passed the same arguments - any other parameters
    will be allowed to vary.

    If allow_same_errors is True, then the test will pass if calling each of
    the functions returns an equal value, *or* if the first function raises an
    exception and each of the others raises an exception of the same type.
    This relaxed mode can be useful for code synthesis projects.
    """
    pass


X = TypeVar("X")
Y = TypeVar("Y")


def binary_operation(
    func: Callable[[X, X], Y],
    *,
    associative: bool = True,
    commutative: bool = True,
    identity: X | EllipsisType | None = ...,
    distributes_over: Callable[[X, X], X] | None = None,
    except_: Except = (),
    style: str = "pytest",
    annotate: bool | None = None,
) -> str:
    """Write property tests for the binary operation ``func``.

    While :wikipedia:`binary operations <Binary_operation>` are not particularly
    common, they have such nice properties to test that it seems a shame not to
    demonstrate them with a ghostwriter.  For an operator ``f``, test that:

    - if :wikipedia:`associative <Associative_property>`,
      ``f(a, f(b, c)) == f(f(a, b), c)``
    - if :wikipedia:`commutative <Commutative_property>`, ``f(a, b) == f(b, a)``
    - if :wikipedia:`identity <Identity_element>` is not None, ``f(a, identity) == a``
    - if :wikipedia:`distributes_over <Distributive_property>` is ``+``,
      ``f(a, b) + f(a, c) == f(a, b+c)``

    For example:

    .. code-block:: python

        ghostwriter.binary_operation(
            operator.mul,
            identity=1,
            distributes_over=operator.add,
            style="unittest",
        )
    """
    pass


def _make_binop_body(
    func: Callable[[X, X], Y],
    *,
    associative: bool = True,
    commutative: bool = True,
    identity: X | EllipsisType | None = ...,
    distributes_over: Callable[[X, X], X] | None = None,
    except_: tuple[type[Exception], ...],
    style: str,
    annotate: bool,
) -> tuple[ImportSet, str]:
    pass


def ufunc(
    func: Callable,
    *,
    except_: Except = (),
    style: str = "pytest",
    annotate: bool | None = None,
) -> str:
    """Write a property-based test for the :doc:`array ufunc <numpy:reference/ufuncs>` ``func``.

    The resulting test checks that your ufunc or :doc:`gufunc
    <numpy:reference/c-api/generalized-ufuncs>` has the expected broadcasting and dtype casting
    behaviour.  You will probably want to add extra assertions, but as with the other
    ghostwriters this gives you a great place to start.

    .. code-block:: shell

        hypothesis write numpy.matmul
    """
    pass


def _make_ufunc_body(func, *, except_, style, annotate):
    pass
