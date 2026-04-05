# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""This module provides the core primitives of Hypothesis, such as given."""

import base64
import contextlib
import dataclasses
import datetime
import inspect
import io
import math
import os
import sys
import threading
import time
import traceback
import types
import unittest
import warnings
import zlib
from collections import defaultdict
from collections.abc import Callable, Coroutine, Generator, Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import partial
from inspect import Parameter
from random import Random
from threading import Lock
from types import EllipsisType
from typing import (
    Any,
    BinaryIO,
    TypeVar,
    overload,
)
from unittest import TestCase

from hypothesis import strategies as st
from hypothesis._settings import (
    HealthCheck,
    Phase,
    Verbosity,
    all_settings,
    local_settings,
    settings as Settings,
)
from hypothesis.control import BuildContext, currently_in_test_context
from hypothesis.database import choices_from_bytes, choices_to_bytes
from hypothesis.errors import (
    BackendCannotProceed,
    DeadlineExceeded,
    DidNotReproduce,
    FailedHealthCheck,
    FlakyFailure,
    FlakyReplay,
    Found,
    Frozen,
    HypothesisException,
    HypothesisWarning,
    InvalidArgument,
    NoSuchExample,
    StopTest,
    Unsatisfiable,
    UnsatisfiedAssumption,
)
from hypothesis.internal import observability
from hypothesis.internal.compat import (
    PYPY,
    BaseExceptionGroup,
    add_note,
    bad_django_TestCase,
    get_type_hints,
    int_from_bytes,
)
from hypothesis.internal.conjecture.choice import ChoiceT
from hypothesis.internal.conjecture.data import ConjectureData, Status
from hypothesis.internal.conjecture.engine import BUFFER_SIZE, ConjectureRunner
from hypothesis.internal.conjecture.junkdrawer import (
    ensure_free_stackframes,
    gc_cumulative_time,
)
from hypothesis.internal.conjecture.providers import (
    BytestringProvider,
    PrimitiveProvider,
)
from hypothesis.internal.conjecture.shrinker import sort_key
from hypothesis.internal.entropy import deterministic_PRNG
from hypothesis.internal.escalation import (
    InterestingOrigin,
    current_pytest_item,
    format_exception,
    get_trimmed_traceback,
    is_hypothesis_file,
)
from hypothesis.internal.healthcheck import fail_health_check
from hypothesis.internal.observability import (
    InfoObservation,
    InfoObservationType,
    deliver_observation,
    make_testcase,
    observability_enabled,
)
from hypothesis.internal.reflection import (
    convert_positional_arguments,
    define_function_signature,
    function_digest,
    get_pretty_function_description,
    get_signature,
    impersonate,
    is_mock,
    nicerepr,
    proxies,
    repr_call,
)
from hypothesis.internal.scrutineer import (
    MONITORING_TOOL_ID,
    Trace,
    Tracer,
    explanatory_lines,
    tractable_coverage_report,
)
from hypothesis.internal.validation import check_type
from hypothesis.reporting import (
    current_verbosity,
    report,
    verbose_report,
    with_reporter,
)
from hypothesis.statistics import describe_statistics, describe_targets, note_statistics
from hypothesis.strategies._internal.misc import NOTHING
from hypothesis.strategies._internal.strategies import (
    Ex,
    SearchStrategy,
    check_strategy,
)
from hypothesis.utils.conventions import not_set
from hypothesis.utils.threading import ThreadLocal
from hypothesis.vendor.pretty import RepresentationPrinter
from hypothesis.version import __version__

TestFunc = TypeVar("TestFunc", bound=Callable)


running_under_pytest = False
pytest_shows_exceptiongroups = True
global_force_seed = None
# this variable stores "engine-global" constants, which are global relative to a
# ConjectureRunner instance (roughly speaking). Since only one conjecture runner
# instance can be active per thread, making engine constants thread-local prevents
# the ConjectureRunner instances of concurrent threads from treading on each other.
threadlocal = ThreadLocal(_hypothesis_global_random=lambda: None)


@dataclass(slots=True, frozen=False)
class Example:
    args: Any
    kwargs: Any
    # Plus two optional arguments for .xfail()
    raises: Any = field(default=None)
    reason: Any = field(default=None)


@dataclass(slots=True, frozen=True)
class ReportableError:
    fragments: list[str]
    exception: BaseException


# TODO_DOCS link to not-yet-existent patch-dumping docs


class example:
    """
    Add an explicit input to a Hypothesis test, which Hypothesis will always
    try before generating random inputs. This combines the randomized nature of
    Hypothesis generation with a traditional parametrized test.

    For example:

    .. code-block:: python

        @example("Hello world")
        @example("some string with special significance")
        @given(st.text())
        def test_strings(s):
            pass

    will call ``test_strings("Hello World")`` and
    ``test_strings("some string with special significance")`` before generating
    any random inputs. |@example| may be placed in any order relative to |@given|
    and |@settings|.

    Explicit inputs from |@example| are run in the |Phase.explicit| phase.
    Explicit inputs do not count towards |settings.max_examples|. Note that
    explicit inputs added by |@example| do not shrink. If an explicit input
    fails, Hypothesis will stop and report the failure without generating any
    random inputs.

    |@example| can also be used to easily reproduce a failure. For instance, if
    Hypothesis reports that ``f(n=[0, math.nan])`` fails, you can add
    ``@example(n=[0, math.nan])`` to your test to quickly reproduce that failure.

    Arguments to ``@example``
    -------------------------

    Arguments to |@example| have the same behavior and restrictions as arguments
    to |@given|. This means they may be either positional or keyword arguments
    (but not both in the same |@example|):

    .. code-block:: python

        @example(1, 2)
        @example(x=1, y=2)
        @given(st.integers(), st.integers())
        def test(x, y):
            pass

    Noting that while arguments to |@given| are strategies (like |st.integers|),
    arguments to |@example| are values instead (like ``1``).

    See the :ref:`given-arguments` section for full details.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args and kwargs:
            raise InvalidArgument(
                "Cannot mix positional and keyword arguments for examples"
            )
        if not (args or kwargs):
            raise InvalidArgument("An example must provide at least one argument")

        self.hypothesis_explicit_examples: list[Example] = []
        self._this_example = Example(tuple(args), kwargs)

    def __call__(self, test: TestFunc) -> TestFunc:
        if not hasattr(test, "hypothesis_explicit_examples"):
            test.hypothesis_explicit_examples = self.hypothesis_explicit_examples  # type: ignore
        test.hypothesis_explicit_examples.append(self._this_example)  # type: ignore
        return test

    def xfail(
        self,
        condition: bool = True,  # noqa: FBT002
        *,
        reason: str = "",
        raises: type[BaseException] | tuple[type[BaseException], ...] = BaseException,
    ) -> "example":
        """Mark this example as an expected failure, similarly to
        :obj:`pytest.mark.xfail(strict=True) <pytest.mark.xfail>`.

        Expected-failing examples allow you to check that your test does fail on
        some examples, and therefore build confidence that *passing* tests are
        because your code is working, not because the test is missing something.

        .. code-block:: python

            @example(...).xfail()
            @example(...).xfail(reason="Prices must be non-negative")
            @example(...).xfail(raises=(KeyError, ValueError))
            @example(...).xfail(sys.version_info[:2] >= (3, 12), reason="needs py 3.12")
            @example(...).xfail(condition=sys.platform != "linux", raises=OSError)
            def test(x):
                pass

        .. note::

            Expected-failing examples are handled separately from those generated
            by strategies, so you should usually ensure that there is no overlap.

            .. code-block:: python

                @example(x=1, y=0).xfail(raises=ZeroDivisionError)
                @given(x=st.just(1), y=st.integers())  # Missing `.filter(bool)`!
                def test_fraction(x, y):
                    # This test will try the explicit example and see it fail as
                    # expected, then go on to generate more examples from the
                    # strategy.  If we happen to generate y=0, the test will fail
                    # because only the explicit example is treated as xfailing.
                    x / y
        """
        check_type(bool, condition, "condition")
        check_type(str, reason, "reason")
        if not (
            isinstance(raises, type) and issubclass(raises, BaseException)
        ) and not (
            isinstance(raises, tuple)
            and raises  # () -> expected to fail with no error, which is impossible
            and all(
                isinstance(r, type) and issubclass(r, BaseException) for r in raises
            )
        ):
            raise InvalidArgument(
                f"{raises=} must be an exception type or tuple of exception types"
            )
        if condition:
            self._this_example = dataclasses.replace(
                self._this_example, raises=raises, reason=reason
            )
        return self

    def via(self, whence: str, /) -> "example":
        """Attach a machine-readable label noting what the origin of this example
        was. |example.via| is completely optional and does not change runtime
        behavior.

        |example.via| is intended to support self-documenting behavior, as well as
        tooling which might add (or remove) |@example| decorators automatically.
        For example:

        .. code-block:: python

            # Annotating examples is optional and does not change runtime behavior
            @example(...)
            @example(...).via("regression test for issue #42")
            @example(...).via("discovered failure")
            def test(x):
                pass

        .. note::

            `HypoFuzz <https://hypofuzz.com/>`_ uses |example.via| to tag examples
            in the patch of its high-coverage set of explicit inputs, on
            `the patches page <https://hypofuzz.com/example-dashboard/#/patches>`_.
        """
        pass


def seed(seed: Hashable) -> Callable[[TestFunc], TestFunc]:
    """
    Seed the randomness for this test.

    ``seed`` may be any hashable object. No exact meaning for ``seed`` is provided
    other than that for a fixed seed value Hypothesis will produce the same
    examples (assuming that there are no other sources of nondeterminisim, such
    as timing, hash randomization, or external state).

    For example, the following test function and |RuleBasedStateMachine| will
    each generate the same series of examples each time they are executed:

    .. code-block:: python

        @seed(1234)
        @given(st.integers())
        def test(n): ...

        @seed(6789)
        class MyMachine(RuleBasedStateMachine): ...

    If using pytest, you can alternatively pass ``--hypothesis-seed`` on the
    command line.

    Setting a seed overrides |settings.derandomize|, which is designed to enable
    deterministic CI tests rather than reproducing observed failures.

    Hypothesis will only print the seed which would reproduce a failure if a test
    fails in an unexpected way, for instance inside Hypothesis internals.
    """

    def accept(test):
        test._hypothesis_internal_use_seed = seed
        current_settings = getattr(test, "_hypothesis_internal_use_settings", None)
        test._hypothesis_internal_use_settings = Settings(
            current_settings, database=None
        )
        return test

    return accept


# TODO_DOCS: link to /explanation/choice-sequence


def reproduce_failure(version: str, blob: bytes) -> Callable[[TestFunc], TestFunc]:
    """
    Run the example corresponding to the binary ``blob`` in order to reproduce a
    failure. ``blob`` is a serialized version of the internal input representation
    of Hypothesis.

    A test decorated with |@reproduce_failure| always runs exactly one example,
    which is expected to cause a failure. If the provided ``blob`` does not
    cause a failure, Hypothesis will raise |DidNotReproduce|.

    Hypothesis will print an |@reproduce_failure| decorator if
    |settings.print_blob| is ``True`` (which is the default in CI).

    |@reproduce_failure| is intended to be temporarily added to your test suite in
    order to reproduce a failure. It is not intended to be a permanent addition to
    your test suite. Because of this, no compatibility guarantees are made across
    Hypothesis versions, and |@reproduce_failure| will error if used on a different
    Hypothesis version than it was created for.

    .. seealso::

        See also the :doc:`/tutorial/replaying-failures` tutorial.
    """
    pass


def reproduction_decorator(choices: Iterable[ChoiceT]) -> str:
    return f"@reproduce_failure({__version__!r}, {encode_failure(choices)!r})"


def encode_failure(choices: Iterable[ChoiceT]) -> bytes:
    blob = choices_to_bytes(choices)
    compressed = zlib.compress(blob)
    if len(compressed) < len(blob):
        blob = b"\1" + compressed
    else:
        blob = b"\0" + blob
    return base64.b64encode(blob)


def decode_failure(blob: bytes) -> Sequence[ChoiceT]:
    try:
        decoded = base64.b64decode(blob)
    except Exception:
        raise InvalidArgument(f"Invalid base64 encoded string: {blob!r}") from None

    prefix = decoded[:1]
    if prefix == b"\0":
        decoded = decoded[1:]
    elif prefix == b"\1":
        try:
            decoded = zlib.decompress(decoded[1:])
        except zlib.error as err:
            raise InvalidArgument(
                f"Invalid zlib compression for blob {blob!r}"
            ) from err
    else:
        raise InvalidArgument(
            f"Could not decode blob {blob!r}: Invalid start byte {prefix!r}"
        )

    choices = choices_from_bytes(decoded)
    if choices is None:
        raise InvalidArgument(f"Invalid serialized choice sequence for blob {blob!r}")

    return choices


def _invalid(message, *, exc=InvalidArgument, test, given_kwargs):
    @impersonate(test)
    def wrapped_test(*arguments, **kwargs):  # pragma: no cover  # coverage limitation
        raise exc(message)

    wrapped_test.is_hypothesis_test = True
    wrapped_test.hypothesis = HypothesisHandle(
        inner_test=test,
        _get_fuzz_target=wrapped_test,
        _given_kwargs=given_kwargs,
    )
    return wrapped_test


def is_invalid_test(test, original_sig, given_arguments, given_kwargs):
    """Check the arguments to ``@given`` for basic usage constraints.

    Most errors are not raised immediately; instead we return a dummy test
    function that will raise the appropriate error if it is actually called.
    When the user runs a subset of tests (e.g via ``pytest -k``), errors will
    only be reported for tests that actually ran.
    """
    invalid = partial(_invalid, test=test, given_kwargs=given_kwargs)

    if not (given_arguments or given_kwargs):
        return invalid("given must be called with at least one argument")

    params = list(original_sig.parameters.values())
    pos_params = [p for p in params if p.kind is p.POSITIONAL_OR_KEYWORD]
    kwonly_params = [p for p in params if p.kind is p.KEYWORD_ONLY]
    if given_arguments and params != pos_params:
        return invalid(
            "positional arguments to @given are not supported with varargs, "
            "varkeywords, positional-only, or keyword-only arguments"
        )

    if len(given_arguments) > len(pos_params):
        return invalid(
            f"Too many positional arguments for {test.__name__}() were passed to "
            f"@given - expected at most {len(pos_params)} "
            f"arguments, but got {len(given_arguments)} {given_arguments!r}"
        )

    if ... in given_arguments:
        return invalid(
            "... was passed as a positional argument to @given, but may only be "
            "passed as a keyword argument or as the sole argument of @given"
        )

    if given_arguments and given_kwargs:
        return invalid("cannot mix positional and keyword arguments to @given")
    extra_kwargs = [
        k for k in given_kwargs if k not in {p.name for p in pos_params + kwonly_params}
    ]
    if extra_kwargs and (params == [] or params[-1].kind is not params[-1].VAR_KEYWORD):
        arg = extra_kwargs[0]
        extra = ""
        if arg in all_settings:
            extra = f". Did you mean @settings({arg}={given_kwargs[arg]!r})?"
        return invalid(
            f"{test.__name__}() got an unexpected keyword argument {arg!r}, "
            f"from `{arg}={given_kwargs[arg]!r}` in @given{extra}"
        )
    if any(p.default is not p.empty for p in params):
        return invalid("Cannot apply @given to a function with defaults.")

    # This case would raise Unsatisfiable *anyway*, but by detecting it here we can
    # provide a much more helpful error message for people e.g. using the Ghostwriter.
    empty = [
        f"{s!r} (arg {idx})" for idx, s in enumerate(given_arguments) if s is NOTHING
    ] + [f"{name}={s!r}" for name, s in given_kwargs.items() if s is NOTHING]
    if empty:
        strats = "strategies" if len(empty) > 1 else "strategy"
        return invalid(
            f"Cannot generate examples from empty {strats}: " + ", ".join(empty),
            exc=Unsatisfiable,
        )


def execute_explicit_examples(state, wrapped_test, arguments, kwargs, original_sig):
    assert isinstance(state, StateForActualGivenExecution)
    posargs = [
        p.name
        for p in original_sig.parameters.values()
        if p.kind is p.POSITIONAL_OR_KEYWORD
    ]

    for example in reversed(getattr(wrapped_test, "hypothesis_explicit_examples", ())):
        assert isinstance(example, Example)
        # All of this validation is to check that @example() got "the same" arguments
        # as @given, i.e. corresponding to the same parameters, even though they might
        # be any mixture of positional and keyword arguments.
        if example.args:
            assert not example.kwargs
            if any(
                p.kind is p.POSITIONAL_ONLY for p in original_sig.parameters.values()
            ):
                raise InvalidArgument(
                    "Cannot pass positional arguments to @example() when decorating "
                    "a test function which has positional-only parameters."
                )
            if len(example.args) > len(posargs):
                raise InvalidArgument(
                    "example has too many arguments for test. Expected at most "
                    f"{len(posargs)} but got {len(example.args)}"
                )
            example_kwargs = dict(
                zip(posargs[-len(example.args) :], example.args, strict=True)
            )
        else:
            example_kwargs = dict(example.kwargs)
        given_kws = ", ".join(
            repr(k) for k in sorted(wrapped_test.hypothesis._given_kwargs)
        )
        example_kws = ", ".join(repr(k) for k in sorted(example_kwargs))
        if given_kws != example_kws:
            raise InvalidArgument(
                f"Inconsistent args: @given() got strategies for {given_kws}, "
                f"but @example() got arguments for {example_kws}"
            ) from None

        # This is certainly true because the example_kwargs exactly match the params
        # reserved by @given(), which are then remove from the function signature.
        assert set(example_kwargs).isdisjoint(kwargs)
        example_kwargs.update(kwargs)

        if Phase.explicit not in state.settings.phases:
            continue

        with local_settings(state.settings):
            fragments_reported = []
            empty_data = ConjectureData.for_choices([])
            try:
                execute_example = partial(
                    state.execute_once,
                    empty_data,
                    is_final=True,
                    print_example=True,
                    example_kwargs=example_kwargs,
                )
                with with_reporter(fragments_reported.append):
                    if example.raises is None:
                        execute_example()
                    else:
                        # @example(...).xfail(...)
                        bits = ", ".join(nicerepr(x) for x in arguments) + ", ".join(
                            f"{k}={nicerepr(v)}" for k, v in example_kwargs.items()
                        )
                        try:
                            execute_example()
                        except failure_exceptions_to_catch() as err:
                            if not isinstance(err, example.raises):
                                raise
                            # Save a string form of this example; we'll warn if it's
                            # ever generated by the strategy (which can't be xfailed)
                            state.xfail_example_reprs.add(
                                repr_call(state.test, arguments, example_kwargs)
                            )
                        except example.raises as err:
                            # We'd usually check this as early as possible, but it's
                            # possible for failure_exceptions_to_catch() to grow when
                            # e.g. pytest is imported between import- and test-time.
                            raise InvalidArgument(
                                f"@example({bits}) raised an expected {err!r}, "
                                "but Hypothesis does not treat this as a test failure"
                            ) from err
                        else:
                            # Unexpectedly passing; always raise an error in this case.
                            reason = f" because {example.reason}" * bool(example.reason)
                            if example.raises is BaseException:
                                name = "exception"  # special-case no raises= arg
                            elif not isinstance(example.raises, tuple):
                                name = example.raises.__name__
                            elif len(example.raises) == 1:
                                name = example.raises[0].__name__
                            else:
                                name = (
                                    ", ".join(ex.__name__ for ex in example.raises[:-1])
                                    + f", or {example.raises[-1].__name__}"
                                )
                            vowel = name.upper()[0] in "AEIOU"
                            raise AssertionError(
                                f"Expected a{'n' * vowel} {name} from @example({bits})"
                                f"{reason}, but no exception was raised."
                            )
            except UnsatisfiedAssumption:
                # Odd though it seems, we deliberately support explicit examples that
                # are then rejected by a call to `assume()`.  As well as iterative
                # development, this is rather useful to replay Hypothesis' part of
                # a saved failure when other arguments are supplied by e.g. pytest.
                # See https://github.com/HypothesisWorks/hypothesis/issues/2125
                with contextlib.suppress(StopTest):
                    empty_data.conclude_test(Status.INVALID)
            except BaseException as err:
                # In order to support reporting of multiple failing examples, we yield
                # each of the (report text, error) pairs we find back to the top-level
                # runner.  This also ensures that user-facing stack traces have as few
                # frames of Hypothesis internals as possible.
                err = err.with_traceback(get_trimmed_traceback())

                # One user error - whether misunderstanding or typo - we've seen a few
                # times is to pass strategies to @example() where values are expected.
                # Checking is easy, and false-positives not much of a problem, so:
                if isinstance(err, failure_exceptions_to_catch()) and any(
                    isinstance(arg, SearchStrategy)
                    for arg in example.args + tuple(example.kwargs.values())
                ):
                    new = HypothesisWarning(
                        "The @example() decorator expects to be passed values, but "
                        "you passed strategies instead.  See https://hypothesis."
                        "readthedocs.io/en/latest/reference/api.html#hypothesis"
                        ".example for details."
                    )
                    new.__cause__ = err
                    err = new

                with contextlib.suppress(StopTest):
                    empty_data.conclude_test(Status.INVALID)
                yield ReportableError(fragments_reported, err)
                if (
                    state.settings.report_multiple_bugs
                    and pytest_shows_exceptiongroups
                    and isinstance(err, failure_exceptions_to_catch())
                    and not isinstance(err, skip_exceptions_to_reraise())
                ):
                    continue
                break
            finally:
                if fragments_reported:
                    assert fragments_reported[0].startswith("Falsifying example")
                    fragments_reported[0] = fragments_reported[0].replace(
                        "Falsifying example", "Falsifying explicit example", 1
                    )

                empty_data.freeze()
                if observability_enabled():
                    tc = make_testcase(
                        run_start=state._start_timestamp,
                        property=state.test_identifier,
                        data=empty_data,
                        how_generated="explicit example",
                        representation=state._string_repr,
                        timing=state._timing_features,
                    )
                    deliver_observation(tc)

            if fragments_reported:
                verbose_report(fragments_reported[0].replace("Falsifying", "Trying", 1))
                for f in fragments_reported[1:]:
                    verbose_report(f)


def get_random_for_wrapped_test(test, wrapped_test):
    settings = wrapped_test._hypothesis_internal_use_settings
    wrapped_test._hypothesis_internal_use_generated_seed = None

    if wrapped_test._hypothesis_internal_use_seed is not None:
        return Random(wrapped_test._hypothesis_internal_use_seed)

    if settings.derandomize:
        return Random(int_from_bytes(function_digest(test)))

    if global_force_seed is not None:
        return Random(global_force_seed)

    if threadlocal._hypothesis_global_random is None:  # pragma: no cover
        threadlocal._hypothesis_global_random = Random()
    seed = threadlocal._hypothesis_global_random.getrandbits(128)
    wrapped_test._hypothesis_internal_use_generated_seed = seed
    return Random(seed)


@dataclass(slots=True, frozen=False)
class Stuff:
    selfy: Any
    args: tuple
    kwargs: dict
    given_kwargs: dict


def process_arguments_to_given(
    wrapped_test: Any,
    arguments: Sequence[object],
    kwargs: dict[str, object],
    given_kwargs: dict[str, SearchStrategy],
    params: dict[str, Parameter],
) -> tuple[Sequence[object], dict[str, object], Stuff]:
    selfy = None
    arguments, kwargs = convert_positional_arguments(wrapped_test, arguments, kwargs)

    # If the test function is a method of some kind, the bound object
    # will be the first named argument if there are any, otherwise the
    # first vararg (if any).
    posargs = [p.name for p in params.values() if p.kind is p.POSITIONAL_OR_KEYWORD]
    if posargs:
        selfy = kwargs.get(posargs[0])
    elif arguments:
        selfy = arguments[0]

    # Ensure that we don't mistake mocks for self here.
    # This can cause the mock to be used as the test runner.
    if is_mock(selfy):
        selfy = None

    arguments = tuple(arguments)

    with ensure_free_stackframes():
        for k, s in given_kwargs.items():
            check_strategy(s, name=k)
            s.validate()

    stuff = Stuff(selfy=selfy, args=arguments, kwargs=kwargs, given_kwargs=given_kwargs)

    return arguments, kwargs, stuff


def skip_exceptions_to_reraise():
    """Return a tuple of exceptions meaning 'skip this test', to re-raise.

    This is intended to cover most common test runners; if you would
    like another to be added please open an issue or pull request adding
    it to this function and to tests/cover/test_lazy_import.py
    """
    # This is a set in case any library simply re-exports another's Skip exception
    exceptions = set()
    # We use this sys.modules trick to avoid importing libraries -
    # you can't be an instance of a type from an unimported module!
    # This is fast enough that we don't need to cache the result,
    # and more importantly it avoids possible side-effects :-)
    if "unittest" in sys.modules:
        exceptions.add(sys.modules["unittest"].SkipTest)
    if "_pytest.outcomes" in sys.modules:
        exceptions.add(sys.modules["_pytest.outcomes"].Skipped)
    return tuple(sorted(exceptions, key=str))


def failure_exceptions_to_catch() -> tuple[type[BaseException], ...]:
    """Return a tuple of exceptions meaning 'this test has failed', to catch.

    This is intended to cover most common test runners; if you would
    like another to be added please open an issue or pull request.
    """
    # While SystemExit and GeneratorExit are instances of BaseException, we also
    # expect them to be deterministic - unlike KeyboardInterrupt - and so we treat
    # them as standard exceptions, check for flakiness, etc.
    # See https://github.com/HypothesisWorks/hypothesis/issues/2223 for details.
    exceptions = [Exception, SystemExit, GeneratorExit]
    if "_pytest.outcomes" in sys.modules:
        exceptions.append(sys.modules["_pytest.outcomes"].Failed)
    return tuple(exceptions)


def new_given_signature(original_sig, given_kwargs):
    """Make an updated signature for the wrapped test."""
    return original_sig.replace(
        parameters=[
            p
            for p in original_sig.parameters.values()
            if not (
                p.name in given_kwargs
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            )
        ],
        return_annotation=None,
    )


def default_executor(data, function):
    pass


def get_executor(runner):
    try:
        execute_example = runner.execute_example
    except AttributeError:
        pass
    else:
        return lambda data, function: execute_example(partial(function, data))

    if hasattr(runner, "setup_example") or hasattr(runner, "teardown_example"):
        setup = getattr(runner, "setup_example", None) or (lambda: None)
        teardown = getattr(runner, "teardown_example", None) or (lambda ex: None)

        def execute(data, function):
            token = None
            try:
                token = setup()
                return function(data)
            finally:
                teardown(token)

        return execute

    return default_executor


# This function is a crude solution, a better way of resolving it would probably
# be to rewrite a bunch of exception handlers to use except*.
T = TypeVar("T", bound=BaseException)


def _flatten_group(excgroup: BaseExceptionGroup[T]) -> list[T]:
    found_exceptions: list[T] = []
    for exc in excgroup.exceptions:
        if isinstance(exc, BaseExceptionGroup):
            found_exceptions.extend(_flatten_group(exc))
        else:
            found_exceptions.append(exc)
    return found_exceptions


@contextlib.contextmanager
def unwrap_markers_from_group() -> Generator[None, None, None]:
    try:
        yield
    except BaseExceptionGroup as excgroup:
        _frozen_exceptions, non_frozen_exceptions = excgroup.split(Frozen)

        # group only contains Frozen, reraise the group
        # it doesn't matter what we raise, since any exceptions get disregarded
        # and reraised as StopTest if data got frozen.
        if non_frozen_exceptions is None:
            raise
        # in all other cases they are discarded

        # Can RewindRecursive end up in this group?
        _, user_exceptions = non_frozen_exceptions.split(
            lambda e: isinstance(e, (StopTest, HypothesisException))
        )

        # this might contain marker exceptions, or internal errors, but not frozen.
        if user_exceptions is not None:
            raise

        # single marker exception - reraise it
        flattened_non_frozen_exceptions: list[BaseException] = _flatten_group(
            non_frozen_exceptions
        )
        if len(flattened_non_frozen_exceptions) == 1:
            e = flattened_non_frozen_exceptions[0]
            # preserve the cause of the original exception to not hinder debugging
            # note that __context__ is still lost though
            raise e from e.__cause__

        # multiple marker exceptions. If we re-raise the whole group we break
        # a bunch of logic so ....?
        stoptests, non_stoptests = non_frozen_exceptions.split(StopTest)

        # TODO: stoptest+hypothesisexception ...? Is it possible? If so, what do?

        if non_stoptests:
            # TODO: multiple marker exceptions is easy to produce, but the logic in the
            # engine does not handle it... so we just reraise the first one for now.
            e = _flatten_group(non_stoptests)[0]
            raise e from e.__cause__
        assert stoptests is not None

        # multiple stoptests: raising the one with the lowest testcounter
        raise min(_flatten_group(stoptests), key=lambda s_e: s_e.testcounter)


class StateForActualGivenExecution:
    def __init__(
        self,
        stuff: Stuff,
        test: Callable[..., Any],
        settings: Settings,
        random: Random,
        wrapped_test: Any,
        *,
        thread_overlap: dict[int, bool] | None = None,
    ):
        self.stuff = stuff
        self.test = test
        self.settings = settings
        self.random = random
        self.wrapped_test = wrapped_test
        self.thread_overlap = {} if thread_overlap is None else thread_overlap

        self.test_runner = get_executor(stuff.selfy)
        self.print_given_args = getattr(
            wrapped_test, "_hypothesis_internal_print_given_args", True
        )

        self.last_exception = None
        self.falsifying_examples = ()
        self.ever_executed = False
        self.xfail_example_reprs: set[str] = set()
        self.failed_normally = False
        self.failed_due_to_deadline = False

        self.explain_traces: dict[None | InterestingOrigin, set[Trace]] = defaultdict(
            set
        )
        self._start_timestamp = time.time()
        self._string_repr = ""
        self._timing_features: dict[str, float] = {}

        self._runner: ConjectureRunner | None = None

    @property
    def test_identifier(self) -> str:
        pass

    def _should_trace(self):
        # NOTE: we explicitly support monkeypatching this. Keep the namespace
        # access intact.
        _trace_obs = (
            observability_enabled() and observability.OBSERVABILITY_COLLECT_COVERAGE
        )
        _trace_failure = (
            self.failed_normally
            and not self.failed_due_to_deadline
            and {Phase.shrink, Phase.explain}.issubset(self.settings.phases)
        )
        return _trace_obs or _trace_failure

    def execute_once(
        self,
        data,
        *,
        print_example=False,
        is_final=False,
        expected_failure=None,
        example_kwargs=None,
    ):
        """Run the test function once, using ``data`` as input.

        If the test raises an exception, it will propagate through to the
        caller of this method. Depending on its type, this could represent
        an ordinary test failure, or a fatal error, or a control exception.

        If this method returns normally, the test might have passed, or
        it might have placed ``data`` in an unsuccessful state and then
        swallowed the corresponding control exception.
        """

        self.ever_executed = True

        self._string_repr = ""
        text_repr = None
        if self.settings.deadline is None and not observability_enabled():

            @proxies(self.test)
            def test(*args, **kwargs):
                with unwrap_markers_from_group(), ensure_free_stackframes():
                    return self.test(*args, **kwargs)

        else:

            @proxies(self.test)
            def test(*args, **kwargs):
                arg_drawtime = math.fsum(data.draw_times.values())
                arg_stateful = math.fsum(data._stateful_run_times.values())
                arg_gctime = gc_cumulative_time()
                with unwrap_markers_from_group(), ensure_free_stackframes():
                    start = time.perf_counter()
                    try:
                        result = self.test(*args, **kwargs)
                    finally:
                        finish = time.perf_counter()
                        in_drawtime = math.fsum(data.draw_times.values()) - arg_drawtime
                        in_stateful = (
                            math.fsum(data._stateful_run_times.values()) - arg_stateful
                        )
                        in_gctime = gc_cumulative_time() - arg_gctime
                        runtime = finish - start - in_drawtime - in_stateful - in_gctime
                        self._timing_features = {
                            "execute:test": runtime,
                            "overall:gc": in_gctime,
                            **data.draw_times,
                            **data._stateful_run_times,
                        }

                if (
                    (current_deadline := self.settings.deadline) is not None
                    # we disable the deadline check under concurrent threads, since
                    # cpython may switch away from a thread for arbitrarily long.
                    and not self.thread_overlap.get(threading.get_ident(), False)
                ):
                    if not is_final:
                        current_deadline = (current_deadline // 4) * 5
                    if runtime >= current_deadline.total_seconds():
                        raise DeadlineExceeded(
                            datetime.timedelta(seconds=runtime), self.settings.deadline
                        )
                return result

        def run(data: ConjectureData) -> None:
            # Set up dynamic context needed by a single test run.
            if self.stuff.selfy is not None:
                data.hypothesis_runner = self.stuff.selfy
            # Generate all arguments to the test function.
            args = self.stuff.args
            kwargs = dict(self.stuff.kwargs)
            if example_kwargs is None:
                kw, argslices = context.prep_args_kwargs_from_strategies(
                    self.stuff.given_kwargs
                )
            else:
                kw = example_kwargs
                argslices = {}
            kwargs.update(kw)
            if expected_failure is not None:
                nonlocal text_repr
                text_repr = repr_call(test, args, kwargs)

            if print_example or current_verbosity() >= Verbosity.verbose:
                printer = RepresentationPrinter(context=context)
                if print_example:
                    printer.text("Falsifying example:")
                else:
                    printer.text("Trying example:")

                if self.print_given_args:
                    printer.text(" ")
                    printer.repr_call(
                        test.__name__,
                        args,
                        kwargs,
                        force_split=True,
                        arg_slices=argslices,
                        leading_comment=(
                            "# " + context.data.slice_comments[(0, 0)]
                            if (0, 0) in context.data.slice_comments
                            else None
                        ),
                        avoid_realization=data.provider.avoid_realization,
                    )
                report(printer.getvalue())

            if observability_enabled():
                printer = RepresentationPrinter(context=context)
                printer.repr_call(
                    test.__name__,
                    args,
                    kwargs,
                    force_split=True,
                    arg_slices=argslices,
                    leading_comment=(
                        "# " + context.data.slice_comments[(0, 0)]
                        if (0, 0) in context.data.slice_comments
                        else None
                    ),
                    avoid_realization=data.provider.avoid_realization,
                )
                self._string_repr = printer.getvalue()

            try:
                return test(*args, **kwargs)
            except TypeError as e:
                # If we sampled from a sequence of strategies, AND failed with a
                # TypeError, *AND that exception mentions SearchStrategy*, add a note:
                if (
                    "SearchStrategy" in str(e)
                    and data._sampled_from_all_strategies_elements_message is not None
                ):
                    msg, format_arg = data._sampled_from_all_strategies_elements_message
                    add_note(e, msg.format(format_arg))
                raise
            finally:
                if data._stateful_repr_parts is not None:
                    self._string_repr = "\n".join(data._stateful_repr_parts)

                if observability_enabled():
                    printer = RepresentationPrinter(context=context)
                    for name, value in data._observability_args.items():
                        if name.startswith("generate:Draw "):
                            try:
                                value = data.provider.realize(value)
                            except BackendCannotProceed:  # pragma: no cover
                                value = "<backend failed to realize symbolic>"
                            printer.text(f"\n{name.removeprefix('generate:')}: ")
                            printer.pretty(value)

                    self._string_repr += printer.getvalue()

        # self.test_runner can include the execute_example method, or setup/teardown
        # _example, so it's important to get the PRNG and build context in place first.
        with (
            local_settings(self.settings),
            deterministic_PRNG(),
            BuildContext(
                data, is_final=is_final, wrapped_test=self.wrapped_test
            ) as context,
        ):
            # providers may throw in per_case_context_fn, and we'd like
            # `result` to still be set in these cases.
            result = None
            with data.provider.per_test_case_context_manager():
                # Run the test function once, via the executor hook.
                # In most cases this will delegate straight to `run(data)`.
                result = self.test_runner(data, run)

        # If a failure was expected, it should have been raised already, so
        # instead raise an appropriate diagnostic error.
        if expected_failure is not None:
            exception, traceback = expected_failure
            if isinstance(exception, DeadlineExceeded) and (
                runtime_secs := math.fsum(
                    v
                    for k, v in self._timing_features.items()
                    if k.startswith("execute:")
                )
            ):
                report(
                    "Unreliable test timings! On an initial run, this "
                    f"test took {exception.runtime.total_seconds() * 1000:.2f}ms, "
                    "which exceeded the deadline of "
                    f"{self.settings.deadline.total_seconds() * 1000:.2f}ms, but "
                    f"on a subsequent run it took {runtime_secs * 1000:.2f} ms, "
                    "which did not. If you expect this sort of "
                    "variability in your test timings, consider turning "
                    "deadlines off for this test by setting deadline=None."
                )
            else:
                report("Failed to reproduce exception. Expected: \n" + traceback)
            raise FlakyFailure(
                f"Hypothesis {text_repr} produces unreliable results: "
                "Falsified on the first call but did not on a subsequent one",
                [exception],
            )
        return result

    def _flaky_replay_to_failure(
        self, err: FlakyReplay, context: BaseException
    ) -> FlakyFailure:
        pass

    def _execute_once_for_engine(self, data: ConjectureData) -> None:
        """Wrapper around ``execute_once`` that intercepts test failure
        exceptions and single-test control exceptions, and turns them into
        appropriate method calls to `data` instead.

        This allows the engine to assume that any exception other than
        ``StopTest`` must be a fatal error, and should stop the entire engine.
        """
        pass

    def _deliver_information_message(
        self, *, type: InfoObservationType, title: str, content: str | dict
    ) -> None:
        deliver_observation(
            InfoObservation(
                type=type,
                run_start=self._start_timestamp,
                property=self.test_identifier,
                title=title,
                content=content,
            )
        )

    def run_engine(self):
        """Run the test function many times, on database input and generated
        input, using the Conjecture engine.
        """
        # Tell pytest to omit the body of this function from tracebacks
        __tracebackhide__ = True
        try:
            database_key = self.wrapped_test._hypothesis_internal_database_key
        except AttributeError:
            if global_force_seed is None:
                database_key = function_digest(self.test)
            else:
                database_key = None

        runner = ConjectureRunner(
            self._execute_once_for_engine,
            settings=self.settings,
            random=self.random,
            database_key=database_key,
            thread_overlap=self.thread_overlap,
        )
        self._runner = runner
        # Use the Conjecture engine to run the test function many times
        # on different inputs.
        runner.run()
        note_statistics(runner.statistics)
        if observability_enabled():
            self._deliver_information_message(
                type="info",
                title="Hypothesis Statistics",
                content=describe_statistics(runner.statistics),
            )
            for msg in (
                p if isinstance(p := runner.provider, PrimitiveProvider) else p(None)
            ).observe_information_messages(lifetime="test_function"):
                self._deliver_information_message(**msg)

        if runner.call_count == 0:
            return
        if runner.interesting_examples:
            self.falsifying_examples = sorted(
                runner.interesting_examples.values(),
                key=lambda d: sort_key(d.nodes),
                reverse=True,
            )
        else:
            if runner.valid_examples == 0:
                explanations = []
                # use a somewhat arbitrary cutoff to avoid recommending spurious
                # fixes.
                # eg, a few invalid examples from internal filters when the
                # problem is the user generating large inputs, or a
                # few overruns during internal mutation when the problem is
                # impossible user filters/assumes.
                if runner.invalid_examples > min(20, runner.call_count // 5):
                    explanations.append(
                        f"{runner.invalid_examples} of {runner.call_count} "
                        "examples failed a .filter() or assume() condition. Try "
                        "making your filters or assumes less strict, or rewrite "
                        "using strategy parameters: "
                        "st.integers().filter(lambda x: x > 0) fails less often "
                        "(that is, never) when rewritten as st.integers(min_value=1)."
                    )
                if runner.overrun_examples > min(20, runner.call_count // 5):
                    explanations.append(
                        f"{runner.overrun_examples} of {runner.call_count} "
                        "examples were too large to finish generating; try "
                        "reducing the typical size of your inputs?"
                    )
                rep = get_pretty_function_description(self.test)
                raise Unsatisfiable(
                    f"Unable to satisfy assumptions of {rep}. "
                    f"{' Also, '.join(explanations)}"
                )

        # If we have not traced executions, warn about that now (but only when
        # we'd expect to do so reliably, i.e. on CPython>=3.12)
        if (
            hasattr(sys, "monitoring")
            and not PYPY
            and self._should_trace()
            and not Tracer.can_trace()
        ):  # pragma: no cover
            # actually covered by our tests, but only on >= 3.12
            warnings.warn(
                "avoiding tracing test function because tool id "
                f"{MONITORING_TOOL_ID} is already taken by tool "
                f"{sys.monitoring.get_tool(MONITORING_TOOL_ID)}.",
                HypothesisWarning,
                stacklevel=3,
            )

        if not self.falsifying_examples:
            return
        elif not (self.settings.report_multiple_bugs and pytest_shows_exceptiongroups):
            # Pretend that we only found one failure, by discarding the others.
            del self.falsifying_examples[:-1]

        # The engine found one or more failures, so we need to reproduce and
        # report them.

        errors_to_report = []

        report_lines = describe_targets(runner.best_observed_targets)
        if report_lines:
            report_lines.append("")

        explanations = explanatory_lines(self.explain_traces, self.settings)
        for falsifying_example in self.falsifying_examples:
            fragments = []

            ran_example = runner.new_conjecture_data(
                falsifying_example.choices, max_choices=len(falsifying_example.choices)
            )
            ran_example.slice_comments = falsifying_example.slice_comments
            tb = None
            origin = None
            assert falsifying_example.expected_exception is not None
            assert falsifying_example.expected_traceback is not None
            try:
                with with_reporter(fragments.append):
                    self.execute_once(
                        ran_example,
                        print_example=True,
                        is_final=True,
                        expected_failure=(
                            falsifying_example.expected_exception,
                            falsifying_example.expected_traceback,
                        ),
                    )
            except StopTest as e:
                # Link the expected exception from the first run. Not sure
                # how to access the current exception, if it failed
                # differently on this run. In fact, in the only known
                # reproducer, the StopTest is caused by OVERRUN before the
                # test is even executed. Possibly because all initial examples
                # failed until the final non-traced replay, and something was
                # exhausted? Possibly a FIXME, but sufficiently weird to
                # ignore for now.
                err = FlakyFailure(
                    "Inconsistent results: An example failed on the "
                    "first run but now succeeds (or fails with another "
                    "error, or is for some reason not runnable).",
                    # (note: e is a BaseException)
                    [falsifying_example.expected_exception or e],
                )
                errors_to_report.append(ReportableError(fragments, err))
            except UnsatisfiedAssumption as e:  # pragma: no cover  # ironically flaky
                err = FlakyFailure(
                    "Unreliable assumption: An example which satisfied "
                    "assumptions on the first run now fails it.",
                    [e],
                )
                errors_to_report.append(ReportableError(fragments, err))
            except BaseException as e:
                # If we have anything for explain-mode, this is the time to report.
                fragments.extend(explanations[falsifying_example.interesting_origin])
                error_with_tb = e.with_traceback(get_trimmed_traceback())
                errors_to_report.append(ReportableError(fragments, error_with_tb))
                tb = format_exception(e, get_trimmed_traceback(e))
                origin = InterestingOrigin.from_exception(e)
            else:
                # execute_once() will always raise either the expected error, or Flaky.
                raise NotImplementedError("This should be unreachable")
            finally:
                ran_example.freeze()
                if observability_enabled():
                    # log our observability line for the final failing example
                    tc = make_testcase(
                        run_start=self._start_timestamp,
                        property=self.test_identifier,
                        data=ran_example,
                        how_generated="minimal failing example",
                        representation=self._string_repr,
                        arguments=ran_example._observability_args,
                        timing=self._timing_features,
                        coverage=None,  # Not recorded when we're replaying the MFE
                        status="passed" if sys.exc_info()[0] else "failed",
                        status_reason=str(origin or "unexpected/flaky pass"),
                        metadata={"traceback": tb},
                    )
                    deliver_observation(tc)

                # Whether or not replay actually raised the exception again, we want
                # to print the reproduce_failure decorator for the failing example.
                if self.settings.print_blob:
                    fragments.append(
                        "\nYou can reproduce this example by temporarily adding "
                        f"{reproduction_decorator(falsifying_example.choices)} "
                        "as a decorator on your test case"
                    )

        _raise_to_user(
            errors_to_report,
            self.settings,
            report_lines,
            # A backend might report a failure and then report verified afterwards,
            # which is to be interpreted as "there are no more failures *other
            # than what we already reported*". Do not report this as unsound.
            unsound_backend=(
                runner._verified_by_backend
                if runner._verified_by_backend and not runner._backend_found_failure
                else None
            ),
        )


def _simplify_explicit_errors(errors: list[ReportableError]) -> list[ReportableError]:
    """
    Group explicit example errors by their InterestingOrigin, keeping only the
    simplest one, and adding a note of how many other examples failed with the same
    error.
    """
    by_origin: dict[InterestingOrigin, list[ReportableError]] = defaultdict(list)
    for error in errors:
        origin = InterestingOrigin.from_exception(error.exception)
        by_origin[origin].append(error)

    result = []
    for group in by_origin.values():
        if len(group) == 1:
            result.append(group[0])
        else:
            # Sort by shortlex of representation (first fragment)
            def shortlex_key(error):
                pass

            sorted_group = sorted(group, key=shortlex_key)
            simplest = sorted_group[0]
            other_count = len(group) - 1
            add_note(
                simplest.exception,
                f"(note: {other_count} other explicit example{'s' * (other_count > 1)} "
                "also failed with this error; use Verbosity.verbose to view)",
            )
            result.append(simplest)

    return result


def _raise_to_user(
    errors_to_report, settings, target_lines, trailer="", *, unsound_backend=None
):
    """Helper function for attaching notes and grouping multiple errors."""
    failing_prefix = "Falsifying example: "
    ls = []
    for error in errors_to_report:
        for note in error.fragments:
            add_note(error.exception, note)
            if note.startswith(failing_prefix):
                ls.append(note.removeprefix(failing_prefix))
    if current_pytest_item.value:
        current_pytest_item.value._hypothesis_failing_examples = ls

    if len(errors_to_report) == 1:
        the_error_hypothesis_found = errors_to_report[0].exception
    else:
        assert errors_to_report
        the_error_hypothesis_found = BaseExceptionGroup(
            f"Hypothesis found {len(errors_to_report)} distinct failures{trailer}.",
            [error.exception for error in errors_to_report],
        )

    if settings.verbosity >= Verbosity.normal:
        for line in target_lines:
            add_note(the_error_hypothesis_found, line)

    if unsound_backend:
        add_note(
            the_error_hypothesis_found,
            f"backend={unsound_backend!r} claimed to verify this test passes - "
            "please send them a bug report!",
        )

    raise the_error_hypothesis_found


@contextlib.contextmanager
def fake_subTest(self, msg=None, **__):
    """Monkeypatch for `unittest.TestCase.subTest` during `@given`.

    If we don't patch this out, each failing example is reported as a
    separate failing test by the unittest test runner, which is
    obviously incorrect. We therefore replace it for the duration with
    this version.
    """
    pass


@dataclass(slots=False, frozen=False)
class HypothesisHandle:
    """This object is provided as the .hypothesis attribute on @given tests.

    Downstream users can reassign its attributes to insert custom logic into
    the execution of each case, for example by converting an async into a
    sync function.

    This must be an attribute of an attribute, because reassignment of a
    first-level attribute would not be visible to Hypothesis if the function
    had been decorated before the assignment.

    See https://github.com/HypothesisWorks/hypothesis/issues/1257 for more
    information.
    """

    inner_test: Any
    _get_fuzz_target: Any
    _given_kwargs: Any

    @property
    def fuzz_one_input(
        self,
    ) -> Callable[[bytes | bytearray | memoryview | BinaryIO], bytes | None]:
        """
        Run the test as a fuzz target, driven with the ``buffer`` of bytes.

        Depending on the passed ``buffer`` one of three things will happen:

        * If the bytestring was invalid, for example because it was too short or was
          filtered out by |assume| or |.filter|, |fuzz_one_input| returns ``None``.
        * If the bytestring was valid and the test passed, |fuzz_one_input| returns
          a canonicalised and pruned bytestring which will replay that test case.
          This is provided as an option to improve the performance of mutating
          fuzzers, but can safely be ignored.
        * If the test *failed*, i.e. raised an exception, |fuzz_one_input| will
          add the pruned buffer to :ref:`the Hypothesis example database <database>`
          and then re-raise that exception.  All you need to do to reproduce,
          minimize, and de-duplicate all the failures found via fuzzing is run
          your test suite!

        To reduce the performance impact of database writes, |fuzz_one_input| only
        records failing inputs which would be valid shrinks for a known failure -
        meaning writes are somewhere between constant and log(N) rather than linear
        in runtime.  However, this tracking only works within a persistent fuzzing
        process; for forkserver fuzzers we recommend ``database=None`` for the main
        run, and then replaying with a database enabled if you need to analyse
        failures.

        Note that the interpretation of both input and output bytestrings is
        specific to the exact version of Hypothesis you are using and the strategies
        given to the test, just like the :ref:`database <database>` and
        |@reproduce_failure|.

        Interaction with |@settings|
        ----------------------------

        |fuzz_one_input| uses just enough of Hypothesis' internals to drive your
        test function with a bytestring, and most settings therefore have no effect
        in this mode.  We recommend running your tests the usual way before fuzzing
        to get the benefits of health checks, as well as afterwards to replay,
        shrink, deduplicate, and report whatever errors were discovered.

        * |settings.database| *is* used by |fuzz_one_input| - adding failures to
          the database to be replayed when
          you next run your tests is our preferred reporting mechanism and response
          to `the 'fuzzer taming' problem <https://blog.regehr.org/archives/925>`__.
        * |settings.verbosity| and |settings.stateful_step_count| work as usual.
        * The |~settings.deadline|, |~settings.derandomize|, |~settings.max_examples|,
          |~settings.phases|, |~settings.print_blob|, |~settings.report_multiple_bugs|,
          and |~settings.suppress_health_check| settings do not affect |fuzz_one_input|.

        Example Usage
        -------------

        .. code-block:: python

            @given(st.text())
            def test_foo(s): ...

            # This is a traditional fuzz target - call it with a bytestring,
            # or a binary IO object, and it runs the test once.
            fuzz_target = test_foo.hypothesis.fuzz_one_input

            # For example:
            fuzz_target(b"\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00")
            fuzz_target(io.BytesIO(b"\\x01"))

        .. tip::

            If you expect to discover many failures while using |fuzz_one_input|,
            consider wrapping your database with |BackgroundWriteDatabase|, for
            low-overhead writes of failures.

        .. tip::

            | Want an integrated workflow for your team's local tests, CI, and continuous fuzzing?
            | Use `HypoFuzz <https://hypofuzz.com/>`__ to fuzz your whole test suite, and find more bugs with the same tests!

        .. seealso::

            See also the :doc:`/how-to/external-fuzzers` how-to.
        """
        pass


@overload
def given(
    _: EllipsisType, /
) -> Callable[
    [Callable[..., Coroutine[Any, Any, None] | None]], Callable[[], None]
]:  # pragma: no cover
    ...


@overload
def given(
    *_given_arguments: SearchStrategy[Any],
) -> Callable[
    [Callable[..., Coroutine[Any, Any, None] | None]], Callable[..., None]
]:  # pragma: no cover
    ...


@overload
def given(
    **_given_kwargs: SearchStrategy[Any] | EllipsisType,
) -> Callable[
    [Callable[..., Coroutine[Any, Any, None] | None]], Callable[..., None]
]:  # pragma: no cover
    ...


def given(
    *_given_arguments: SearchStrategy[Any] | EllipsisType,
    **_given_kwargs: SearchStrategy[Any] | EllipsisType,
) -> Callable[[Callable[..., Coroutine[Any, Any, None] | None]], Callable[..., None]]:
    """
    The |@given| decorator turns a function into a Hypothesis test. This is the
    main entry point to Hypothesis.

    .. seealso::

        See also the :doc:`/tutorial/introduction` tutorial, which introduces
        defining Hypothesis tests with |@given|.

    .. _given-arguments:

    Arguments to ``@given``
    -----------------------

    Arguments to |@given| may be either positional or keyword arguments:

    .. code-block:: python

        @given(st.integers(), st.floats())
        def test_one(x, y):
            pass

        @given(x=st.integers(), y=st.floats())
        def test_two(x, y):
            pass

    If using keyword arguments, the arguments may appear in any order, as with
    standard Python functions:

    .. code-block:: python

        # different order, but still equivalent to before
        @given(y=st.floats(), x=st.integers())
        def test(x, y):
            assert isinstance(x, int)
            assert isinstance(y, float)

    If |@given| is provided fewer positional arguments than the decorated test,
    the test arguments are filled in on the right side, leaving the leftmost
    positional arguments unfilled:

    .. code-block:: python

        @given(st.integers(), st.floats())
        def test(manual_string, y, z):
            assert manual_string == "x"
            assert isinstance(y, int)
            assert isinstance(z, float)

        # `test` is now a callable which takes one argument `manual_string`

        test("x")
        # or equivalently:
        test(manual_string="x")

    The reason for this "from the right" behavior is to support using |@given|
    with instance methods, by automatically passing through ``self``:

    .. code-block:: python

        class MyTest(TestCase):
            @given(st.integers())
            def test(self, x):
                assert isinstance(self, MyTest)
                assert isinstance(x, int)

    If (and only if) using keyword arguments, |@given| may be combined with
    ``**kwargs`` or ``*args``:

    .. code-block:: python

        @given(x=integers(), y=integers())
        def test(x, **kwargs):
            assert "y" in kwargs

        @given(x=integers(), y=integers())
        def test(x, *args, **kwargs):
            assert args == ()
            assert "x" not in kwargs
            assert "y" in kwargs

    It is an error to:

    * Mix positional and keyword arguments to |@given|.
    * Use |@given| with a function that has a default value for an argument.
    * Use |@given| with positional arguments with a function that uses ``*args``,
      ``**kwargs``, or keyword-only arguments.

    The function returned by given has all the same arguments as the original
    test, minus those that are filled in by |@given|. See the :ref:`notes on
    framework compatibility <framework-compatibility>` for how this interacts
    with features of other testing libraries, such as :pypi:`pytest` fixtures.
    """

    if currently_in_test_context():
        fail_health_check(
            Settings(),
            "Nesting @given tests results in quadratic generation and shrinking "
            "behavior, and can usually be more cleanly expressed by replacing the "
            "inner function with an st.data() parameter on the outer @given."
            "\n\n"
            "If it is difficult or impossible to refactor this test to remove the "
            "nested @given, you can disable this health check with "
            "@settings(suppress_health_check=[HealthCheck.nested_given]) on the "
            "outer @given. See "
            "https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.HealthCheck "
            "for details.",
            HealthCheck.nested_given,
        )

    def run_test_as_given(test):
        pass

    return run_test_as_given


def find(
    specifier: SearchStrategy[Ex],
    condition: Callable[[Any], bool],
    *,
    settings: Settings | None = None,
    random: Random | None = None,
    database_key: bytes | None = None,
) -> Ex:
    """Returns the minimal example from the given strategy ``specifier`` that
    matches the predicate function ``condition``."""
    pass
