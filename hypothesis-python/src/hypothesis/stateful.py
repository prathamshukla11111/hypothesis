# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""This module provides support for a stateful style of testing, where tests
attempt to find a sequence of operations that cause a breakage rather than just
a single value.

Notably, the set of steps available at any point may depend on the
execution to date.
"""

import collections
import dataclasses
import inspect
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from io import StringIO
from time import perf_counter
from typing import Any, ClassVar, TypeVar, overload
from unittest import TestCase

from hypothesis import strategies as st
from hypothesis._settings import HealthCheck, Verbosity, settings as Settings
from hypothesis.control import _current_build_context, current_build_context
from hypothesis.core import TestFunc, given
from hypothesis.errors import (
    FlakyStrategyDefinition,
    InvalidArgument,
    InvalidDefinition,
)
from hypothesis.internal.compat import add_note, batched
from hypothesis.internal.conjecture.engine import BUFFER_SIZE
from hypothesis.internal.conjecture.junkdrawer import gc_cumulative_time
from hypothesis.internal.conjecture.utils import calc_label_from_name
from hypothesis.internal.healthcheck import fail_health_check
from hypothesis.internal.observability import observability_enabled
from hypothesis.internal.reflection import (
    function_digest,
    get_pretty_function_description,
    nicerepr,
    proxies,
)
from hypothesis.internal.validation import check_type
from hypothesis.reporting import current_verbosity, report
from hypothesis.strategies._internal.featureflags import FeatureStrategy
from hypothesis.strategies._internal.strategies import (
    Ex,
    OneOfStrategy,
    SearchStrategy,
    check_strategy,
)
from hypothesis.utils.deprecation import note_deprecation
from hypothesis.vendor.pretty import RepresentationPrinter

T = TypeVar("T")
STATE_MACHINE_RUN_LABEL = calc_label_from_name("another state machine step")


def _is_singleton(obj: object) -> bool:
    """
    Returns True if two separately created instances of v will have the same id
    (due to interning).
    """
    pass


class _OmittedArgument:
    """Sentinel class to prevent overlapping overloads in type hints. See comments
    above the overloads of @rule."""


class TestCaseProperty:  # pragma: no cover
    def __get__(self, obj, typ=None):
        if obj is not None:
            typ = type(obj)
        return typ._to_test_case()

    def __set__(self, obj, value):
        raise AttributeError("Cannot set TestCase")

    def __delete__(self, obj):
        raise AttributeError("Cannot delete TestCase")


def get_state_machine_test(
    state_machine_factory, *, settings=None, _min_steps=0, _flaky_state=None
):
    # This function is split out from run_state_machine_as_test so that
    # HypoFuzz can get and call the test function directly.
    pass


def run_state_machine_as_test(state_machine_factory, *, settings=None, _min_steps=0):
    """Run a state machine definition as a test, either silently doing nothing
    or printing a minimal breaking program and raising an exception.

    state_machine_factory is anything which returns an instance of
    RuleBasedStateMachine when called with no arguments - it can be a class or a
    function. settings will be used to control the execution of the test.
    """
    pass


class StateMachineMeta(type):
    def __setattr__(cls, name, value):
        if name == "settings" and isinstance(value, Settings):
            descr = f"settings({value.show_changed()})"
            raise AttributeError(
                f"Assigning {cls.__name__}.settings = {descr} does nothing. Assign "
                f"to {cls.__name__}.TestCase.settings, or use @{descr} as a decorator "
                f"on the {cls.__name__} class."
            )
        return super().__setattr__(name, value)


@dataclass(slots=True, frozen=True)
class _SetupState:
    rules: list["Rule"]
    invariants: list["Invariant"]
    initializers: list["Rule"]


class RuleBasedStateMachine(metaclass=StateMachineMeta):
    """A RuleBasedStateMachine gives you a structured way to define state machines.

    The idea is that a state machine carries the system under test and some supporting
    data. This data can be stored in instance variables or
    divided into Bundles. The state machine has a set of rules which may read data
    from bundles (or just from normal strategies), push data onto
    bundles, change the state of the machine, or verify properties.
    At any given point a random applicable rule will be executed.
    """

    _setup_state_per_class: ClassVar[dict[type, _SetupState]] = {}

    def __init__(self) -> None:
        setup_state = self.setup_state()
        if not setup_state.rules:
            raise InvalidDefinition(
                f"State machine {type(self).__name__} defines no rules"
            )

        if isinstance(s := vars(type(self)).get("settings"), Settings):
            tname = type(self).__name__
            descr = f"settings({s.show_changed()})"
            raise InvalidDefinition(
                f"Assigning settings = {descr} as a class attribute does nothing. "
                f"Assign to {tname}.TestCase.settings, or use @{descr} as a decorator "
                f"on the {tname} class."
            )

        self.rules = setup_state.rules
        self.invariants = setup_state.invariants
        # copy since we pop from this as we run initialize rules.
        self._initialize_rules_to_run = setup_state.initializers.copy()

        self.bundles: dict[str, list] = {}
        self.names_counters: collections.Counter = collections.Counter()
        self.names_list: list[str] = []
        self.names_to_values: dict[str, Any] = {}
        self.__stream = StringIO()
        self.__printer = RepresentationPrinter(
            self.__stream, context=_current_build_context.value
        )
        self._rules_strategy = RuleStrategy(self)

    def _pretty_print(self, value):
        pass

    def __repr__(self):
        return f"{type(self).__name__}({nicerepr(self.bundles)})"

    def _new_name(self, target):
        pass

    def _last_names(self, n: int) -> list[str]:
        pass

    def bundle(self, name):
        return self.bundles.setdefault(name, [])

    @classmethod
    def setup_state(cls):
        try:
            return cls._setup_state_per_class[cls]
        except KeyError:
            pass

        rules: list[Rule] = []
        initializers: list[Rule] = []
        invariants: list[Invariant] = []

        for _name, f in inspect.getmembers(cls):
            rule = getattr(f, RULE_MARKER, None)
            initializer = getattr(f, INITIALIZE_RULE_MARKER, None)
            invariant = getattr(f, INVARIANT_MARKER, None)
            if rule is not None:
                rules.append(rule)
            if initializer is not None:
                initializers.append(initializer)
            if invariant is not None:
                invariants.append(invariant)

            if (
                getattr(f, PRECONDITIONS_MARKER, None) is not None
                and rule is None
                and invariant is None
            ):
                raise InvalidDefinition(
                    f"{_rule_qualname(f)} has been decorated with @precondition, "
                    "but not @rule (or @invariant), which is not allowed. A "
                    "precondition must be combined with a rule or an invariant, "
                    "since it has no effect alone."
                )

        state = _SetupState(
            rules=rules, initializers=initializers, invariants=invariants
        )
        cls._setup_state_per_class[cls] = state
        return state

    def _repr_step(self, rule: "Rule", data: Any, result: Any) -> str:
        pass

    def _add_results_to_targets(self, targets, results):
        # Note, the assignment order here is reflected in _repr_step
        pass

    def check_invariants(self, settings, output, runtimes):
        for invar in self.invariants:
            if self._initialize_rules_to_run and not invar.check_during_init:
                continue
            if not all(precond(self) for precond in invar.preconditions):
                continue
            name = invar.function.__name__
            if (
                current_build_context().is_final
                or settings.verbosity >= Verbosity.debug
                or observability_enabled()
            ):
                output(f"state.{name}()")
            start = perf_counter()
            result = invar.function(self)
            runtimes[f"execute:invariant:{name}"] += perf_counter() - start
            if result is not None:
                fail_health_check(
                    settings,
                    "The return value of an @invariant is always ignored, but "
                    f"{invar.function.__qualname__} returned {result!r} "
                    "instead of None",
                    HealthCheck.return_value,
                )

    def teardown(self):
        """Called after a run has finished executing to clean up any necessary
        state.

        Does nothing by default.
        """

    TestCase = TestCaseProperty()

    @classmethod
    @lru_cache
    def _to_test_case(cls):
        pass


@dataclass(slots=True, frozen=False)
class Rule:
    targets: Any
    function: Any
    arguments: Any
    preconditions: Any
    bundles: tuple["Bundle", ...] = field(init=False)
    _cached_hash: int | None = field(init=False, default=None)
    _cached_repr: str | None = field(init=False, default=None)
    arguments_strategies: dict[Any, Any] = field(init=False, default_factory=dict)

    def __post_init__(self):
        bundles = []
        for k, v in sorted(self.arguments.items()):
            assert not isinstance(v, BundleReferenceStrategy)
            if isinstance(v, Bundle):
                bundles.append(v)
                consume = isinstance(v, BundleConsumer)
                v = BundleReferenceStrategy(v.name, consume=consume)
            self.arguments_strategies[k] = v
        self.bundles = tuple(bundles)

    def __repr__(self) -> str:
        if self._cached_repr is None:
            bits = [
                f"{field.name}="
                f"{get_pretty_function_description(getattr(self, field.name))}"
                for field in dataclasses.fields(self)
                if getattr(self, field.name)
            ]
            self._cached_repr = f"{self.__class__.__name__}({', '.join(bits)})"
        return self._cached_repr

    def __hash__(self):
        # sampled_from uses hash in calc_label, and we want this to be fast when
        # sampling stateful rules, so we cache here.
        if self._cached_hash is None:
            self._cached_hash = hash(
                (
                    self.targets,
                    self.function,
                    tuple(self.arguments.items()),
                    self.preconditions,
                    self.bundles,
                )
            )
        return self._cached_hash


self_strategy = st.runner()


class BundleReferenceStrategy(SearchStrategy):
    def __init__(self, name: str, *, consume: bool = False):
        super().__init__()
        self.name = name
        self.consume = consume

    def do_draw(self, data):
        machine = data.draw(self_strategy)
        bundle = machine.bundle(self.name)
        if not bundle:
            data.mark_invalid(f"Cannot draw from empty bundle {self.name!r}")
        # Shrink towards the right rather than the left. This makes it easier
        # to delete data generated earlier, as when the error is towards the
        # end there can be a lot of hard to remove padding.
        position = data.draw_integer(0, len(bundle) - 1, shrink_towards=len(bundle))
        if self.consume:
            return bundle.pop(position)  # pragma: no cover  # coverage is flaky here
        else:
            return bundle[position]


class Bundle(SearchStrategy[Ex]):
    """A collection of values for use in stateful testing.

    Bundles are a kind of strategy where values can be added by rules,
    and (like any strategy) used as inputs to future rules.

    The ``name`` argument they are passed is the they are referred to
    internally by the state machine; no two bundles may have
    the same name. It is idiomatic to use the attribute
    being assigned to as the name of the Bundle::

        class MyStateMachine(RuleBasedStateMachine):
            keys = Bundle("keys")

    Bundles can contain the same value more than once; this becomes
    relevant when using :func:`~hypothesis.stateful.consumes` to remove
    values again.

    If the ``consume`` argument is set to True, then all values that are
    drawn from this bundle will be consumed (as above) when requested.
    """

    def __init__(
        self, name: str, *, consume: bool = False, draw_references: bool = True
    ) -> None:
        super().__init__()
        self.name = name
        self.__reference_strategy = BundleReferenceStrategy(name, consume=consume)
        self.draw_references = draw_references

    def do_draw(self, data):
        machine = data.draw(self_strategy)
        reference = data.draw(self.__reference_strategy)
        return machine.names_to_values[reference.name]

    def __repr__(self):
        consume = self.__reference_strategy.consume
        if consume is False:
            return f"Bundle(name={self.name!r})"
        return f"Bundle(name={self.name!r}, {consume=})"

    def calc_is_empty(self, recur):
        # We assume that a bundle will grow over time
        pass

    def is_currently_empty(self, data):
        # ``self_strategy`` is an instance of the ``st.runner()`` strategy.
        # Hence drawing from it only returns the current state machine without
        # modifying the underlying choice sequence.
        machine = data.draw(self_strategy)
        return not bool(machine.bundle(self.name))

    def flatmap(self, expand):
        if self.draw_references:
            return type(self)(
                self.name,
                consume=self.__reference_strategy.consume,
                draw_references=False,
            ).flatmap(expand)
        return super().flatmap(expand)

    def __hash__(self):
        # Making this hashable means we hit the fast path of "everything is
        # hashable" in st.sampled_from label calculation when sampling which rule
        # to invoke next.

        # Mix in "Bundle" for collision resistance
        return hash(("Bundle", self.name))


class BundleConsumer(Bundle[Ex]):
    def __init__(self, bundle: Bundle[Ex]) -> None:
        super().__init__(bundle.name, consume=True)


def consumes(bundle: Bundle[Ex]) -> SearchStrategy[Ex]:
    """When introducing a rule in a RuleBasedStateMachine, this function can
    be used to mark bundles from which each value used in a step with the
    given rule should be removed. This function returns a strategy object
    that can be manipulated and combined like any other.

    For example, a rule declared with

    ``@rule(value1=b1, value2=consumes(b2), value3=lists(consumes(b3)))``

    will consume a value from Bundle ``b2`` and several values from Bundle
    ``b3`` to populate ``value2`` and ``value3`` each time it is executed.
    """
    pass


@dataclass(slots=True, frozen=True)
class MultipleResults(Iterable[Ex]):
    values: tuple[Ex, ...]

    def __iter__(self):
        return iter(self.values)


def multiple(*args: T) -> MultipleResults[T]:
    """This function can be used to pass multiple results to the target(s) of
    a rule. Just use ``return multiple(result1, result2, ...)`` in your rule.

    It is also possible to use ``return multiple()`` with no arguments in
    order to end a rule without passing any result.
    """
    pass


def _convert_targets(targets, target):
    """Single validator and converter for target arguments."""
    if target is not None:
        if targets:
            raise InvalidArgument(
                f"Passing both targets={targets!r} and target={target!r} is "
                f"redundant - pass targets={(*targets, target)!r} instead."
            )
        targets = (target,)

    converted_targets = []
    for t in targets:
        if not isinstance(t, Bundle):
            msg = "Got invalid target %r of type %r, but all targets must be Bundles."
            if isinstance(t, OneOfStrategy):
                msg += (
                    "\nIt looks like you passed `one_of(a, b)` or `a | b` as "
                    "a target.  You should instead pass `targets=(a, b)` to "
                    "add the return value of this rule to both the `a` and "
                    "`b` bundles, or define a rule for each target if it "
                    "should be added to exactly one."
                )
            raise InvalidArgument(msg % (t, type(t)))
        while isinstance(t, Bundle):
            if isinstance(t, BundleConsumer):
                note_deprecation(
                    f"Using consumes({t.name}) doesn't makes sense in this context.  "
                    "This will be an error in a future version of Hypothesis.",
                    since="2021-09-08",
                    has_codemod=False,
                    stacklevel=2,
                )
            t = t.name
        converted_targets.append(t)
    return tuple(converted_targets)


RULE_MARKER = "hypothesis_stateful_rule"
INITIALIZE_RULE_MARKER = "hypothesis_stateful_initialize_rule"
PRECONDITIONS_MARKER = "hypothesis_stateful_preconditions"
INVARIANT_MARKER = "hypothesis_stateful_invariant"


_RuleType = Callable[..., MultipleResults[Ex] | Ex]
_RuleWrapper = Callable[[_RuleType[Ex]], _RuleType[Ex]]


def _rule_qualname(f: Any) -> str:
    # we define rules / invariants / initializes inside of wrapper functions, which
    # makes f.__qualname__ look like:
    #   test_precondition.<locals>.BadStateMachine.has_precondition_but_no_rule
    # which is not ideal. This function returns just
    #   BadStateMachine.has_precondition_but_no_rule
    # instead.
    return f.__qualname__.rsplit("<locals>.")[-1]


# We cannot exclude `target` or `targets` from any of these signatures because
# otherwise they would be matched against the `kwargs`, either leading to
# overlapping overloads of incompatible return types, or a concrete
# implementation that does not accept all overloaded variant signatures.
# Although it is possible to reorder the variants to fix the former, it will
# always lead to the latter, as then the omitted parameter could be typed as
# a `SearchStrategy`, which the concrete implementation does not accept.
#
# Omitted `targets` parameters, where the default value is used, are typed with
# a special `_OmittedArgument` type. We cannot type them as `tuple[()]`, because
# `tuple[()]` is a subtype of `Sequence[Bundle[Ex]]`, leading to signature
# overlaps with incompatible return types. The `_OmittedArgument` type will never be
# encountered at runtime, and exists solely to annotate the default of `targets`.
# PEP 661 (Sentinel Values) might provide a more elegant alternative in the future.
#
# We could've also annotated `targets` as `tuple[_OmittedArgument]`, but then when
# both `target` and `targets` are provided, mypy describes the type error as an
# invalid argument type for `targets` (expected `tuple[_OmittedArgument]`, got ...).
# By annotating it as a bare `_OmittedArgument` type, mypy's error will warn that
# there is no overloaded signature matching the call, which is more descriptive.
#
# When `target` xor `targets` is provided, the function to decorate must return
# a value whose type matches the one stored in the bundle. When neither are
# provided, the function to decorate must return nothing. There is no variant
# for providing `target` and `targets`, as these parameters are mutually exclusive.
@overload
def rule(
    *,
    targets: Sequence[Bundle[Ex]],
    target: None = ...,
    **kwargs: SearchStrategy,
) -> _RuleWrapper[Ex]:  # pragma: no cover
    ...


@overload
def rule(
    *, target: Bundle[Ex], targets: _OmittedArgument = ..., **kwargs: SearchStrategy
) -> _RuleWrapper[Ex]:  # pragma: no cover
    ...


@overload
def rule(
    *,
    target: None = ...,
    targets: _OmittedArgument = ...,
    **kwargs: SearchStrategy,
) -> Callable[[Callable[..., None]], Callable[..., None]]:  # pragma: no cover
    ...


def rule(
    *,
    targets: Sequence[Bundle[Ex]] | _OmittedArgument = (),
    target: Bundle[Ex] | None = None,
    **kwargs: SearchStrategy,
) -> _RuleWrapper[Ex] | Callable[[Callable[..., None]], Callable[..., None]]:
    """Decorator for RuleBasedStateMachine. Any Bundle present in ``target`` or
    ``targets`` will define where the end result of this function should go. If
    both are empty then the end result will be discarded.

    ``target`` must be a Bundle, or if the result should be replicated to multiple
    bundles you can pass a tuple of them as the ``targets`` argument.
    It is invalid to use both arguments for a single rule.  If the result
    should go to exactly one of several bundles, define a separate rule for
    each case.

    kwargs then define the arguments that will be passed to the function
    invocation. If their value is a Bundle, or if it is ``consumes(b)``
    where ``b`` is a Bundle, then values that have previously been produced
    for that bundle will be provided. If ``consumes`` is used, the value
    will also be removed from the bundle.

    Any other kwargs should be strategies and values from them will be
    provided.
    """
    converted_targets = _convert_targets(targets, target)
    for k, v in kwargs.items():
        check_strategy(v, name=k)

    def accept(f):
        if getattr(f, INVARIANT_MARKER, None):
            raise InvalidDefinition(
                f"{_rule_qualname(f)} is used with both @rule and @invariant, "
                "which is not allowed. A function may be either a rule or an "
                "invariant, but not both."
            )
        existing_rule = getattr(f, RULE_MARKER, None)
        existing_initialize_rule = getattr(f, INITIALIZE_RULE_MARKER, None)
        if existing_rule is not None:
            raise InvalidDefinition(
                f"{_rule_qualname(f)} has been decorated with @rule twice, which is "
                "not allowed."
            )
        if existing_initialize_rule is not None:
            raise InvalidDefinition(
                f"{_rule_qualname(f)} has been decorated with both @rule and "
                "@initialize, which is not allowed."
            )

        preconditions = getattr(f, PRECONDITIONS_MARKER, ())
        rule = Rule(
            targets=converted_targets,
            arguments=kwargs,
            function=f,
            preconditions=preconditions,
        )

        @proxies(f)
        def rule_wrapper(*args, **kwargs):
            pass

        setattr(rule_wrapper, RULE_MARKER, rule)
        return rule_wrapper

    return accept


# See also comments of `rule`'s overloads.
@overload
def initialize(
    *,
    targets: Sequence[Bundle[Ex]],
    target: None = ...,
    **kwargs: SearchStrategy,
) -> _RuleWrapper[Ex]:  # pragma: no cover
    ...


@overload
def initialize(
    *, target: Bundle[Ex], targets: _OmittedArgument = ..., **kwargs: SearchStrategy
) -> _RuleWrapper[Ex]:  # pragma: no cover
    ...


@overload
def initialize(
    *,
    target: None = ...,
    targets: _OmittedArgument = ...,
    **kwargs: SearchStrategy,
) -> Callable[[Callable[..., None]], Callable[..., None]]:  # pragma: no cover
    ...


def initialize(
    *,
    targets: Sequence[Bundle[Ex]] | _OmittedArgument = (),
    target: Bundle[Ex] | None = None,
    **kwargs: SearchStrategy,
) -> _RuleWrapper[Ex] | Callable[[Callable[..., None]], Callable[..., None]]:
    """Decorator for RuleBasedStateMachine.

    An initialize decorator behaves like a rule, but all ``@initialize()`` decorated
    methods will be called before any ``@rule()`` decorated methods, in an arbitrary
    order.  Each ``@initialize()`` method will be called exactly once per run, unless
    one raises an exception - after which only the ``.teardown()`` method will be run.
    ``@initialize()`` methods may not have preconditions.
    """
    converted_targets = _convert_targets(targets, target)
    for k, v in kwargs.items():
        check_strategy(v, name=k)

    def accept(f):
        if getattr(f, INVARIANT_MARKER, None):
            raise InvalidDefinition(
                f"{_rule_qualname(f)} is used with both @initialize and @invariant, "
                "which is not allowed. A function may be either an initialization "
                "rule or an invariant, but not both."
            )
        existing_rule = getattr(f, RULE_MARKER, None)
        existing_initialize_rule = getattr(f, INITIALIZE_RULE_MARKER, None)
        if existing_rule is not None:
            raise InvalidDefinition(
                f"{_rule_qualname(f)} has been decorated with both @rule and "
                "@initialize, which is not allowed."
            )
        if existing_initialize_rule is not None:
            raise InvalidDefinition(
                f"{_rule_qualname(f)} has been decorated with @initialize twice, "
                "which is not allowed."
            )
        preconditions = getattr(f, PRECONDITIONS_MARKER, ())
        if preconditions:
            raise InvalidDefinition(
                f"{_rule_qualname(f)} has been decorated with both @initialize and "
                "@precondition, which is not allowed. An initialization rule "
                "runs unconditionally and may not have a precondition."
            )
        rule = Rule(
            targets=converted_targets,
            arguments=kwargs,
            function=f,
            preconditions=preconditions,
        )

        @proxies(f)
        def rule_wrapper(*args, **kwargs):
            pass

        setattr(rule_wrapper, INITIALIZE_RULE_MARKER, rule)
        return rule_wrapper

    return accept


@dataclass(slots=True, frozen=True)
class VarReference:
    name: str


# There are multiple alternatives for annotating the `precond` type, all of them
# have drawbacks. See https://github.com/HypothesisWorks/hypothesis/pull/3068#issuecomment-906642371
def precondition(precond: Callable[[Any], bool]) -> Callable[[TestFunc], TestFunc]:
    """Decorator to apply a precondition for rules in a RuleBasedStateMachine.
    Specifies a precondition for a rule to be considered as a valid step in the
    state machine, which is more efficient than using :func:`~hypothesis.assume`
    within the rule.  The ``precond`` function will be called with the instance of
    RuleBasedStateMachine and should return True or False. Usually it will need
    to look at attributes on that instance.

    For example::

        class MyTestMachine(RuleBasedStateMachine):
            state = 1

            @precondition(lambda self: self.state != 0)
            @rule(numerator=integers())
            def divide_with(self, numerator):
                self.state = numerator / self.state

    If multiple preconditions are applied to a single rule, it is only considered
    a valid step when all of them return True.  Preconditions may be applied to
    invariants as well as rules.
    """

    def decorator(f):
        @proxies(f)
        pass

    return decorator


@dataclass(slots=True, frozen=True)
class Invariant:
    function: Any
    preconditions: Any
    check_during_init: bool

    def __repr__(self) -> str:
        parts = [
            f"function={get_pretty_function_description(self.function)}",
            f"{self.preconditions=}",
            f"{self.check_during_init=}",
        ]
        return f"Invariant({', '.join(parts)})"


def invariant(*, check_during_init: bool = False) -> Callable[[TestFunc], TestFunc]:
    """Decorator to apply an invariant for rules in a RuleBasedStateMachine.
    The decorated function will be run after every rule and can raise an
    exception to indicate failed invariants.

    For example::

        class MyTestMachine(RuleBasedStateMachine):
            state = 1

            @invariant()
            def is_nonzero(self):
                assert self.state != 0

    By default, invariants are only checked after all
    :func:`@initialize() <hypothesis.stateful.initialize>` rules have been run.
    Pass ``check_during_init=True`` for invariants which can also be checked
    during initialization.
    """
    pass


class RuleStrategy(SearchStrategy):
    def __init__(self, machine: RuleBasedStateMachine) -> None:
        super().__init__()
        self.machine = machine
        self.rules = machine.rules.copy()

        self.enabled_rules_strategy = st.shared(
            FeatureStrategy(at_least_one_of={r.function.__name__ for r in self.rules}),
            key=("enabled rules", machine),
        )

        # The order is a bit arbitrary. Primarily we're trying to group rules
        # that write to the same location together, and to put rules with no
        # target first as they have less effect on the structure. We order from
        # fewer to more arguments on grounds that it will plausibly need less
        # data. This probably won't work especially well and we could be
        # smarter about it, but it's better than just doing it in definition
        # order.
        self.rules.sort(
            key=lambda rule: (
                sorted(rule.targets),
                len(rule.arguments),
                rule.function.__name__,
            )
        )
        self.rules_strategy = st.sampled_from(self.rules)

    def __repr__(self):
        return f"{self.__class__.__name__}(machine={self.machine.__class__.__name__}({{...}}))"

    def do_draw(self, data):
        if not any(self.is_valid(rule) for rule in self.rules):
            rules = ", ".join([rule.function.__name__ for rule in self.rules])
            msg = (
                f"No progress can be made from state {self.machine!r}, because no "
                f"available rule had a True precondition. rules: {rules}"
            )
            raise InvalidDefinition(msg) from None

        feature_flags = data.draw(self.enabled_rules_strategy)

        def rule_is_enabled(r):
            # Note: The order of the filters here is actually quite important,
            # because checking is_enabled makes choices, so increases the size of
            # the choice sequence. This means that if we are in a case where many
            # rules are invalid we would make a lot more choices if we ask if they
            # are enabled before we ask if they are valid, so our test cases would
            # be artificially large.
            pass

        rule = data.draw(self.rules_strategy.filter(rule_is_enabled))

        arguments = {}
        for k, strat in rule.arguments_strategies.items():
            try:
                arguments[k] = data.draw(strat)
            except Exception as err:
                rname = rule.function.__name__
                add_note(err, f"while generating {k!r} from {strat!r} for rule {rname}")
                raise
        return (rule, arguments)

    def is_valid(self, rule):
        for b in rule.bundles:
            if not self.machine.bundle(b.name):
                return False

        predicates = self.machine._observability_predicates
        desc = f"{self.machine.__class__.__qualname__}, rule {rule.function.__name__},"
        for pred in rule.preconditions:
            meets_precond = pred(self.machine)
            where = f"{desc} precondition {get_pretty_function_description(pred)}"
            predicates[where].update_count(condition=meets_precond)
            if not meets_precond:
                return False

        return True
