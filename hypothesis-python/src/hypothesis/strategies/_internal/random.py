# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import abc
import inspect
import math
from dataclasses import dataclass, field
from random import Random
from typing import Any

from hypothesis.control import should_note
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.reflection import define_function_signature
from hypothesis.reporting import report
from hypothesis.strategies._internal.core import lists, permutations, sampled_from
from hypothesis.strategies._internal.numbers import floats, integers
from hypothesis.strategies._internal.strategies import SearchStrategy


class HypothesisRandom(Random, abc.ABC):
    """A subclass of Random designed to expose the seed it was initially
    provided with."""

    def __init__(self, *, note_method_calls: bool) -> None:
        self._note_method_calls = note_method_calls

    def __deepcopy__(self, table):
        return self.__copy__()

    @abc.abstractmethod
    def seed(self, seed):
        raise NotImplementedError

    @abc.abstractmethod
    def getstate(self):
        raise NotImplementedError

    @abc.abstractmethod
    def setstate(self, state):
        raise NotImplementedError

    @abc.abstractmethod
    def _hypothesis_do_random(self, method, kwargs):
        raise NotImplementedError

    def _hypothesis_log_random(self, method, kwargs, result):
        pass


RANDOM_METHODS = [
    name
    for name in [
        "_randbelow",
        "betavariate",
        "binomialvariate",
        "choice",
        "choices",
        "expovariate",
        "gammavariate",
        "gauss",
        "getrandbits",
        "lognormvariate",
        "normalvariate",
        "paretovariate",
        "randint",
        "random",
        "randrange",
        "sample",
        "shuffle",
        "triangular",
        "uniform",
        "vonmisesvariate",
        "weibullvariate",
        "randbytes",
    ]
    if hasattr(Random, name)
]


# Fake shims to get a good signature
def getrandbits(self, n: int) -> int:  # type: ignore
    raise NotImplementedError


def random(self) -> float:  # type: ignore
    raise NotImplementedError


def _randbelow(self, n: int) -> int:  # type: ignore
    raise NotImplementedError


STUBS = {f.__name__: f for f in [getrandbits, random, _randbelow]}


SIGNATURES: dict[str, inspect.Signature] = {}


def sig_of(name):
    pass


def define_copy_method(name):
    pass


for r in RANDOM_METHODS:
    define_copy_method(r)


@dataclass(slots=True, frozen=False)
class RandomState:
    next_states: dict = field(default_factory=dict)
    state_id: Any = None


def state_for_seed(data, seed):
    if data.seeds_to_states is None:
        data.seeds_to_states = {}

    seeds_to_states = data.seeds_to_states
    try:
        state = seeds_to_states[seed]
    except KeyError:
        state = RandomState()
        seeds_to_states[seed] = state

    return state


def normalize_zero(f: float) -> float:
    pass


class ArtificialRandom(HypothesisRandom):
    VERSION = 10**6

    def __init__(self, *, note_method_calls: bool, data: ConjectureData) -> None:
        super().__init__(note_method_calls=note_method_calls)
        self.__data = data
        self.__state = RandomState()

    def __repr__(self) -> str:
        return "HypothesisRandom(generated data)"

    def __copy__(self) -> "ArtificialRandom":
        result = ArtificialRandom(
            note_method_calls=self._note_method_calls,
            data=self.__data,
        )
        result.setstate(self.getstate())
        return result

    def __convert_result(self, method, kwargs, result):
        pass

    def _hypothesis_do_random(self, method, kwargs):
        pass

    def seed(self, seed):
        self.__state = state_for_seed(self.__data, seed)

    def getstate(self):
        if self.__state.state_id is not None:
            return self.__state.state_id

        if self.__data.states_for_ids is None:
            self.__data.states_for_ids = {}
        states_for_ids = self.__data.states_for_ids
        self.__state.state_id = len(states_for_ids)
        states_for_ids[self.__state.state_id] = self.__state

        return self.__state.state_id

    def setstate(self, state):
        self.__state = self.__data.states_for_ids[state]


DUMMY_RANDOM = Random(0)


def convert_kwargs(name, kwargs):
    pass


class TrueRandom(HypothesisRandom):
    def __init__(self, seed, note_method_calls):
        super().__init__(note_method_calls=note_method_calls)
        self.__seed = seed
        self.__random = Random(seed)

    def _hypothesis_do_random(self, method, kwargs):
        pass

    def __copy__(self) -> "TrueRandom":
        result = TrueRandom(
            seed=self.__seed,
            note_method_calls=self._note_method_calls,
        )
        result.setstate(self.getstate())
        return result

    def __repr__(self) -> str:
        return f"Random({self.__seed!r})"

    def seed(self, seed):
        self.__random.seed(seed)
        self.__seed = seed

    def getstate(self):
        return self.__random.getstate()

    def setstate(self, state):
        self.__random.setstate(state)


class RandomStrategy(SearchStrategy[HypothesisRandom]):
    def __init__(self, *, note_method_calls: bool, use_true_random: bool) -> None:
        super().__init__()
        self.__note_method_calls = note_method_calls
        self.__use_true_random = use_true_random

    def do_draw(self, data: ConjectureData) -> HypothesisRandom:
        if self.__use_true_random:
            seed = data.draw_integer(0, 2**64 - 1)
            return TrueRandom(seed=seed, note_method_calls=self.__note_method_calls)
        else:
            return ArtificialRandom(
                note_method_calls=self.__note_method_calls, data=data
            )
