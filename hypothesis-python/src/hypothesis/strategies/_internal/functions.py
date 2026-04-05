# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from weakref import WeakKeyDictionary

from hypothesis.control import note, should_note
from hypothesis.errors import InvalidState
from hypothesis.internal.reflection import (
    convert_positional_arguments,
    nicerepr,
    proxies,
    repr_call,
)
from hypothesis.strategies._internal.strategies import RecurT, SearchStrategy


class FunctionStrategy(SearchStrategy):
    def __init__(self, like, returns, pure):
        super().__init__()
        self.like = like
        self.returns = returns
        self.pure = pure
        # Using wekrefs-to-generated-functions means that the cache can be
        # garbage-collected at the end of each example, reducing memory use.
        self._cache = WeakKeyDictionary()

    def calc_is_empty(self, recur: RecurT) -> bool:
        pass

    def do_draw(self, data):
        @proxies(self.like)
        def inner(*args, **kwargs):
            pass

        return inner
