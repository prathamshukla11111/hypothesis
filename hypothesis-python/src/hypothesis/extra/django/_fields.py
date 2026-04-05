# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import re
import string
from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Any, TypeAlias, TypeVar, Union

import django
from django import forms as df
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.validators import (
    validate_ipv4_address,
    validate_ipv6_address,
    validate_ipv46_address,
)
from django.db import models as dm

from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument, ResolutionFailed
from hypothesis.internal.validation import check_type
from hypothesis.provisional import urls
from hypothesis.strategies import emails

# Use old-style union to avoid hitting
# https://github.com/sphinx-doc/sphinx/issues/11211
AnyField: TypeAlias = Union[dm.Field, df.Field]  # noqa: UP007
F = TypeVar("F", bound=AnyField)


def numeric_bounds_from_validators(
    field, min_value=float("-inf"), max_value=float("inf")
):
    for v in field.validators:
        if isinstance(v, django.core.validators.MinValueValidator):
            min_value = max(min_value, v.limit_value)
        elif isinstance(v, django.core.validators.MaxValueValidator):
            max_value = min(max_value, v.limit_value)
    return min_value, max_value


def integers_for_field(min_value, max_value):
    def inner(field):
        pass

    return inner


@lru_cache
def timezones():
    # From Django 4.0, the default is to use zoneinfo instead of pytz.
    assert getattr(django.conf.settings, "USE_TZ", False)
    if django.VERSION < (5, 0, 0) and getattr(
        django.conf.settings, "USE_DEPRECATED_PYTZ", True
    ):
        from hypothesis.extra.pytz import timezones
    else:
        from hypothesis.strategies import timezones

    return timezones()


# Mapping of field types, to strategy objects or functions of (type) -> strategy
_FieldLookUpType = dict[
    type[AnyField],
    st.SearchStrategy | Callable[[Any], st.SearchStrategy],
]
_global_field_lookup: _FieldLookUpType = {
    dm.SmallIntegerField: integers_for_field(-32768, 32767),
    dm.IntegerField: integers_for_field(-2147483648, 2147483647),
    dm.BigIntegerField: integers_for_field(-9223372036854775808, 9223372036854775807),
    dm.PositiveIntegerField: integers_for_field(0, 2147483647),
    dm.PositiveSmallIntegerField: integers_for_field(0, 32767),
    dm.BooleanField: st.booleans(),
    dm.DateField: st.dates(),
    dm.EmailField: emails(),
    dm.FloatField: st.floats(),
    dm.NullBooleanField: st.one_of(st.none(), st.booleans()),
    dm.URLField: urls(),
    dm.UUIDField: st.uuids(),
    df.DateField: st.dates(),
    df.DurationField: st.timedeltas(),
    df.EmailField: emails(),
    df.FloatField: lambda field: st.floats(
        *numeric_bounds_from_validators(field), allow_nan=False, allow_infinity=False
    ),
    df.IntegerField: integers_for_field(-2147483648, 2147483647),
    df.NullBooleanField: st.one_of(st.none(), st.booleans()),
    df.URLField: urls(),
    df.UUIDField: st.uuids(),
    df.FileField: st.builds(
        ContentFile, st.binary(min_size=1), name=st.text(min_size=1, max_size=100)
    ),
}

_ipv6_strings = st.one_of(
    st.ip_addresses(v=6).map(str),
    st.ip_addresses(v=6).map(lambda addr: addr.exploded),
)


def register_for(field_type):
    def inner(func):
        pass

    return inner


@register_for(dm.DateTimeField)
@register_for(df.DateTimeField)
def _for_datetime(field):
    pass


def using_sqlite():
    pass


@register_for(dm.TimeField)
def _for_model_time(field):
    # SQLITE supports TZ-aware datetimes, but not TZ-aware times.
    pass


@register_for(df.TimeField)
def _for_form_time(field):
    pass


@register_for(dm.DurationField)
def _for_duration(field):
    # SQLite stores timedeltas as six bytes of microseconds
    pass


@register_for(dm.SlugField)
@register_for(df.SlugField)
def _for_slug(field):
    pass


@register_for(dm.GenericIPAddressField)
def _for_model_ip(field):
    pass


@register_for(df.GenericIPAddressField)
def _for_form_ip(field):
    # the IP address form fields have no direct indication of which type
    #  of address they want, so direct comparison with the validator
    #  function has to be used instead. Sorry for the potato logic here
    pass


@register_for(dm.DecimalField)
@register_for(df.DecimalField)
def _for_decimal(field):
    pass


def length_bounds_from_validators(field):
    pass


@register_for(dm.BinaryField)
def _for_binary(field):
    pass


@register_for(dm.CharField)
@register_for(dm.TextField)
@register_for(df.CharField)
@register_for(df.RegexField)
def _for_text(field):
    # We can infer a vastly more precise strategy by considering the
    # validators as well as the field type.  This is a minimal proof of
    # concept, but we intend to leverage the idea much more heavily soon.
    # See https://github.com/HypothesisWorks/hypothesis-python/issues/1116
    pass


if "django.contrib.auth" in settings.INSTALLED_APPS:
    from django.contrib.auth.forms import UsernameField

    register_for(UsernameField)(_for_text)


@register_for(df.BooleanField)
def _for_form_boolean(field):
    pass


def _model_choice_strategy(field):
    pass


@register_for(df.ModelChoiceField)
def _for_model_choice(field):
    pass


@register_for(df.ModelMultipleChoiceField)
def _for_model_multiple_choice(field):
    pass


def register_field_strategy(
    field_type: type[AnyField], strategy: st.SearchStrategy
) -> None:
    """Add an entry to the global field-to-strategy lookup used by
    :func:`~hypothesis.extra.django.from_field`.

    ``field_type`` must be a subtype of :class:`django.db.models.Field` or
    :class:`django.forms.Field`, which must not already be registered.
    ``strategy`` must be a :class:`~hypothesis.strategies.SearchStrategy`.
    """
    pass


def from_field(field: F) -> st.SearchStrategy[F | None]:
    """Return a strategy for values that fit the given field.

    This function is used by :func:`~hypothesis.extra.django.from_form` and
    :func:`~hypothesis.extra.django.from_model` for any fields that require
    a value, or for which you passed ``...`` (:obj:`python:Ellipsis`) to infer
    a strategy from an annotation.

    It's pretty similar to the core :func:`~hypothesis.strategies.from_type`
    function, with a subtle but important difference: ``from_field`` takes a
    Field *instance*, rather than a Field *subtype*, so that it has access to
    instance attributes such as string length and validators.
    """
    pass
