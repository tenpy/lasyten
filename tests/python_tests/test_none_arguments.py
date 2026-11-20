"""Static guard: bound functions must not accept ``None`` for required arguments of a bound class type.

Such an argument arrives in C++ as an empty ``std::shared_ptr``, which most functions dereference without a check:
``Block + None`` used to crash the interpreter, see tenpy/lasyten#334. nanobind rejects ``None`` for pointer-like
parameters unless the binding says ``nb::arg("x") = nb::none()`` or ``nb::arg("x").none()``; a parameter that accepts
it shows up as ``<class> | None`` in the signature. Parameters with a default are fine (``None`` is the default).

This test only reads the signatures (``__nb_signature__``) and calls nothing, so a regression fails the test instead of
crashing the pytest process.
"""
# Copyright (C) TeNPy Developers, Apache license

from __future__ import annotations

import inspect
import re

import pytest

import lasyten._core

# Parameters that accept ``None`` on purpose: (name of the function, parameter name) -> reason.
# The class a function is found on is not part of the key, since inherited functions are found on any subclass.
ALLOWED_NONE = {
    ('_parse_default_dtype', 'dtype'): '``std::optional<Dtype>``: ``None`` selects the default dtype.',
}


def _split_parameters(signature: str) -> list[tuple[str, str, bool]]:
    """Parse ``def f(a: A, b: B = ..., /) -> R`` into ``(name, annotation, has_default)``."""
    body = re.match(r'def \w+\((.*)\)( -> .*)?$', signature, re.S).group(1)
    parts, depth, current = [], 0, ''
    for char in body:
        depth += char in '[('
        depth -= char in '])'
        if char == ',' and depth == 0:
            parts.append(current.strip())
            current = ''
        else:
            current += char
    if current.strip():
        parts.append(current.strip())
    result = []
    for part in parts:
        if part in ('/', '*') or part.startswith('*'):
            continue
        name, _, annotation = part.split(' = ')[0].partition(':')
        result.append((name.strip(), annotation.strip(), ' = ' in part))
    return result


def _bound_functions(namespace, path, seen):
    """Yield ``(qualified name, nanobind signature)`` for all bound functions in the namespace, recursively."""
    if id(namespace) in seen:
        return
    seen.add(id(namespace))
    for name in dir(namespace):
        try:
            member = getattr(namespace, name)
        except AttributeError:
            continue
        qualname = f'{path}.{name}'
        if hasattr(member, '__nb_signature__'):
            if id(member) not in seen:
                seen.add(id(member))
                for signature, *_ in member.__nb_signature__:
                    yield qualname, signature
        elif inspect.isclass(member) and member.__module__.startswith('lasyten'):
            yield from _bound_functions(member, qualname, seen)


def test_required_class_arguments_reject_none():
    bound_class = re.compile(r'lasyten\._core\.[\w.]+ \| None')
    accepting_none = []
    for qualname, signature in _bound_functions(lasyten._core, 'lasyten._core', set()):
        for name, annotation, has_default in _split_parameters(signature):
            if name == 'self' or has_default or not bound_class.fullmatch(annotation):
                continue
            if (qualname.rsplit('.', 1)[-1], name) not in ALLOWED_NONE:
                accepting_none.append(f'{qualname}({name}: {annotation})')
    assert not accepting_none, (
        'required arguments of a bound class type accept None (it reaches C++ as an empty pointer); use a '
        f'default of None, or list them in ALLOWED_NONE with a reason:\n' + '\n'.join(sorted(accepting_none))
    )


@pytest.mark.parametrize(
    'signature, expect',
    [
        pytest.param(
            'def f(self, a: lasyten._core.X | None, b: int = 1) -> int',
            [('self', '', False), ('a', 'lasyten._core.X | None', False), ('b', 'int', True)],
            id='simple',
        ),
        pytest.param(
            'def f(a: collections.abc.Sequence[tuple[int, int]], /, *args, k: bool = False) -> int',
            [('a', 'collections.abc.Sequence[tuple[int, int]]', False), ('k', 'bool', True)],
            id='nested-brackets',
        ),
    ],
)
def test_split_parameters(signature, expect):
    assert _split_parameters(signature) == expect
