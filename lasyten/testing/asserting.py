"""Assertion wrappers for testing."""

# Copyright (C) TeNPy Developers, Apache license
from ..symmetries import Leg, TensorProduct, is_leg_pipe
from ..tensors import SymmetricTensor, Tensor, almost_equal


def assert_tensors_almost_equal(a: SymmetricTensor, expect: SymmetricTensor, rtol: float = 1e-12, atol: float = 1e-12):
    """Verify two tensors have the same legs and almost equal numerical entries."""
    assert a.codomain == expect.codomain
    assert a.domain == expect.domain
    assert almost_equal(a, expect, rtol=rtol, atol=atol)


def assert_equivalent_legs(a: Tensor, b: Tensor):
    """Verify two tensors have equivalent legs.

    This is a more lose check than "equal" legs; we allow different types of pipes.
    """
    assert a.symmetry == b.symmetry, 'Mismatching symmetry'
    _equivalent_leg_recursion(a.codomain, b.codomain, loc='.codomain')
    _equivalent_leg_recursion(a.domain, b.domain, loc='.codomain')


def _equivalent_leg_recursion(a: TensorProduct | Leg, b: TensorProduct | Leg, loc: str):
    if isinstance(a, TensorProduct) and isinstance(b, TensorProduct):
        assert a.num_factors == b.num_factors, f'Mismatching num_factors at TensorProduct {loc}'
        for i in range(a.num_factors):
            _equivalent_leg_recursion(a[i], b[i], loc=f'{loc}[{i}]')

    elif is_leg_pipe(a) and is_leg_pipe(b):
        assert a.num_legs == b.num_legs, f'Mismatching num_legs at pipe {loc}'
        assert a.is_dual == b.is_dual, f'Mismatching is_dual at pipe {loc}'
        assert a.combine_cstyle == b.combine_cstyle, f'Mismatching combine_cstyle at pipe {loc}'
        for i in range(a.num_legs):
            _equivalent_leg_recursion(a[i], b[i], loc=f'{loc}[{i}]')

    elif is_leg_pipe(a) or is_leg_pipe(b):
        raise AssertionError(f'Mismatching types (pipe vs no pipe) at {loc}')

    elif isinstance(a, Leg) and isinstance(b, Leg):
        assert a == b, f'Mismatching non-pipe legs at {loc}'

    else:
        raise TypeError
