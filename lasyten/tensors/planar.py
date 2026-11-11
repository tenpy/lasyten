"""Provides useful classes and functions for dealing with planar (systems of) tensors.

Planar systems of tensors consist of tensors whose legs are contracted in such a way that they do
not require braids. More graphically, this means that such systems of tensors can be represented as
diagrams with the individual tensor legs not crossing one another. Such planar diagrams are
represented by :class:`PlanarDiagram` s, which take :class:`TensorPlaceholder` as representations
of tensors and give an order in terms of :class:`ContractionTree` s in which all the tensors can be
contracted in the most efficient way. The result of planar diagrams can then be computed for
concrete tensors using :meth:`evaluate`. See :class:`PlanarDiagram` for more details and examples.

For planar linear operators, :class:`PlanarLinearOperator` provides the opportunity to specify the
operator itself as well as its action on a vector in terms planar diagrams, which can simplify the
implementation of planar algorithms such as TEBD or DMRG.

There are additional useful planar functions provided ranging from planar decompositions
(:func:`planar_eigh`, :func:`planar_eig`, :func:`planar_eigvals`, :func:`planar_lq`,
:func:`planar_qr`, :func:`planar_svd`,
:func:`planar_truncated_svd`) and planar leg permutations (:func:`planar_permute_legs`),
to planar leg combinations (:func:`planar_combine_legs`), planar partial traces
(:func:`planar_partial_trace`), and planar tensor contractions
(:func:`planar_contraction`, :func:`planar_contract_mask`).
It is also possible to compare two tensors up to cyclic leg permutations
(:func:`planar_almost_equal`).
"""

# Copyright (C) TeNPy Developers, Apache license

from .._core import (  # noqa: F401
    ContractionTree,
    ContractionTreeNode,
    PlanarDiagram,
    TensorPlaceholder,
    horizontal_factorization,
    parse_leg_bipartition,
    planar_almost_equal,
    planar_contract_mask,
    planar_combine_legs,
    planar_contraction,
    planar_decomposition,
    planar_eig,
    planar_eigh,
    planar_eigvals,
    planar_lq,
    planar_permute_legs,
    planar_partial_trace,
    planar_qr,
    planar_svd,
    planar_truncated_svd,
)
from .._core import (
    PlanarLinearOperator as _PlanarLinearOperator,
)
from ._tensors import partial_trace  # noqa: F401


class PlanarLinearOperator(_PlanarLinearOperator):
    r"""Base class for :class:`LinearOperator`\ s defined in terms of :class:`PlanarDiagram`\ s.

    .. warning ::
        Instantiating a :class:`PlanarDiagram` may be expensive if the order is optimized.
        Make sure to either hard-code the order, or make the planar diagram instance as early as
        possible, e.g., as a *class* variable of the parent class instead of during its
        ``__init__``.

    Parameters
    ----------
    op_diagram : :class:`PlanarDiagram`
        The diagram that defines the operator (without acting on a vector).
    matvec_diagram : :class:`PlanarDiagram`
        The diagram that defines the action of the operator on a vector.
        Must have the same tensor names as the `op_diagram` in addition to a single tensor
        with `vec_name`.
    op_tensors : {str : :class:`Tensor`}
        The concrete tensors that define the operator, see `op_diagram`.
    vec_name : str
        The name of the "vector", i.e., the tensor that the linear operator acts on in the
        `matvec_diagram`.

    """

    def __init__(self, op_diagram, matvec_diagram, op_tensors, vec_name):
        # C++ stores the diagrams / tensors. Re-assign them as Python instance attributes so that
        # subclass *class* variables of the same name (the documented pattern) are not shadowed by
        # pybind11 data descriptors during ``self.op_diagram`` lookup before ``__init__`` finishes.
        super().__init__(op_diagram, matvec_diagram, op_tensors, vec_name)
        self.op_diagram = op_diagram
        self.matvec_diagram = matvec_diagram
        self.op_tensors = op_tensors
        self.vec_name = vec_name
