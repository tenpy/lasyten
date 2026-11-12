"""Providing support for sparse algorithms (using matrix-vector products only).

Some linear algebra algorithms, e.g. Lanczos, do not require the full representations of a linear
operator, but only the action on a vector, i.e., a matrix-vector product `matvec`. Here we define
the structure of such a general operator, :class:`LinearOperator`, as it is used in our own
implementations of these algorithms (e.g., :mod:`~lasyten.krylov_based`). Moreover, the
:class:`NumpyArrayLinearOperator` allows to use all the scipy sparse methods by providing
functionality to convert flat numpy arrays to and from lasyten tensors.
"""
# Copyright (C) TeNPy Developers, Apache license

import warnings
from typing import Literal

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator

# Monkey-patch converted sparse classes/functions from C++ bindings.
from .._core import (  # noqa: E402,F401
    DirectSumLinearOperator,
    LinearOperator,
    LinearOperatorWrapper,
    ProjectedLinearOperator,
    ShiftedLinearOperator,
    SumLinearOperator,
    TensorLinearOperator,
    gram_schmidt,
)
from ..backends import TensorBackend
from ..block_backends import Dtype
from ..symmetries import Sector, Space, SymmetryError, TensorProduct
from ..tools.math import speigs, speigsh
from ..tools.misc import argsort
from ._tensors import (
    ChargedTensor,
    DirectSum,
    SymmetricTensor,
    Tensor,
    combine_legs,
    permute_legs,
    split_legs,
    tdot,
)


class NumpyArrayLinearOperator(ScipyLinearOperator):
    """Square Linear operator acting on numpy arrays based on a matvec acting on lasyten tensors.

    Note that this class represents a square linear operator.

    Parameters
    ----------
    lasyten_matvec : callable
        Function with signature ``lasyten_matvec(vec: Tensor) -> Tensor``.
        Has to return a tensor with the same legs and has to be linear.
        Unless `labels` are given, the leg order of the output must be the same as for the input.
    legs : list of :class:`~lasyten.spaces.ElementarySpace`
        The legs of a Tensor that `lasyten_matvec` can act on.
    backend : :class:`~lasyten.backends.abstract_backend.Backend`
        The backend for self
    dtype
        The numpy dtype for this operator.
    labels : list of str, optional
        The labels for inputs to `lasyten_matvec`.
    charge_sector : None | Sector | 'trivial'
        If given, only the specified charge sector is considered.
        Per default, or if the string ``'trivial'`` is given, the trivial sector of the symmetry is used.
        ``None`` stands for *all* sectors: the numpy vector is the full dense representation,
        converted via a :class:`~lasyten.tensors.ChargedTensor` with a specified
        :attr:`~lasyten.tensors.ChargedTensor.charged_state`.
        This requires a group-like symmetry with :attr:`~lasyten.symmetries.Symmetry.can_be_dropped`.

    Attributes
    ----------
    lasyten_matvec : callable
        Function with signature ``lasyten_matvec(vec: Tensor) -> Tensor``.
    legs : list of :class:`~lasyten.spaces.Space`
        The legs of a Tensor that `lasyten_matvec` can act on.
    backend : :class:`~lasyten.backends.abstract_backend.Backend`
        The backend for self
    dtype
        The numpy dtype for this operator.
    labels : list of str, optional
        The labels for inputs to `lasyten_matvec`.
    matvec_count : int
        The number of times `lasyten_matvec` was called.
    N : int
        The length of the numpy vectors that this operator acts on
    domain : :class:`~lasyten.spaces.TensorProduct`
        The product of the :attr:`legs`. Self is an operator on either this entire space,
        or one of its sectors, as specified by :attr:`charge_sector`.
    pipe : LegPipe
        Combined pipe of :attr:`legs`, used for convertion to/from a 1-leg tensor.
    symmetry
        The symmetry of all involved spaces
    shape : (int, int)
        The shape of self as an operator on 1D numpy arrays

    """

    def __init__(
        self,
        lasyten_matvec,
        legs: TensorProduct | list[Space] | None = None,
        backend: TensorBackend | None = None,
        dtype=None,
        labels: list[str] = None,
        charge_sector: None | Sector | Literal['trivial'] = 'trivial',
        component_converters: list[NumpyArrayLinearOperator] | None = None,
    ):
        self.lasyten_matvec = lasyten_matvec
        self.component_converters = component_converters
        self.matvec_count = 0
        if component_converters is not None:
            if len(component_converters) == 0:
                raise ValueError('component_converters must be non-empty')
            self.backend = component_converters[0].backend
            self.legs = None
            self.labels = None
            self.domain = None
            self.pipe = None
            self.symmetry = component_converters[0].symmetry
            sizes = [int(op.shape[0]) for op in component_converters]
            self._component_sizes = sizes
            n = sum(sizes)
            self.shape = (n, n)
            self._charge_sector = None
            ScipyLinearOperator.__init__(self, dtype=dtype, shape=self.shape)
            return
        self.backend = backend
        if not isinstance(legs, TensorProduct):
            self.legs = list(legs)
            self.domain = TensorProduct(self.legs)
            self.symmetry = self.legs[0].symmetry
        else:
            self.domain = legs
            self.legs = list(legs.factors)
            self.symmetry = legs.symmetry
        if len(self.legs) == 1:
            self.pipe = self.legs[0]
        else:
            self.pipe = backend.make_pipe(self.legs, is_dual=False)
        self.labels = labels

        self.shape = None  # set by charge_sector.setter
        self._charge_sector = None  # set by charge_sector.setter
        self.charge_sector = charge_sector  # uses setter with its input checks and conversions

        ScipyLinearOperator.__init__(self, dtype=dtype, shape=self.shape)

    @classmethod
    def from_Tensor(
        cls,
        tensor: SymmetricTensor,
        legs1: list[int | str],
        legs2: list[int | str],
        charge_sector: None | Sector | Literal['trivial'] = 'trivial',
    ) -> NumpyArrayLinearOperator:
        """Create a :class:`NumpyArrayLinearOperator` from a tensor that acts via contraction (`tdot`).

        The `lasyten_matvec` acting on ``vec`` is given by ``tdot(tensor, vec, legs1, legs2)``.

        Parameters
        ----------
        tensor : Tensor
            A tensors whose legs specified by `legs1` are contractible with the remaining legs.
        legs1 : list of {int | str}
            Which legs of `tensor` should be contracted on matvec
        legs2 : list of {int | str}
            Which legs of the "vector" should be contracted on `matvec`
        charge_sector : None | Sector | 'trivial'
            If given, only the specified charge sector is considered.
            If ``'trivial'`` is given, the trivial sector of the symmetry is used.
            ``None`` stands for *all* sectors.

        """
        idcs1 = tensor.labels.indices(legs1)
        tensor_contr_legs = [tensor.legs[idx] for idx in idcs1]
        res_legs = [tensor.legs[idx] for idx in range(tensor.num_legs) if idx not in idcs1]
        res_labels = [tensor.labels[idx] for idx in range(tensor.num_legs) if idx not in idcs1]
        if '' in res_labels:
            res_labels = None
        vec_contr_legs = []
        for l in legs2:
            if isinstance(l, int):
                vec_contr_legs.append(res_legs[l])
            else:
                vec_contr_legs.append(tensor.get_leg(l))
        if not all(l_t == l_v.dual for l_t, l_v in zip(tensor_contr_legs, vec_contr_legs)):
            raise ValueError('Expected contractible legs')

        def lasyten_matvec(vec):
            return tdot(tensor, vec, legs1, legs2)

        return cls(
            lasyten_matvec,
            legs=vec_contr_legs,
            backend=tensor.backend,
            dtype=tensor.dtype.to_numpy_dtype(),
            labels=res_labels,
            charge_sector=charge_sector,
        )

    @classmethod
    def from_matvec_and_vector(
        cls, lasyten_matvec, vector: Tensor | DirectSum, dtype=None
    ) -> tuple[NumpyArrayLinearOperator, np.ndarray]:
        """Create a :class:`NumpyArrayLinearOperator` from a matvec and a vector that it can act on.

        This is a convenience wrapper around the constructor where arguments are inferred
        from the example `vector` that is given.
        Additionally, the `vector` is converted via :meth:`tensor_to_flat_array`.
        The resulting `NumpyArrayLinearOperator` has a `charge_sector` set to be the sector of
        `vector`.

        Parameters
        ----------
        lasyten_matvec : callable
            Function with signature ``lasyten_matvec(vec) -> vec`` on :class:`~lasyten.tensors.Tensor`
            or :class:`~lasyten.tensors.DirectSum`.
        vector : :class:`~lasyten.tensors.Tensor` | :class:`~lasyten.tensors.DirectSum`
            A vector that `lasyten_matvec` can act on.
            For a Tensor / ChargedTensor, the charge sector is inferred as before.
            For a DirectSum, each component is flattened and concatenated.
        dtype
            The *numpy* dtype of the operator. Per default, the dtype of `vector` is used.

        Returns
        -------
        op : :class:`NumpyArrayLinearOperator`
            The resulting operator
        vec_flat : 1D ndarray
            Flat numpy vector representing `vector` within its charge sector.

        """
        if isinstance(vector, DirectSum):
            if dtype is None:
                dtype = vector.dtype.to_numpy_dtype()
            converters = []
            flats = []
            for comp in vector.components:
                conv_i, flat_i = cls.from_matvec_and_vector(lambda v: v, comp, dtype=dtype)
                converters.append(conv_i)
                flats.append(flat_i)
            op = cls(lasyten_matvec, dtype=dtype, component_converters=converters)
            return op, np.concatenate(flats)

        if isinstance(vector, ChargedTensor):
            charge = vector.charge_leg
            try:
                single_charge = charge.num_sectors == 1 and charge.multiplicities[0] == 1
            except AttributeError:
                # e.g. a LegPipe charge leg from the all-sector identity construction
                single_charge = False
            if single_charge:
                sector = charge.sector_decomposition[0]
            else:
                sector = None
        else:
            sector = 'trivial'
        if dtype is None:
            dtype = vector.dtype.to_numpy_dtype()
        labels = vector.labels
        if labels is not None and not any(l is not None for l in labels):
            labels = None
        op = cls(
            lasyten_matvec,
            legs=vector.legs,
            backend=vector.backend,
            dtype=dtype,
            labels=labels,
            charge_sector=sector,
        )
        vec_flat = op.tensor_to_flat_array(vector)
        return op, vec_flat

    @property
    def charge_sector(self):
        """Charge sector this operator acts in, ``'trivial'``, or ``None`` for all sectors."""
        return self._charge_sector

    @charge_sector.setter
    def charge_sector(self, value):
        if isinstance(value, str) and value == 'trivial':
            sector = self.symmetry.trivial_sector
        elif value is None:
            if not self.symmetry.can_be_dropped:
                raise SymmetryError(
                    'charge_sector=None uses ChargedTensor.charged_state and is only defined '
                    f'for symmetries that can be dropped. Got {self.symmetry}.'
                )
            sector = None
        else:
            assert self.symmetry.is_valid_sector(value)
            sector = value
        self._charge_sector = value
        if sector is None:
            size = int(self.domain.dim)
        else:
            sector_idx = self.domain.sector_decomposition_where(sector)
            if sector_idx is None:
                raise ValueError('Domain of linear operator does not have this sector')
            size = int(self.domain.block_size(sector))
        self.shape = (size, size)

    def _matvec(self, vec):
        """Matvec operation acting on a numpy ndarray of the selected charge sector.

        Parameters
        ----------
        vec : np.ndarray
            A length ``N`` vector (or ``N`` x 1 matrix) where ``N`` is the total dimension
            of the selected charge sector in the parent space, or the total dimension of the
            parent space if "all" charge sectors are selected.

        Returns
        -------
        matvec_vec : 1D ndarray
            The result of the linear operation as a length ``N`` vector

        """
        vec = np.asarray(vec)
        if vec.ndim != 1:  # convert Nx1 matrix to vector
            vec = np.squeeze(vec, axis=1)
            assert vec.ndim == 1
        tens = self.flat_array_to_tensor(vec)
        tens = self.lasyten_matvec(tens)
        self.matvec_count += 1
        return self.tensor_to_flat_array(tens)

    def _combine_vector_legs(self, tens: Tensor) -> Tensor:
        if tens.num_legs == 1:
            return tens
        return combine_legs(tens, list(range(tens.num_legs)), pipes=[self.pipe])

    def _split_vector_legs(self, tens: Tensor) -> Tensor:
        if len(self.legs) == 1:
            return tens
        return split_legs(tens, 0)

    def flat_array_to_tensor(self, vec: np.ndarray) -> Tensor | DirectSum:
        """Convert flat numpy data to a tensor (or DirectSum) in the selected charge sector."""
        assert vec.shape == (self.shape[1],)
        if self.component_converters is not None:
            parts = []
            offset = 0
            for conv, size in zip(self.component_converters, self._component_sizes):
                parts.append(conv.flat_array_to_tensor(vec[offset : offset + size]))
                offset += size
            return DirectSum(parts)
        block = self.backend.block_backend.block_from_numpy(vec)
        if self._charge_sector is None:
            # The numpy vector is the full dense state on the parent space. Represent it as
            # ChargedTensor(from_eye, charged_state=block): the identity invariant part maps the
            # charge leg onto the vector space, and charged_state holds the dense components.
            inv = SymmetricTensor.from_eye(
                [self.pipe],
                backend=self.backend,
                labels=['', ChargedTensor._CHARGE_LEG_LABEL],
                dtype=Dtype.from_numpy_dtype(self.dtype),
            )
            tens = ChargedTensor(inv, charged_state=block)
        elif isinstance(self._charge_sector, str) and self._charge_sector == 'trivial':
            tens = SymmetricTensor.from_dense_block_trivial_sector(vector=block, space=self.pipe, backend=self.backend)
        else:
            tens = ChargedTensor.from_dense_block_single_sector(
                vector=block,
                space=self.pipe,
                sector=self._charge_sector,
                backend=self.backend,
            )
        res = self._split_vector_legs(tens)
        if self.labels is not None:
            res.labels.set(self.labels)
        return res

    def tensor_to_flat_array(self, tens: Tensor | DirectSum) -> np.ndarray:
        """Convert a tensor (or DirectSum) in the selected charge sector to a flat numpy array."""
        if self.component_converters is not None:
            if not isinstance(tens, DirectSum):
                raise TypeError('Expected a DirectSum for this operator')
            if len(tens) != len(self.component_converters):
                raise ValueError('DirectSum component count does not match operator')
            parts = [conv.tensor_to_flat_array(comp) for conv, comp in zip(self.component_converters, tens.components)]
            return np.concatenate(parts)
        if self.labels is not None and all(self.labels) and all(l in tens.labels for l in self.labels):
            tens = permute_legs(tens, tens.labels.indices(self.labels))
        tens = self._combine_vector_legs(tens)
        if self._charge_sector is None:
            res = tens.to_dense_block(understood_braiding=True)
        elif isinstance(self._charge_sector, str) and self._charge_sector == 'trivial':
            res = tens.to_dense_block_trivial_sector()
        else:
            res = tens.to_dense_block_single_sector()
        res = self.backend.block_backend.to_numpy(res)
        res = np.reshape(res, (self.shape[0],))
        return res

    def eigenvectors(
        self,
        num_ev: int = 1,
        max_num_ev: int = None,
        max_tol: float = 1.0e-12,
        which: str = 'LM',
        v0_np: np.ndarray = None,
        v0_tensor: Tensor | DirectSum = None,
        cutoff: float = 1.0e-10,
        hermitian: bool = False,
        **kwargs,
    ):
        """Find the (dominant) eigenvector(s) of self using :func:`scipy.sparse.linalg.eigs`.

        If a charge_sector was specified, these are the dominant eigenvectors *within that sector*.
        Otherwise, we look in all charge sectors.

        Parameters
        ----------
        num_ev : int
            Number of eigenvalues/vectors to look for.
        max_num_ev : int
            :func:`scipy.sparse.linalg.speigs` sometimes raises a NoConvergenceError for small
            `num_ev`, which might be avoided by increasing `num_ev`. As a work-around,
            we try it again in the case of an error, just with larger `num_ev` up to `max_num_ev`.
            ``None`` defaults to ``num_ev + 2``.
        max_tol : float
            After the first `NoConvergenceError` we increase the `tol` argument to that value.
        which : str
            Which eigenvalues to look for, see :func:`scipy.sparse.linalg.eigs`.
            More details also in :func:`~lasyten.tools.misc.argsort`.
        v0_np : 1D ndarray
            Initial guess as a flat numpy array, i.e. a suitable input to :meth:`_matvec`.
        v0_tensor : :class:`~lasyten.tensors.Tensor` | :class:`~lasyten.tensors.ChargedTensor`
            Initial guess as a tensor, i.e. a suitable input to :meth:`tensor_to_flat_array`.
        cutoff : float
            Only used if ``self.charge_sector is None``; in that case it determines when entries in
            a given charge-block are considered nonzero, and what counts as degenerate.
        hermitian : bool
            If False (default), use :func:`scipy.sparse.linalg.eigs`
            If True, assume that self is hermitian and use :func:`scipy.sparse.linalg.eigsh`.
        **kwargs :
            Further keyword arguments given to :func:`scipy.sparse.linalg.eigsh` or
            :func:`scipy.sparse.linalg.eigs`, respectively.

        Returns
        -------
        eta : 1D ndarray
            The eigenvalues, sorted according to `which`.
        w : list of :class:`~lasyten.tensors.Tensor` or :class:`~lasyten.tensors.ChargedTensor`
            The corresponding eigenvectors as tensors.

        """
        if max_num_ev is None:
            max_num_ev = num_ev + 2
        if v0_tensor is not None:
            assert v0_np is None
            v0_np = self.tensor_to_flat_array(v0_tensor)
        if v0_np is not None:
            kwargs['v0'] = v0_np

        for k in range(num_ev, max_num_ev + 1):
            if k > num_ev:
                warnings.warn(f'Increasing `num_ev` to {k}')
            try:
                if hermitian:
                    eta, A = speigsh(self, k=k, which=which, **kwargs)
                else:
                    eta, A = speigs(self, k=k, which=which, **kwargs)
                break
            except ArpackNoConvergence:
                if k == max_num_ev:
                    raise
            kwargs['tol'] = max(max_tol, kwargs.get('tol', 0))
        cutoff = max(cutoff, 10 * kwargs.get('tol', 1.0e-16))
        A = np.real_if_close(A)

        vecs = [self.flat_array_to_tensor(A[:, j]) for j in range(A.shape[1])]

        perm = argsort(eta, which)
        return np.array(eta)[perm], [vecs[j] for j in perm]


class HermitianNumpyArrayLinearOperator(NumpyArrayLinearOperator):
    """Hermitian variant of :class:`NumpyArrayLinearOperator`.

    Note that we don't check hermicity of :meth:`matvec`.
    """

    def _adjoint(self):
        return self

    def eigenvectors(self, *args, **kwargs):
        kwargs['hermitian'] = True
        return NumpyArrayLinearOperator.eigenvectors(self, *args, **kwargs)
