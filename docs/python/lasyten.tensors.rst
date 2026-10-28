tensors
=======

- full name: lasyten.tensors
- parent module: :mod:`lasyten`
- type: module

.. py:module:: lasyten.tensors

Tensor types and operations. Bindings live in ``pybind/tensors/``.

Classes and functions imported from the pybind11 module :mod:`lasyten._core`.

.. toctree::
   :maxdepth: 1

   tensors/sparse
   tensors/planar
   tensors/krylov_based

.. rubric:: Inheritance Diagram

.. lasyten-inheritance-diagram::

lasyten.tensors.LegLabels
-----------------------

.. autoclass:: lasyten.tensors.LegLabels
   :members:
   :undoc-members:
   :show-inheritance:

lasyten.tensors.Tensor
--------------------

.. autoclass:: lasyten.tensors.Tensor
   :members:
   :undoc-members:
   :show-inheritance:

lasyten.tensors.SymmetricTensor
-----------------------------

.. autoclass:: lasyten.tensors.SymmetricTensor
   :members:
   :undoc-members:
   :show-inheritance:

lasyten.tensors.DiagonalTensor
----------------------------

.. autoclass:: lasyten.tensors.DiagonalTensor
   :members:
   :undoc-members:
   :show-inheritance:

lasyten.tensors.Identity
----------------------

.. autoclass:: lasyten.tensors.Identity
   :members:
   :undoc-members:
   :show-inheritance:

lasyten.tensors.Mask
------------------

.. autoclass:: lasyten.tensors.Mask
   :members:
   :undoc-members:
   :show-inheritance:

lasyten.tensors.ChargedTensor
---------------------------

.. autoclass:: lasyten.tensors.ChargedTensor
   :members:
   :undoc-members:
   :show-inheritance:

lasyten.tensors.HiddenLegTensor
-----------------------------

.. autoclass:: lasyten.tensors.HiddenLegTensor
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: lasyten.tensors.add_trivial_leg

.. autofunction:: lasyten.tensors.almost_equal

.. autofunction:: lasyten.tensors.angle

.. autofunction:: lasyten.tensors.apply_mask

.. autofunction:: lasyten.tensors.apply_mask_DiagonalTensor

.. autofunction:: lasyten.tensors.bend_legs

.. autofunction:: lasyten.tensors.check_same_legs

.. autofunction:: lasyten.tensors.combine_legs

.. autofunction:: lasyten.tensors.combine_to_matrix

.. autofunction:: lasyten.tensors.complex_conj

.. autofunction:: lasyten.tensors.compose

.. autofunction:: lasyten.tensors.cutoff_inverse

.. autofunction:: lasyten.tensors.dagger

.. autofunction:: lasyten.tensors.eigh

.. autofunction:: lasyten.tensors.eig

.. autofunction:: lasyten.tensors.eigvalsh

.. autofunction:: lasyten.tensors.eigvals

.. autofunction:: lasyten.tensors.enlarge_leg

.. autofunction:: lasyten.tensors.entropy

.. autofunction:: lasyten.tensors.exp

.. autofunction:: lasyten.tensors.eye

.. autofunction:: lasyten.tensors.get_same_device

.. autofunction:: lasyten.tensors.imag

.. autofunction:: lasyten.tensors.inner

.. autofunction:: lasyten.tensors.is_scalar

.. autofunction:: lasyten.tensors.is_valid_leg_label

.. autofunction:: lasyten.tensors.item

.. autofunction:: lasyten.tensors.linear_combination

.. autofunction:: lasyten.tensors.lq

.. autofunction:: lasyten.tensors.move_hidden_leg

.. autofunction:: lasyten.tensors.move_leg

.. autofunction:: lasyten.tensors.norm

.. autofunction:: lasyten.tensors.on_device

.. autofunction:: lasyten.tensors.outer

.. autofunction:: lasyten.tensors.partial_compose

.. autofunction:: lasyten.tensors.partial_trace

.. autofunction:: lasyten.tensors.permute_legs

.. autofunction:: lasyten.tensors.pinv

.. autofunction:: lasyten.tensors.qr

.. autofunction:: lasyten.tensors.real

.. autofunction:: lasyten.tensors.real_if_close

.. autofunction:: lasyten.tensors.scalar_multiply

.. autofunction:: lasyten.tensors.scale_axis

.. autofunction:: lasyten.tensors.slice_leg

.. autofunction:: lasyten.tensors.split_legs

.. autofunction:: lasyten.tensors.sqrt

.. autofunction:: lasyten.tensors.squeeze_legs

.. autofunction:: lasyten.tensors.stable_log

.. autofunction:: lasyten.tensors.svd

.. autofunction:: lasyten.tensors.svd_apply_mask

.. autofunction:: lasyten.tensors.tdot

.. autofunction:: lasyten.tensors.tensor

.. autofunction:: lasyten.tensors.tensor_from_grid

.. autofunction:: lasyten.tensors.trace

.. autofunction:: lasyten.tensors.transpose

.. autofunction:: lasyten.tensors.truncate_singular_values

.. autofunction:: lasyten.tensors.truncated_svd

.. autofunction:: lasyten.tensors.zero_like
