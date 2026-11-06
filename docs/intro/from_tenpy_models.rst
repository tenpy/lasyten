From TeNPy models to LASyTen
==========================

:doc:`from_np_conserved` covers the linear algebra: how to replace
:external+tenpy_v1:py:mod:`tenpy.linalg.np_conserved` with LASyTen tensors. This page is
the **high-level** counterpart — sites, few-site operators, fermions, and
the planar diagrams that replace hand-written ``tensordot`` chains in
algorithms such as DMRG's effective Hamiltonian.

TeNPy v2 is being rewritten against these LASyTen objects. The notes below are
the LASyTen side of that conversion.

If you are new to LASyTen, start with :doc:`first_steps`. The scripts under
``docs/intro/examples/`` are compact versions of the examples on this page.
Code snippets below assume ``import lasyten as lasy``.


Sites still exist, operators do not all survive
-----------------------------------------------

A :class:`~lasyten.models.Site` is still the local Hilbert space plus a
dictionary of **symmetric** on-site operators
(:attr:`~lasyten.models.Site.onsite_operators`). The usual constructors live in
:mod:`lasyten.models` (:class:`~lasyten.models.SpinSite`,
:class:`~lasyten.models.SpinlessFermionSite`,
:class:`~lasyten.models.SpinHalfFermionSite`, …).

The change that affects model code:

- Operators that conserve the symmetry (``Sz``, ``N``, ``Id``, density-density
  pieces, …) stay on the site, as :class:`~lasyten.tensors.SymmetricTensor`\s.
- Operators that *change* the charge — TeNPy's ``Sp`` / ``Sm`` at conserved
  :math:`S^z`, and especially ``C`` / ``Cd`` — are **not** on-site operators.
  They cannot be :class:`~lasyten.tensors.SymmetricTensor`\s. In TeNPy they
  were arrays with a non-zero ``qtotal`` and a ``need_JW`` flag. In LASyTen a
  lone charged map is a :class:`~lasyten.tensors.ChargedTensor`; a *product* of
  them that is overall symmetric is a :class:`~lasyten.models.Coupling`.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - TeNPy v1
     - LASyTen
   * - :external+tenpy_v1:py:class:`~tenpy.networks.site.SpinHalfSite`
     - :class:`~lasyten.models.SpinSite` ``(S=0.5, ...)``
   * - :external+tenpy_v1:py:class:`~tenpy.networks.site.FermionSite`
     - :class:`~lasyten.models.SpinlessFermionSite`
   * - ``site.get_op('Sz')`` / ``site.Sz``
     - ``site.onsite_operators['Sz']`` / ``site.get_op('Sz')``
   * - ``site.get_op('Cd')`` (charged, ``need_JW=True``)
     - not an on-site operator; use a :class:`~lasyten.models.Coupling`
   * - ``site.op_needs_JW('Cd')``
     - no analogue (see :ref:`no_jw_on_mps`)
   * - :external+tenpy_v1:py:meth:`~tenpy.models.model.CouplingModel.add_coupling`
     - build a :class:`~lasyten.models.Coupling`, then place it on the lattice


Couplings are mini-MPOs
-----------------------

A :class:`~lasyten.models.Coupling` is a few-site operator stored as **one
tensor per site**, i.e. a short MPO. Graphically the operator is the usual
multi-site map

::

    |        p0*  p1*  ..  pN*
    |        │    │    │    │
    |       ┏┷━━━━┷━━━━┷━━━━┷┓
    |       ┃       h        ┃
    |       ┗┯━━━━┯━━━━┯━━━━┯┛
    |        │    │    │    │
    |        p0   p1   ..   pN

and the factorization that LASyTen actually stores is

::

    |        p0*         p1*               pN*
    |         │           │                 │
    |     wL ┏┷━┓ wR  wL ┏┷━┓ wR        wL ┏┷━┓ wR
    |     ───┃0 ┃────────┃1 ┃──── ···  ────┃N ┃───
    |        ┗┯━┛        ┗┯━┛              ┗┯━┛
    |         │           │                 │
    |         p0          p1                pN

Each ``factorization[i]`` is a :class:`~lasyten.tensors.SymmetricTensor` with
labels ``['wL', 'p', 'wR', 'p*']``. The physical pair ``p, p*`` is the
:attr:`~lasyten.models.Site.leg` of ``sites[i]``. Contracting the virtual bonds
``wR`` of site :math:`i` with ``wL`` of site :math:`i+1` recovers ``h``.

Those virtual legs are the high-level replacement for TeNPy's ``qtotal``:
the charge that ``Sp`` or ``Cd`` would have carried by itself lives on the
bond *between* the factors. The whole coupling is a symmetric tensor; the
individual factors are too, because the virtual legs make the local maps
charge-neutral.

Useful methods:

:meth:`~lasyten.models.Coupling.to_tensor`
    Contract the virtual bonds. Returns a
    :class:`~lasyten.tensors.SymmetricTensor` with labels ``p0, p1, …``
    (codomain) and ``p0*, p1*, …`` (domain).
:meth:`~lasyten.models.Coupling.from_tensor`
    Split a multi-site :class:`~lasyten.tensors.SymmetricTensor` with
    :func:`~lasyten.tensors.horizontal_factorization`.
:meth:`~lasyten.models.Coupling.from_dense_block`
    Same, from a numpy (or backend) block whose axes are
    ``p0, p1, …, p1*, p0*``.
:meth:`~lasyten.models.Coupling.stretch_with_identities`
    Place the factors on a longer chain, filling every site in between with
    :meth:`~lasyten.models.Site.identity_tensor`. That is how a two-site
    :math:`c^\dagger_i c_j` becomes an operator on all sites
    :math:`i, i+1, \ldots, j`.

Factories such as :func:`~lasyten.models.heisenberg_coupling`,
:func:`~lasyten.models.hopping`, and
:func:`~lasyten.models.chiral_3spin_coupling` build the factorization for you.


Examples
--------

Heisenberg
~~~~~~~~~~

The two-site coupling :math:`h_{ij} = J\, \vec{S}_i \cdot \vec{S}_j` is

.. code-block:: python

    site0 = lasy.models.SpinSite(S=0.5, conserve='Sz')
    site1 = lasy.models.SpinSite(S=0.5, conserve='Sz')
    heisenberg = lasy.models.heisenberg_coupling([site0, site1], J=1)
    H = heisenberg.to_tensor()          # four-leg map  p⊗p → p⊗p

With conserved :math:`S^z`, the ``Sz Sz`` piece is charge-neutral while
``Sp Sm`` (and ``Sm Sp``) exchange charge :math:`\pm 2` between the two
sites. That charge sits on the virtual bond: ``heisenberg.factorization``
is two tensors whose ``wR`` / ``wL`` spaces include those sectors. In TeNPy
you would have written ``add_coupling(0.5, 0, 'Sp', 0, 'Sm', 1, plus_hc=True)``
plus a separate ``Sz Sz`` term; here it is one coupling.

:doc:`first_steps` diagonalizes this Hamiltonian.

Hopping :math:`c^\dagger c`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fermion hopping in a Hamiltonian is the combination
:math:`t_\mathrm{left}\,c^\dagger_i c_j + t_\mathrm{right}\,c^\dagger_j c_i`:

.. code-block:: python

    ferm = lasy.models.SpinlessFermionSite(num_species=1, conserve='N')
    hop = lasy.models.hopping([ferm, ferm], t_left=1, t_right=1)

A correlation function :math:`c^\dagger_i c_j` is the same geometry **without**
the hermitian conjugate: one creation on the left, one annihilation on the
right, connected by a virtual bond that carries one fermion (odd
:class:`~lasyten.symmetries.FermionParity`, and :math:`N = 1` if number is
conserved). ``C`` and ``Cd`` are not on-site operators, so the coupling is
built from the dense two-site block (axes ``p0, p1, p1*, p0*``):

.. code-block:: python

    Cd = ferm.get_creator_numpy(species=0, include_JW=True)
    C = ferm.get_annihilator_numpy(species=0, include_JW=True)
    h = (Cd @ ferm.JW)[:, None, None, :] * C[None, :, :, None]
    cdag_c = lasy.Coupling.from_dense_block(
        h, [ferm, ferm], name='Cd C', understood_braiding=True
    )

``include_JW=True`` and the extra ``ferm.JW`` on the *left* factor are the
**on-site** JW dressing (TeNPy's ``"Cd JW"`` on site :math:`i`). They are
*not* the string on the sites between :math:`i` and :math:`j` — that string
is the next section.

.. todo ::
  revisit this.


Chiral three-spin
~~~~~~~~~~~~~~~~~

The scalar triple product
:math:`h_{ijk} = \chi\, \vec{S}_i \cdot (\vec{S}_j \times \vec{S}_k)` is a
three-site coupling, TeNPy's ``add_multi_coupling`` case:

.. code-block:: python

    chiral = lasy.models.chiral_3spin_coupling([site0, site0, site0], chi=1)
    # chiral.num_sites == 3, two virtual bonds in the mini-MPO

Order of ``sites`` is the operator order. The result is Hermitian and
traceless.


.. _no_jw_on_mps:

No Jordan-Wigner strings on the MPS
-----------------------------------

TeNPy v1 represents fermions as spins plus an extra bookkeeping layer. The
:external+tenpy_v1:doc:`intro/JordanWigner` userguide is the reference for that
convention. The physical operators are *global*:

.. math::

    c_j \;\leftrightarrow\;
    \Bigl(\prod_{l < j} \mathrm{JW}_l\Bigr)\, C_j\,,
    \qquad
    \mathrm{JW}_l = (-1)^{n_l}\,.

So an MPS that stores only local ``C`` / ``Cd`` tensors is incomplete. Every
algorithm that inserts those operators — building an MPO,
:external+tenpy_v1:py:meth:`~tenpy.networks.mps.MPS.correlation_function`,
:external+tenpy_v1:py:meth:`~tenpy.networks.mps.MPS.expectation_value_term`,
``apply_local_op``, … — has to multiply extra ``JW`` tensors onto the
physical legs of the sites to the left (or in between). Sites carry
``need_JW_string``; ``add_coupling`` defaults to ``op_string='JW'``.

LASyTen does **not** do that. Fermions are a symmetry with a non-trivial
braid: :class:`~lasyten.symmetries.FermionParity` (always present on
:class:`~lasyten.models.SpinlessFermionSite` /
:class:`~lasyten.models.SpinHalfFermionSite`; ``conserve='None'`` is invalid).
The swap of two odd-parity legs is :math:`-1`. The MPS is an ordinary
tensor network whose physical legs transform under that symmetry. There is
no ``JW`` operator on the MPS, no ``need_JW``, and no
``apply_JW_string_left_of_virt_leg``.

The signs still exist: they are produced when **legs cross** in a diagram.
A correlation function :math:`c^\dagger_i c_j` is specified as the coupling
above, with a **non-trivial virtual leg** between the two factors. That
virtual bond carries the odd fermion that was created on the left and has
not yet been annihilated on the right. Putting the coupling on a chain
with sites in between means that odd line has to pass those sites — and
therefore has to cross their physical legs.

The identity that fills a gap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~lasyten.models.Coupling.stretch_with_identities` places ``Cd`` and
``C`` on chosen sites and fills every site in between with
:meth:`~lasyten.models.Site.identity_tensor`:

.. code-block:: python

    # c^dagger_0 c_2  on a three-site chain: identity on site 1.
    stretched = cdag_c.stretch_with_identities([ferm, ferm, ferm], [0, 2])

::

    |       p0*          p1*          p2*
    |        │            │            │
    |    wL ┏┷━┓ wR   wL ┏┷━┓ wR   wL ┏┷━┓ wR
    |    ───┃Cd┃─────────┃Id┃─────────┃C ┃───
    |       ┗┯━┛         ┗┯━┛         ┗┯━┛
    |        │            │            │
    |        p0           p1           p2

The middle tensor is *not* a Kronecker product of ``Id`` on the physical
space with ``Id`` on the virtual space. It is built as the identity on
:math:`p \otimes w` and then **permuted** so that :math:`w` runs left–right
while :math:`p` runs down. That permutation braids the virtual leg past the
physical one:

::

    |     p*    w*                   wL      p*
    |     │     │                      ╲    ╱
    |    ┏┷━━━━━┷┓                      ╲  ╱
    |    ┃  Id   ┃     permute           X
    |    ┗┯━━━━━┯┛     w past p         ╱ ╲
    |     │     │                      ╱   ╲
    |     p     w                     p       wR

For an even virtual sector the crossing is trivial. For an **odd** virtual
sector — exactly the bond of :math:`c^\dagger_i c_j` — the swap gate is
:math:`(-1)^{n_p}`. Packed back into an MPO tensor, that is TeNPy's local
``JW`` operator.

In the example script the virtual bond of ``cdag_c`` is one-dimensional
(one fermion), and the middle factor's dense block *is* ``ferm.JW``:

.. code-block:: python

    wR = cdag_c.factorization[0].get_leg_co_domain('wR')
    Id_middle = ferm.identity_tensor(wR)
    jw_block = Id_middle.to_numpy(understood_braiding=True)
    # jw_block[0, :, 0, :]  ==  ferm.JW  ==  diag(+1, -1)

TeNPy v1 would have inserted the same operator by hand on the physical leg
of every site between :math:`i` and :math:`j`::

    ["Cd JW", "JW", "C"]     # i < j = i+2, TeNPy on-site names

LASyTen inserts nothing on the MPS. The ``JW`` is the crossing *inside* the
coupling's virtual line.

What this means for measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :math:`\langle c^\dagger_i c_j \rangle` is the expectation value of the
  **stretched coupling**, not of two on-site operators plus a JW string.
- Density-density :math:`n_i n_j` has an even (usually trivial) virtual
  bond. Stretching it still fills identities, but those identities braid
  an even line and introduce no signs — the analogue of TeNPy's "JW
  strings cancel".
- The virtual bond of a coupling is the charge *in flight*, not the
  total charge of the operator. :math:`c^\dagger_i c^\dagger_j` is even
  overall, yet the bond *between* the two creators is odd (one fermion
  has been created, the second has not). Intermediate identities then
  contribute JW signs, same as for :math:`c^\dagger_i c_j`.
- Fermionic sites require a backend that can braid (the fusion-tree
  backend). The abelian backend that replaced ``np_conserved`` is not
  enough, which is why
  :class:`~lasyten.models.SpinlessFermionSite` refuses
  ``NoSymmetryBackend`` / ``AbelianBackend``.

.. todo ::
  :pull:`225` is about adressing this for the AbelianBackend.

The same crossing rule is why :meth:`~lasyten.models.Coupling.permute` of a
fermionic coupling picks up a minus sign when two odd factors are swapped,
without anyone multiplying by ``JW``.


Planar diagrams
---------------

TeNPy v1 algorithms are written as a sequence of ``npc.tensordot`` and
``itranspose``. That is fine when legs commute. In LASyTen a
:func:`~lasyten.tensors.permute_legs` that makes two fermion (or anyon) lines
cross is a *braid*, and you must specify its chirality. The networks that
show up in MPS algorithms — applying a two-site gate, contracting an MPO
environment, the DMRG effective Hamiltonian — are drawn **without**
crossings. Those contractions should be written as a
:class:`~lasyten.tensors.PlanarDiagram`, not as a chain of
:func:`~lasyten.tensors.tdot` / :func:`~lasyten.tensors.compose` plus
``bend_right``.

A planar diagram names every tensor, lists every contraction and every open
leg by **label**, and forgets the (co)domain split. Cyclic permutations of
a tensor's labels are allowed (they are planar). The contraction is
connected and braid-free; LASyTen checks that at construction. Evaluating
the diagram on concrete tensors never introduces a swap gate.

Syntax
~~~~~~

.. code-block:: python

    diagram = lasy.PlanarDiagram(
        tensors='theta[vL, p0, p1, vR], U[p0, p1, p1*, p0*]',
        definition=(
            'theta:p0 @ U:p0*, theta:p1 @ U:p1*, '
            'theta:vL -> vL, theta:vR -> vR, U:p0 -> p0, U:p1 -> p1'
        ),
        dims=dict(chi=['vL', 'vR'], d=['p0', 'p0*', 'p1', 'p1*']),
    )
    theta_new = diagram.evaluate(dict(theta=theta, U=U))
    # equivalent: diagram(theta=theta, U=U)

- ``tensors``: ``name[leg, leg, ...]`` entries, comma-separated. Leg order
  is the conventional counter-clockwise order around the tensor, not
  ``Tensor.legs``.
- ``definition``: ``A:leg @ B:leg`` contracts two legs; ``A:leg -> new``
  leaves an open leg of the result.
- ``dims``: optional symbols for a cost polynomial (used when optimizing
  the contraction order).
- ``order``: ``'greedy'`` (default), ``'optimal'``, ``'definition'``, or a
  hard-coded tree. Instantiating a diagram with an optimized order can be
  expensive: build it **once**, as a class or module attribute, and hard-code
  ``order`` once you know it.

The result of :meth:`~lasyten.tensors.PlanarDiagram.evaluate` is determined
only up to a cyclic permutation of the open legs. Bring it to a definite
(co)domain with :func:`~lasyten.tensors.planar_permute_legs` (the planar
analogue of :func:`~lasyten.tensors.permute_legs`; it refuses non-planar
moves instead of asking for ``bend_right``).

:meth:`~lasyten.tensors.PlanarDiagram.add_tensor` /
:meth:`~lasyten.tensors.PlanarDiagram.remove_tensor` build a related diagram
without repeating the contractions you already have. That is the intended
way to go from "the operator" to "the operator acting on a vector".

EffectiveH as a planar linear operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TeNPy's :external+tenpy_v1:py:class:`~tenpy.algorithms.mps_common.TwoSiteH` is the
two-site DMRG effective Hamiltonian

::

    |        .---       ---.
    |        |    |   |    |
    |       LP----W0--W1---RP
    |        |    |   |    |
    |        .---       ---.

and :external+tenpy_v1:py:meth:`~tenpy.algorithms.mps_common.TwoSiteH.matvec` is four
``tensordot``\ s plus an ``itranspose``. The LASyTen toycode
``toycodes/tenpy_toycodes/d_dmrg.py`` still has a ``HEffective.matvec``
written that way, with an explicit ``bend_right`` on every
:func:`~lasyten.tensors.permute_legs`. That style does not scale to fermions:
one missed braid is a wrong sign.

The replacement is :class:`~lasyten.tensors.PlanarLinearOperator`, a
:class:`~lasyten.tensors.sparse.LinearOperator` whose
:meth:`~lasyten.tensors.sparse.LinearOperator.matvec` and
:meth:`~lasyten.tensors.sparse.LinearOperator.to_tensor` are two planar
diagrams:

- ``op_diagram`` — the network above (open legs ``vL, p0, p1, vR`` and
  their duals): the operator as a tensor.
- ``matvec_diagram`` — the same network with a ``theta`` plugged into the
  open ket legs: :math:`H_{\mathrm{eff}} |\theta\rangle`.

Define both diagrams as **class attributes**. Obtain the matvec diagram by
adding ``theta`` to the operator diagram (or, equivalently, define the
matvec diagram first and :meth:`~lasyten.tensors.PlanarDiagram.remove_tensor`
``theta`` to get the operator):

.. code-block:: python

    class TwoSiteEffectiveH(lasy.PlanarLinearOperator):
        op_diagram = lasy.PlanarDiagram(
            tensors='Lp[vR*, wR, vR], W0[wL, p, wR, p*], W1[wL, p, wR, p*], Rp[vL*, vL, wL]',
            definition=(
                'Lp:vR* -> vL, Lp:wR @ W0:wL, Lp:vR -> vL*, '
                'W0:p -> p0, W0:wR @ W1:wL, W0:p* -> p0*, '
                'W1:p -> p1, W1:wR @ Rp:wL, W1:p* -> p1*, '
                'Rp:vL* -> vR, Rp:vL -> vR*'
            ),
            dims=dict(chi=['vR', 'vR*', 'vL', 'vL*'], w=['wL', 'wR'], d=['p', 'p*']),
        )
        matvec_diagram = op_diagram.add_tensor(
            tensor='theta[vL, p0, p1, vR]',
            extra_definition=(
                'theta:vL @ Lp:vR, theta:p0 @ W0:p*, '
                'theta:p1 @ W1:p*, theta:vR @ Rp:vL'
            ),
            extra_dims=dict(chi=['vL', 'vR'], d=['p0', 'p1']),
        )

        def __init__(self, Lp, W0, W1, Rp):
            super().__init__(
                op_diagram=self.op_diagram,
                matvec_diagram=self.matvec_diagram,
                op_tensors=dict(Lp=Lp, W0=W0, W1=W1, Rp=Rp),
                vec_name='theta',
            )

``W0`` / ``W1`` are MPO tensors, i.e. the factors of a
:class:`~lasyten.models.Coupling` (or a full MPO built from them), with the
usual ``wL, p, wR, p*`` labels. ``Lp`` / ``Rp`` are the left and right
environments. :meth:`~lasyten.tensors.sparse.LinearOperator.matvec` is then
just ``Heff.matvec(theta)``, suitable for
:mod:`~lasyten.tensors.krylov_based` Lanczos. After ``Heff.to_tensor()``,
fix the cyclic permutation::

    H = lasy.planar_permute_legs(Heff.to_tensor(), codomain=['vL', 'p0', 'p1', 'vR'])

The same pattern applies to one-site and zero-site effective Hamiltonians
(fewer ``W`` tensors) and to the environment updates themselves. The
planar DMRG toycode stores ``update_LP_diagram`` / ``update_RP_diagram`` as
class attributes and evaluates them as ``self.update_RP_diagram(Rp=...,
W=..., B=..., B_hc=...)``.

What not to do
~~~~~~~~~~~~~~

- Do not port ``TwoSiteH.matvec`` as a sequence of
  :func:`~lasyten.tensors.tdot` and :func:`~lasyten.tensors.permute_legs` with
  guessed ``bend_right``. If the network is planar, write a
  :class:`~lasyten.tensors.PlanarDiagram`.
- Do not construct the diagram inside ``__init__`` or inside ``matvec``
  (re-optimizing the order on every Lanczos step).
- Do not use :func:`~lasyten.tensors.combine_legs` to "make a matrix" before
  Lanczos. :class:`~lasyten.tensors.PlanarLinearOperator` already acts on the
  uncombined ``theta``.
- For the rare network that *does* braid (a JW line crossing a physical
  leg, as in the previous section), a planar diagram will refuse it. That
  crossing belongs in :meth:`~lasyten.models.Site.identity_tensor` /
  :func:`~lasyten.tensors.permute_legs`, not in the DMRG contraction.


Porting checklist
-----------------

1. Replace TeNPy :external+tenpy_v1:py:class:`~tenpy.networks.site.Site` subclasses with
   :mod:`lasyten.models` sites. Drop ``need_JW`` when adding custom operators;
   only symmetric maps belong in ``onsite_operators``.
2. Replace ``add_coupling`` / ``add_multi_coupling`` / ``add_onsite`` with
   :class:`~lasyten.models.Coupling` factories, or
   :meth:`~lasyten.models.Coupling.from_tensor` of a symmetric few-site map.
3. Treat ``Sp``, ``Sm``, ``C``, ``Cd`` as factors of a coupling, not as
   tensors you insert into an MPS. The virtual bond carries the old
   ``qtotal``.
4. For :math:`c^\dagger_i c_j` (Hamiltonian or correlator), build a coupling
   with a non-trivial virtual leg and
   :meth:`~lasyten.models.Coupling.stretch_with_identities` across the sites
   in between. Do not multiply ``JW`` onto MPS physical legs.
5. Delete ``op_string='JW'``, ``str_on_first``, ``autoJW``, and
   ``apply_JW_string_left_of_virt_leg``. The braid of
   :class:`~lasyten.symmetries.FermionParity` replaces them.
6. Use the fusion-tree backend for fermions. Keep the abelian backend for
   spin / boson models that never braid.
7. Replace :external+tenpy_v1:py:class:`~tenpy.algorithms.mps_common.TwoSiteH` /
   ``OneSiteH`` ``matvec`` chains with a
   :class:`~lasyten.tensors.PlanarLinearOperator`. Store the
   :class:`~lasyten.tensors.PlanarDiagram`\ s as class attributes. Use
   :func:`~lasyten.tensors.planar_permute_legs` after ``evaluate`` /
   ``to_tensor``.
8. Reactivate tests for what has been ported.

Worked example: couplings
-------------------------

.. literalinclude:: examples/from_tenpy_couplings.py
   :language: python

Worked example: two-site EffectiveH
-----------------------------------

.. literalinclude:: examples/from_tenpy_planar.py
   :language: python
