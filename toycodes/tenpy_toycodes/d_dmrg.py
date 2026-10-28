"""Toy code implementing the 2-site DMRG algorithm.

Note that this toy code uses the outdated convention of kets pointing upwards in the diagrams.
"""

# Copyright (C) TeNPy Developers, Apache license

import lasyten as lasy
from toycodes.tenpy_toycodes.a_mps import SimpleMPS, split_truncate_theta
from toycodes.tenpy_toycodes.b_model import TFIModel


class HEffective(lasy.sparse.LinearOperator):
    """Class for the effective Hamiltonian.

    To be diagonalized in `DMRGEngine.update_bond`. Looks like this::

        .--vL             vR--.
        |       i     j       |
        |       |     |       |
        (LP)---(W1)--(W2)----(RP)
        |       |     |       |
        |       i*    j*      |
        .--vL*           vR*--.

    Parameters
    ----------
    LP : lasy.SymmetricTensor
        Left environment with legs `'vL', 'wL*', 'vL*'`, where `'wL*'` is contrated with the MPO
        tensors and `'vL*'` with the MPS tensors. The leg `'vL'` is expected to be in the codomain,
        `'wL*', 'vL*'` in the domain.
    RP : lasy.SymmetricTensor
        Right environment with legs `'vR', 'wR*', 'vR*'`, where `'wR*'` is contrated with the MPO
        tensors and `'vR*'` with the MPS tensors. The leg `'vR'` is expected to be in the codomain,
        `'wR*', 'vR*'` in the domain.
    W1, W2 : lasy.SymmetricTensor
        MPO tensors of the two-site effective Hamiltonian, each with legs ``'wL', 'p', 'wR', 'p*'``
        with the codomain and domain consisting of `'wL', 'p'` and `'wR', 'p*'`, respectively.

    Attributes
    ----------
        LP, RP, W1, W2: same as parameters, but with the following changed legs in the (co)domain:
        LP : codomain: `'vL', 'wL*'`, domain: `'vL*'`.
        RP : codomain: `'vR*', 'wR*'`, domain: `'vR'`.
        W1 : codomain: `'p0', 'wC'`, domain: 'wL', 'p0*'
            (with renaming `'p', 'p*', 'wR'` to `'p0', 'p0*', 'wC'`)
        W2 : codomain: `'p1', 'wR'`, domain: 'wC', 'p1*'
            (with renaming `'p', 'p*', 'wL'` to `'p1', 'p1*', 'wC'`)

    """

    def __init__(self, LP, RP, W1, W2):
        # C++ LinearOperator requires its __init__ when a Python subclass overrides __init__.
        super().__init__(vector_legs=[], dtype=LP.dtype, vector_labels=None)
        # bend such that we can directly compose it with theta
        self.LP = lasy.permute_legs(LP, ['vL', 'wL*'], ['vL*'], bend_right=True)  # vL wL* vL*
        self.RP = lasy.permute_legs(
            RP, ['vR*', 'wR*'], ['vR'], bend_right=[True, False, False]
        )  # vR vR* wR* -> vR* wR* vR
        self.W1 = lasy.permute_legs(
            W1, ['p', 'wR'], ['wL', 'p*'], bend_right=[False, None, True, None]
        )  # wL i wC i* -> i wC i* wL
        self.W1.labels.update({'p': 'p0', 'p*': 'p0*', 'wR': 'wC'})
        self.W2 = lasy.permute_legs(
            W2, ['p', 'wR'], ['wL', 'p*'], bend_right=[False, None, True, None]
        )  # wC j wR j* -> j wR j* wC
        self.W2.labels.update({'p': 'p1', 'p*': 'p1*', 'wL': 'wC'})

    def matvec(self, theta: lasy.Tensor) -> lasy.Tensor:
        """Calculate |theta'> = H_eff |theta>"""
        # get_theta2(i) has 2 legs in codomain and 2 legs domain
        x = lasy.permute_legs(theta, ['vL'], ['vR', 'p1', 'p0'], bend_right=True)  # vL p0 p1 vR
        x = lasy.compose(self.LP, x)  # vL wL* p0 p1 vR
        x = lasy.permute_legs(
            x, ['wL*', 'p0'], ['vL', 'vR', 'p1'], bend_right=[False, None, True, None, None]
        )  # wL* p0 p1 vR vL
        x = lasy.compose(self.W1, x)  # p0 wC p1 vR vL
        x = lasy.permute_legs(
            x, ['wC', 'p1'], ['p0', 'vL', 'vR'], bend_right=[False, None, True, None, None]
        )  # wC p1 vR vL p0
        x = lasy.compose(self.W2, x)  # p1 wR vR vL p0
        x = lasy.permute_legs(
            x, ['vL', 'p0', 'p1'], ['vR', 'wR'], bend_right=[None, True, None, False, False]
        )  # vL p0 p1 wR vR
        x = lasy.compose(x, self.RP)  # vL p0 p1 vR
        x = lasy.permute_legs(x, domain=['vR', 'p1'], bend_right=True)  # vL p0 p1 vR
        return x

    def to_tensor(self) -> lasy.Tensor:
        raise NotImplementedError


class PlanarHEffective(lasy.PlanarLinearOperator):
    """Same as ``HEffective``, but as ``PlanarLinearOperator``."""

    dmrg_diagram = lasy.PlanarDiagram(
        tensors='Lp[vL, wL*, vL*], W0[wL, p, wR, p*], W1[wL, p, wR, p*], Rp[vR*, wR*, vR], theta[vL, p0, p1, vR]',
        definition='Lp:vL -> vL, Lp:vL* @ theta:vL, Lp:wL* @ W0:wL, '
        'W0:p -> p0, W0:p* @ theta:p0, W0:wR @ W1:wL, '
        'W1:p -> p1, W1:p* @ theta:p1, W1:wR @ Rp:wR*, '
        'Rp:vR -> vR, Rp:vR* @ theta:vR',
        dims=dict(chi=['vR', 'vR*', 'vL', 'vL*'], w=['wL', 'wL*', 'wR', 'wR*'], d=['p', 'p*', 'p0', 'p1']),
    )
    op_diagram = dmrg_diagram.remove_tensor(
        name='theta', extra_definition='Lp:vL* -> vL*,  W0:p* -> p0*, W1:p* -> p1*, Rp:vR* -> vR*'
    )

    def __init__(self, Lp, W0, W1, Rp):
        lasy.planar.PlanarLinearOperator.__init__(
            self,
            op_diagram=self.op_diagram,
            matvec_diagram=self.dmrg_diagram,
            op_tensors=dict(Lp=Lp, W0=W0, W1=W1, Rp=Rp),
            vec_name='theta',
        )


class DMRGEngine:
    """DMRG algorithm, implemented as class holding the necessary data.

    Parameters
    ----------
    psi, model, chi_max, max_E_err, eps:
        See attributes

    Attributes
    ----------
    psi : MPS
        The current ground-state (approximation).
    model :
        The model of which the groundstate is to be calculated.
    chi_max, eps:
        Truncation parameters, see :func:`a_mps.split_truncate_theta`.
    LPs, RPs : list of lasy.SymmetricTensor
        Left and right parts ("environments") of the effective Hamiltonian.
        ``LPs[i]`` is the contraction of all parts left of site `i` in the network ``<psi|H|psi>``,
        and similar ``RPs[i]`` for all parts right of site `i`.
        Each ``LPs[i]`` has legs ``vL wL* vL*``, ``RPs[i]`` has legs ``vR wR* vR*``
    max_E_err : float
        Convergence criterion for the energy difference between two consecutive sweeps.
    energies : list of float
        Energies after every update.

    """

    def __init__(
        self, psi: SimpleMPS, model: TFIModel, chi_max: int = 100, max_E_err: float = 1.0e-12, eps: float = 1.0e-12
    ):
        assert psi.L == model.L  # ensure compatibility
        assert psi.bc == model.bc
        self.H_mpo = model.H_mpo
        self.psi = psi
        self.LPs = [None] * psi.L
        self.RPs = [None] * psi.L
        self.chi_max = chi_max
        self.eps = eps
        self.max_E_err = max_E_err
        self.n_sweeps = 0
        self.energies = []
        # initialize left and right environment
        self.LPs[0] = self.init_LP()
        self.RPs[-1] = self.init_RP()
        # initialize necessary RPs
        for i in range(psi.L - 1, 1, -1):
            self.update_RP(i)

    def init_LP(self):
        mps_left_leg = self.psi.Bs[0].codomain[0]
        mpo_left_leg = self.H_mpo[0].codomain[0]
        backend = self.psi.backend
        dtype = self.psi.Bs[0].dtype
        eye = lasy.SymmetricTensor.from_eye([mps_left_leg], backend=backend, labels=['vL', 'vL*'], dtype=dtype)
        # IdL is summand 0 of the finite MPO virtual leg
        unit = mpo_left_leg.unit_vector_of_summand(0, backend=backend, labels=['wL'], dtype=dtype)
        return lasy.outer(eye, lasy.dagger(unit))

    def init_RP(self):
        mps_right_leg = self.psi.Bs[-1].domain[0].dual
        mpo_right_leg = self.H_mpo[-1].domain[1].dual
        backend = self.psi.backend
        dtype = self.psi.Bs[-1].dtype
        eye = lasy.SymmetricTensor.from_eye([mps_right_leg], backend=backend, labels=['vR', 'vR*'], dtype=dtype)
        # IdR is the last summand of the finite MPO virtual leg
        unit = mpo_right_leg.unit_vector_of_summand(-1, backend=backend, labels=['wR'], dtype=dtype)
        return lasy.outer(lasy.dagger(unit), eye)

    def sweep(self):
        # sweep from left to right
        for i in range(self.psi.nbonds - 1):
            self.update_bond(i)
        # sweep from right to left
        for i in range(self.psi.nbonds - 1, 0, -1):
            self.update_bond(i)
        self.n_sweeps += 1

    def update_bond(self, i):
        j = i + 1
        # get effective Hamiltonian
        Heff = HEffective(self.LPs[i], self.RPs[j], self.H_mpo[i], self.H_mpo[j])
        # Diagonalize Heff, find ground state `theta`
        theta0 = self.psi.get_theta2(i)
        e, theta, _ = lasy.krylov_based.lanczos(Heff, theta0)
        self.energies.append(e)
        # split and truncate
        Ai, Sj, Bj = split_truncate_theta(theta, self.chi_max, self.eps)
        # put back into MPS
        Gi = lasy.scale_axis(Ai, lasy.pinv(self.psi.Ss[i], cutoff=self.eps), leg='vL')
        Bi = lasy.scale_axis(Gi, Sj, leg='vR')
        self.psi.Bs[i] = Bi
        self.psi.Ss[j] = Sj
        self.psi.Bs[j] = Bj
        self.update_LP(i)
        self.update_RP(j)

    def update_RP(self, i):
        """Calculate RP right of site `i-1` from RP right of site `i`."""
        j = i - 1
        RP = self.RPs[i]  # vR vR* wR*
        B = self.psi.Bs[i]  # vL p vR
        Bc = B.hc  # vR* p* vL*
        W = self.H_mpo[i]  # wL p wR p*

        Bc = lasy.permute_legs(Bc, ['p*', 'vL*'], ['vR*'], bend_right=[True, False, False])  # p* vL* vR*
        RP = lasy.compose(Bc, RP)  # p* vL* vR* wR*
        RP = lasy.permute_legs(
            RP, ['vL*', 'vR*'], ['p*', 'wR*'], bend_right=[False, None, True, None]
        )  # vL* vR* wR* p*
        W_ = lasy.permute_legs(W, ['p', 'wR'], ['wL', 'p*'], bend_right=[False, None, True, None])  # p wR p* wL
        RP = lasy.compose(RP, W_)  # vL* vR* p* wL
        RP = lasy.permute_legs(RP, ['wL', 'vL*'], ['p*', 'vR*'], bend_right=[None, True, None, False])  # wL vL* vR* p*
        B_ = lasy.permute_legs(B, ['p', 'vR'], ['vL'], bend_right=[False, None, True])  # p vR vL
        RP = lasy.compose(RP, B_, relabel1={'vL*': 'vR', 'wL': 'wR*'}, relabel2={'vL': 'vR*'})
        # wL vL* vL == wR* vR vR*
        RP = lasy.permute_legs(RP, ['vR'], ['wR*', 'vR*'], bend_right=[False, None, None])
        self.RPs[j] = RP

    def update_LP(self, i):
        """Calculate LP left of site `i+1` from LP left of site `i`."""
        j = i + 1
        LP = self.LPs[i]  # vL wL* vL*
        B = self.psi.Bs[i]  # vL p vR
        G = lasy.scale_axis(B, lasy.pinv(self.psi.Ss[j], cutoff=self.eps), leg='vR')
        A = lasy.scale_axis(G, self.psi.Ss[i], leg='vL')
        Ac = A.hc
        W = self.H_mpo[i]  # wL p wR p*

        Ac = lasy.permute_legs(Ac, codomain=['vR*', 'p*'], bend_right=True)
        LP = lasy.compose(Ac, LP)  # vR* p* wL* vL*
        LP = lasy.permute_legs(
            LP, ['vL*', 'vR*'], ['wL*', 'p*'], bend_right=[None, True, None, False]
        )  # vL* vR* p* wL*
        LP = lasy.compose(LP, W)  # vL* vR* wR p*
        LP = lasy.permute_legs(LP, ['vR*', 'wR'], ['vL*', 'p*'], bend_right=[False, None, True, None])  # vR* wR* p* vL*
        # vR* wR vR == vL wL* vL*
        LP = lasy.compose(LP, A, relabel1={'vR*': 'vL', 'wR': 'wL*'}, relabel2={'vR': 'vL*'})
        LP = lasy.permute_legs(LP, domain=['vL*', 'wL*'], bend_right=True)
        self.LPs[j] = LP

    def run(self) -> float:
        self.sweep()
        e_new = self.energies[-1]
        e_old = e_new + 2 * self.max_E_err
        while abs(e_new - e_old) > self.max_E_err:
            e_old = e_new
            self.sweep()
            e_new = self.energies[-1]
        return self.energies[-1]


class PlanarDMRGEngine(DMRGEngine):
    """Same as ``DMRGEngine``, but using ``PlanarDiagram``s."""

    update_RP_diagram = lasy.PlanarDiagram(
        tensors='Rp[vR*, wR*, vR], W[wL, p, wR, p*], B[vL, p, vR], B_hc[vR*, p*, vL*]',
        definition='Rp:vR @ B_hc:vR*, Rp:wR* @ W:wR, Rp:vR* @ B:vR, '
        'W:p* @ B:p, W:p @ B_hc:p*, W:wL -> wR*, '
        'B_hc:vL* -> vR, B:vL -> vR*',
        dims=dict(chi=['vR', 'vR*', 'vL', 'vL*'], w=['wL', 'wR', 'wR*'], d=['p', 'p*']),
    )
    update_LP_diagram = lasy.PlanarDiagram(
        tensors='Lp[vL, wL*, vL*], W[wL, p, wR, p*], A[vL, p, vR], A_hc[vR*, p*, vL*]',
        definition='Lp:vL @ A_hc:vL*, Lp:wL* @ W:wL, Lp:vL* @ A:vL, '
        'W:p* @ A:p, W:p @ A_hc:p*, W:wR -> wL*, '
        'A_hc:vR* -> vL, A:vR -> vL*',
        dims=dict(chi=['vR', 'vR*', 'vL', 'vL*'], w=['wL', 'wL*', 'wR'], d=['p', 'p*']),
    )

    def init_RP(self):
        # Need to do this in a way s.t. updating RP using the planar diagram yields the same leg
        # configuration. This is already fulfilled for LP with the default choice from DMRGEngine.
        return lasy.planar_permute_legs(super().init_RP(), domain=['vR', 'wR*', 'vR*'])

    def update_bond(self, i):
        j = i + 1
        # get effective Hamiltonian
        Heff = PlanarHEffective(Lp=self.LPs[i], Rp=self.RPs[j], W0=self.H_mpo[i], W1=self.H_mpo[j])
        # Diagonalize Heff, find ground state `theta`
        theta0 = self.psi.get_theta2(i)
        theta0 = lasy.planar_permute_legs(theta0, domain=['vR', 'p1', 'p0', 'vL'])
        e, theta, _ = lasy.krylov_based.lanczos(Heff, theta0)
        self.energies.append(e)
        # split and truncate
        theta = lasy.planar_permute_legs(theta, codomain=['vL', 'p0'], domain=['vR', 'p1'])
        Ai, Sj, Bj = split_truncate_theta(theta, self.chi_max, self.eps)
        # put back into MPS
        Gi = lasy.scale_axis(Ai, lasy.pinv(self.psi.Ss[i], cutoff=self.eps), leg='vL')
        Bi = lasy.scale_axis(Gi, Sj, leg='vR')
        self.psi.Bs[i] = Bi
        self.psi.Ss[j] = Sj
        self.psi.Bs[j] = Bj
        self.update_LP(i)
        self.update_RP(j)

    def update_RP(self, i):
        """Calculate RP right of site `i-1` from RP right of site `i`."""
        self.RPs[i - 1] = self.update_RP_diagram(
            Rp=self.RPs[i], W=self.H_mpo[i], B=self.psi.Bs[i], B_hc=self.psi.Bs[i].hc
        )

    def update_LP(self, i):
        """Calculate LP left of site `i+1` from LP left of site `i`."""
        B = self.psi.Bs[i]
        G = lasy.scale_axis(B, lasy.pinv(self.psi.Ss[i + 1], cutoff=self.eps), leg='vR')
        A = lasy.scale_axis(G, self.psi.Ss[i], leg='vL')
        self.LPs[i + 1] = self.update_LP_diagram(Lp=self.LPs[i], W=self.H_mpo[i], A=A, A_hc=A.hc)
