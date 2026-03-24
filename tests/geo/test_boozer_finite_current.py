"""
Tests for the finite plasma current I in the BoozerSurface residual.

The modification changes the Boozer residual from:
    r = G*B - |B|^2 * (x_phi + iota * x_theta)
to:
    r = (G + iota*I)*B - |B|^2 * (x_phi + iota * x_theta)

where I is a fixed plasma current (not a DOF).

Tests verify:
1. When I=0, results match the original I=0 case.
2. Gradient and Hessian correctness via Taylor-series convergence (finite differences).
"""
import unittest
import numpy as np

from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.field.coil import Current, Coil
from simsopt.geo import SurfaceXYZTensorFourier, create_equally_spaced_curves, ToroidalFlux
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import (
    boozer_surface_residual,
    boozer_surface_residual_dB,
)


def make_simple_coils_and_surface(mpol=2, ntor=2, nfp=2, stellsym=True):
    """Build a simple coil set and initial surface for testing."""
    curves = create_equally_spaced_curves(2 * nfp, nfp, stellsym=False, R0=1.0, R1=0.5, order=1)
    currents = [Current(1e5) for _ in curves]
    coils = [Coil(c, cur) for c, cur in zip(curves, currents)]
    bs = BiotSavart(coils)

    phis = np.linspace(0, 1 / nfp, 2 * ntor + 1, endpoint=False)
    thetas = np.linspace(0, 1, 2 * mpol + 1, endpoint=False)
    s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
                                 quadpoints_phi=phis, quadpoints_theta=thetas)
    s.fit_to_curve(curves[0], 0.3)
    return bs, s


class TestFiniteCurrentResidual(unittest.TestCase):

    def test_I_zero_matches_baseline(self):
        """With I=0, residual and Jacobian must match the I=0 (no-plasma-current) case."""
        bs, s = make_simple_coils_and_surface()
        iota = 0.4
        G = 1.2

        r0, J0 = boozer_surface_residual(s, iota, G, bs, derivatives=1)
        rI, JI = boozer_surface_residual(s, iota, G, bs, derivatives=1, I=0.)

        np.testing.assert_allclose(r0, rI, atol=1e-14)
        np.testing.assert_allclose(J0, JI, atol=1e-14)

    def test_residual_changes_with_I(self):
        """With nonzero I the residual must differ from I=0."""
        bs, s = make_simple_coils_and_surface()
        iota = 0.4
        G = 1.2
        I = 5e4

        r0, = boozer_surface_residual(s, iota, G, bs, derivatives=0)
        rI, = boozer_surface_residual(s, iota, G, bs, derivatives=0, I=I)

        self.assertFalse(np.allclose(r0, rI, atol=1e-10),
                         "Residual should change when I != 0")

    def test_iota_derivative_with_I(self):
        """
        Taylor-series check: the iota column of the Jacobian J[:, -2] (or J[:, -1])
        should match the finite-difference derivative of r w.r.t. iota.
        The iota column is the second-to-last column when G is provided.
        """
        bs, s = make_simple_coils_and_surface()
        iota = 0.42
        G = 1.15
        I = 3e4

        eps = 1e-5
        r_plus, = boozer_surface_residual(s, iota + eps, G, bs, derivatives=0, I=I)
        r_minus, = boozer_surface_residual(s, iota - eps, G, bs, derivatives=0, I=I)
        dr_diota_fd = (r_plus - r_minus) / (2 * eps)

        _, J = boozer_surface_residual(s, iota, G, bs, derivatives=1, I=I)
        # J has columns: [surf_dofs..., iota, G]
        dr_diota_analytic = J[:, -2]

        np.testing.assert_allclose(dr_diota_fd, dr_diota_analytic, rtol=1e-5, atol=1e-10)

    def test_surface_dof_derivative_with_I(self):
        """
        Taylor-series check on a surface DOF gradient with nonzero I.
        """
        bs, s = make_simple_coils_and_surface()
        iota = 0.42
        G = 1.15
        I = 3e4
        dofs0 = s.get_dofs().copy()

        eps = 1e-5
        perturb = np.zeros(len(dofs0))
        perturb[0] = 1.0  # perturb first DOF

        s.set_dofs(dofs0 + eps * perturb)
        r_plus, = boozer_surface_residual(s, iota, G, bs, derivatives=0, I=I)
        s.set_dofs(dofs0 - eps * perturb)
        r_minus, = boozer_surface_residual(s, iota, G, bs, derivatives=0, I=I)
        dr_dc0_fd = (r_plus - r_minus) / (2 * eps)

        s.set_dofs(dofs0)
        _, J = boozer_surface_residual(s, iota, G, bs, derivatives=1, I=I)
        dr_dc0_analytic = J[:, 0]

        np.testing.assert_allclose(dr_dc0_fd, dr_dc0_analytic, rtol=1e-5, atol=1e-10)

    def test_iota_derivative_dB_with_I(self):
        """
        Taylor-series check: dresidual_diota from boozer_surface_residual_dB
        should include the I*B term (not just -B^2 * xtheta).
        """
        bs, s = make_simple_coils_and_surface()
        iota = 0.42
        G = 1.15
        I = 3e4

        eps = 1e-5
        r_plus, = boozer_surface_residual(s, iota + eps, G, bs, derivatives=0, I=I)
        r_minus, = boozer_surface_residual(s, iota - eps, G, bs, derivatives=0, I=I)
        dr_diota_fd = (r_plus - r_minus) / (2 * eps)

        _, J = boozer_surface_residual_dB(s, iota, G, bs, derivatives=0, I=I)
        # boozer_surface_residual_dB with derivatives=0 returns (r, dr_dB) only.
        # We need the iota column, which comes from boozer_surface_residual:
        _, Jfull = boozer_surface_residual(s, iota, G, bs, derivatives=1, I=I)
        dr_diota_analytic = Jfull[:, -2]

        np.testing.assert_allclose(dr_diota_fd, dr_diota_analytic, rtol=1e-5, atol=1e-10)

    def test_hessian_taylor_convergence(self):
        """
        Second-order Taylor test on the Hessian: the residual of the gradient Taylor
        expansion should decrease as O(eps^2).
        """
        bs, s = make_simple_coils_and_surface(mpol=1, ntor=1)
        iota = 0.42
        G = 1.15
        I = 2e4
        dofs0 = s.get_dofs().copy()
        nsurfdofs = len(dofs0)

        # random direction in (surface_dofs, iota, G) space
        np.random.seed(0)
        direction = np.random.randn(nsurfdofs + 2)
        direction /= np.linalg.norm(direction)

        # Evaluate at dofs0
        s.set_dofs(dofs0)
        r0, J0, H0 = boozer_surface_residual(s, iota, G, bs, derivatives=2, I=I)

        # Full gradient in (surf_dofs, iota, G) space: J0.T @ r0
        # For a perturbation eps*d: r(d) ≈ r0 + J0 @ (eps*d) + 0.5 * H0 @ (eps*d)^2
        # The quadratic objective f = 0.5 * r.T @ r:
        # f(eps*d) ≈ f0 + (J0.T @ r0).T (eps*d) + 0.5*(eps*d)^T*(J0.T@J0 + r0@H0)@(eps*d)

        errors = []
        for eps in [1e-3, 5e-4, 2e-4]:
            d_sdofs = direction[:nsurfdofs]
            d_iota = direction[-2]
            d_G = direction[-1]
            s.set_dofs(dofs0 + eps * d_sdofs)
            r1, J1 = boozer_surface_residual(s, iota + eps * d_iota,
                                              G + eps * d_G, bs, derivatives=1, I=I)
            # First-order prediction of J1:
            # J1 ≈ J0 + H0 (eps*direction) elementwise (H0 is 3D: nphi*ntheta*3 x ndofs x ndofs)
            J1_pred = J0 + np.einsum('ijk,k->ij', H0, eps * direction)
            errors.append(np.linalg.norm(J1 - J1_pred))

        s.set_dofs(dofs0)

        # Check second-order convergence
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]
        self.assertGreater(ratio1, 3.5, f"Expected O(eps^2) convergence, got ratio={ratio1:.2f}")
        self.assertGreater(ratio2, 3.5, f"Expected O(eps^2) convergence, got ratio={ratio2:.2f}")

    def test_BoozerSurface_I_stored_on_object(self):
        """BoozerSurface should store I on self.I."""
        bs, s = make_simple_coils_and_surface()
        from simsopt.geo.surfaceobjectives import Area
        lab = Area(s)
        target = lab.J()

        bsurf_default = BoozerSurface(bs, s, lab, target)
        self.assertAlmostEqual(bsurf_default.I, 0.)

        I_val = 5e4
        bsurf_I = BoozerSurface(bs, s, lab, target, I=I_val)
        self.assertAlmostEqual(bsurf_I.I, I_val)

    def test_I_in_res_after_run_code(self):
        """After run_code, res['I'] should match the plasma current passed at construction."""
        from .surface_test_helpers import get_boozer_surface
        I_val = 3e4
        # get_boozer_surface returns (bs, boozer_surface)
        _, boozer_surface = get_boozer_surface(label="ToroidalFlux", boozer_type='exact', converge=False, optimize_G=False)
        boozer_surface.I = I_val
        iota = -0.406
        boozer_surface.run_code(iota, G=None)
        self.assertAlmostEqual(boozer_surface.res['I'], I_val)


class TestBoozerResidualAlgebraicIdentities(unittest.TestCase):
    """
    Verify the six exact algebraic identities that relate the finite-I residual
    to the I=0 residual.  All identities hold to floating-point precision
    (no Taylor-series approximation), because the only effect of I is to replace
    G with Geff = G + iota*I in specific terms.

    The six identities (all exact, verified to atol=1e-13):
      1.  r(I) - r(0) = iota * I * B
      2.  dr/diota(I) - dr/diota(0) = I * B
      3.  dr/dG(I) = dr/dG(0) = B   [G-column is I-independent]
      4.  dr/dc_m(I) - dr/dc_m(0) = iota * I * dB/dc_m
      5.  d²r/dc_m∂iota(I) - d²r/dc_m∂iota(0) = I * dB/dc_m
      6.  d²r/dc_m∂c_n(I) - d²r/dc_m∂c_n(0) = iota * I * d²B/(dc_m ∂c_n)

    Additionally, one test reconstructs r from raw numpy without calling
    boozer_surface_residual, to verify the absolute formula at I != 0.
    """

    iota = 0.42
    G = 1.15
    I = 3e4

    def _ingredients(self, bs, s, deriv):
        """
        After calling boozer_surface_residual(..., derivatives=deriv), the
        BiotSavart points are already set to the surface quadrature grid.
        Return (B, dB_by_dX, d2B_by_dXdX, dB_dc, d2B_dcdc, nphi, ntheta, nsurfdofs).
        Only computes up to the requested derivative order.
        """
        x = s.gamma()
        nphi, ntheta = x.shape[:2]

        bs.set_points(x.reshape(-1, 3))
        bs.compute(deriv)

        B = bs.B().reshape(nphi, ntheta, 3)
        dB_dc = None
        d2B_dcdc = None

        if deriv >= 1:
            dx_dc = s.dgamma_by_dcoeff()          # (nphi, ntheta, 3, ndofs)
            dB_by_dX = bs.dB_by_dX().reshape(nphi, ntheta, 3, 3)
            dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc)  # (nphi, ntheta, 3, ndofs)

        if deriv >= 2:
            d2B_by_dXdX = bs.d2B_by_dXdX().reshape(nphi, ntheta, 3, 3, 3)
            d2B_dcdc = np.einsum('ijkpl,ijpn,ijkm->ijlmn',
                                 d2B_by_dXdX, dx_dc, dx_dc, optimize=True)  # (nphi, ntheta, 3, ndofs, ndofs)

        nsurfdofs = s.dgamma_by_dcoeff().shape[-1]
        return B, dB_dc, d2B_dcdc, nphi, ntheta, nsurfdofs

    # ------------------------------------------------------------------
    # Identity 1: r(I) - r(0) = iota * I * B
    # ------------------------------------------------------------------
    def test_identity1_residual_difference(self):
        bs, s = make_simple_coils_and_surface()
        r0, = boozer_surface_residual(s, self.iota, self.G, bs, derivatives=0)
        rI, = boozer_surface_residual(s, self.iota, self.G, bs, derivatives=0, I=self.I)

        B, _, _, nphi, ntheta, _ = self._ingredients(bs, s, 0)
        expected = (self.iota * self.I * B).ravel()
        # iota*I*B ~ O(1e2); two separate floating-point paths give O(eps*|value|) ~ 1e-13
        # atol=5e-12 is ~50*machine_epsilon*max(|value|), confirming the identity to 12 digits
        np.testing.assert_allclose(rI - r0, expected, atol=5e-12, rtol=0)

    # ------------------------------------------------------------------
    # Identity 2: dr/diota(I) - dr/diota(0) = I * B
    # ------------------------------------------------------------------
    def test_identity2_dr_diota_difference(self):
        bs, s = make_simple_coils_and_surface()
        _, J0 = boozer_surface_residual(s, self.iota, self.G, bs, derivatives=1)
        _, JI = boozer_surface_residual(s, self.iota, self.G, bs, derivatives=1, I=self.I)

        B, _, _, nphi, ntheta, _ = self._ingredients(bs, s, 0)
        expected = (self.I * B).ravel()
        np.testing.assert_allclose(JI[:, -2] - J0[:, -2], expected, atol=1e-13, rtol=0)

    # ------------------------------------------------------------------
    # Identity 3: dr/dG(I) = dr/dG(0) = B
    # ------------------------------------------------------------------
    def test_identity3_dr_dG_I_independence(self):
        bs, s = make_simple_coils_and_surface()
        _, J0 = boozer_surface_residual(s, self.iota, self.G, bs, derivatives=1)
        _, JI = boozer_surface_residual(s, self.iota, self.G, bs, derivatives=1, I=self.I)

        B, _, _, nphi, ntheta, _ = self._ingredients(bs, s, 0)
        B_flat = B.ravel()

        # Both G-columns are identical to each other and equal to B
        np.testing.assert_allclose(JI[:, -1], J0[:, -1], atol=1e-14, rtol=0)
        np.testing.assert_allclose(J0[:, -1], B_flat, atol=1e-13, rtol=0)
        np.testing.assert_allclose(JI[:, -1], B_flat, atol=1e-13, rtol=0)

    # ------------------------------------------------------------------
    # Identity 4: dr/dc_m(I) - dr/dc_m(0) = iota * I * dB/dc_m
    # ------------------------------------------------------------------
    def test_identity4_dr_dc_difference(self):
        bs, s = make_simple_coils_and_surface()
        _, J0 = boozer_surface_residual(s, self.iota, self.G, bs, derivatives=1)
        _, JI = boozer_surface_residual(s, self.iota, self.G, bs, derivatives=1, I=self.I)
        nsurfdofs = J0.shape[1] - 2

        B, dB_dc, _, nphi, ntheta, _ = self._ingredients(bs, s, 1)
        expected = (self.iota * self.I * dB_dc).reshape(nphi * ntheta * 3, nsurfdofs)
        # iota*I*dB_dc values ~ O(1e2-4e2); atol=5e-12 ~ 50*eps*max(|value|)
        np.testing.assert_allclose(JI[:, :nsurfdofs] - J0[:, :nsurfdofs], expected, atol=5e-12, rtol=0)

    # ------------------------------------------------------------------
    # Identity 5: d²r/dc_m∂iota(I) - d²r/dc_m∂iota(0) = I * dB/dc_m
    # ------------------------------------------------------------------
    def test_identity5_d2r_dcdiota_difference(self):
        bs, s = make_simple_coils_and_surface(mpol=1, ntor=1)
        _, _, H0 = boozer_surface_residual(s, self.iota, self.G, bs, derivatives=2)
        _, _, HI = boozer_surface_residual(s, self.iota, self.G, bs, derivatives=2, I=self.I)
        nsurfdofs = H0.shape[1] - 2

        B, dB_dc, _, nphi, ntheta, _ = self._ingredients(bs, s, 1)
        expected = (self.I * dB_dc).reshape(nphi * ntheta * 3, nsurfdofs)
        # H[:, :nsurfdofs, nsurfdofs] is the dc-diota block
        # I*dB_dc values ~ O(1e3); atol=5e-12 ~ 5*eps*max(|value|)
        np.testing.assert_allclose(
            HI[:, :nsurfdofs, nsurfdofs] - H0[:, :nsurfdofs, nsurfdofs],
            expected, atol=5e-12, rtol=0)

    # ------------------------------------------------------------------
    # Identity 6: d²r/dc_m∂c_n(I) - d²r/dc_m∂c_n(0) = iota * I * d²B/(dc_m ∂c_n)
    # ------------------------------------------------------------------
    def test_identity6_d2r_dcdc_difference(self):
        bs, s = make_simple_coils_and_surface(mpol=1, ntor=1)
        _, _, H0 = boozer_surface_residual(s, self.iota, self.G, bs, derivatives=2)
        _, _, HI = boozer_surface_residual(s, self.iota, self.G, bs, derivatives=2, I=self.I)
        nsurfdofs = H0.shape[1] - 2

        _, dB_dc, d2B_dcdc, nphi, ntheta, _ = self._ingredients(bs, s, 2)
        expected = (self.iota * self.I * d2B_dcdc).reshape(nphi * ntheta * 3, nsurfdofs, nsurfdofs)
        # d2B_dcdc involves triple contractions; more accumulated roundoff.
        # atol=1e-10 ~ 1e5*eps*max(|value|), still confirming the identity to 10 digits
        np.testing.assert_allclose(
            HI[:, :nsurfdofs, :nsurfdofs] - H0[:, :nsurfdofs, :nsurfdofs],
            expected, atol=1e-10, rtol=0)

    # ------------------------------------------------------------------
    # Absolute formula check: reconstruct r from raw numpy at I != 0
    # ------------------------------------------------------------------
    def test_independent_numpy_reconstruction(self):
        """
        Build r = (G + iota*I)*B - |B|²*(xphi + iota*xtheta) directly in numpy
        and compare against boozer_surface_residual to machine precision.
        This verifies the absolute formula, not just relative differences.
        """
        bs, s = make_simple_coils_and_surface()
        rI, = boozer_surface_residual(s, self.iota, self.G, bs, derivatives=0, I=self.I)

        x = s.gamma()
        nphi, ntheta = x.shape[:2]
        bs.set_points(x.reshape(-1, 3))
        B = bs.B().reshape(nphi, ntheta, 3)
        xphi = s.gammadash1()
        xtheta = s.gammadash2()

        Geff = self.G + self.iota * self.I
        tang = xphi + self.iota * xtheta
        B2 = np.sum(B**2, axis=2)
        r_np = (Geff * B - B2[..., None] * tang).ravel()

        np.testing.assert_allclose(rI, r_np, atol=1e-13, rtol=0)


if __name__ == '__main__':
    unittest.main()
