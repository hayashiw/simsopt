import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

try:
    from simsopt.util import proc0_print as _proc0_print
except ImportError:
    _proc0_print = print


def _print(*args, **kwargs):
    """MPI-aware print with flush=True default."""
    kwargs.setdefault('flush', True)
    _proc0_print(*args, **kwargs)

__all__ = ['augmented_lagrangian_objective', 
           'grad_augmented_lagrangian', 'augmented_lagrangian_method',
]

class dummyObjective:
    def __init__(self, x):
        self.x = x
    def J(self):
        return 0.0
    def dJ(self):
        return np.zeros_like(self.x)

def jac_constraint(constraint_list, f):
    """
    Compute the Jacobian matrix of a list of constraint functions with respect to f's DOF space.

    Args:
        constraint_list (list): List of constraint objects, each with .x and .dJ() methods.
        f: The objective Optimizable (or dummyObjective) whose .x defines the full DOF vector.

    Returns:
        np.ndarray: Jacobian matrix of shape (len(constraint_list), len(f.x)), where each row is the gradient of a constraint.
    """
    dofs = f.x
    J = np.zeros((len(constraint_list), len(dofs)), dtype=float)
    for i, c_i in enumerate(constraint_list):
        grad_ci = np.array([], dtype=np.float64)
        try:
            with threadpool_limits(limits=1, user_api='openmp'):
                grad_ci = c_i.dJ()
        except Exception as e:
            _print(f"Exception during c_i.dJ() in jac_constraint: {e}")
            grad_ci = c_i.dJ()
        grad_ci_arr = np.atleast_1d(np.asarray(grad_ci))
        if isinstance(f, dummyObjective):
            # Original behaviour: constraint DOFs assumed to be a suffix of f.x
            dof_dif = len(dofs) - len(c_i.x)
            J[i, dof_dif:] = grad_ci_arr
        else:
            # Use dof_indices to place each constraint's gradient at the correct
            # positions in f's full DOF space.
            for opt, (f_start, f_end) in f.dof_indices.items():
                if opt in c_i.dof_indices:
                    c_start, c_end = c_i.dof_indices[opt]
                    J[i, f_start:f_end] = grad_ci_arr[c_start:c_end]
    return J


def augmented_lagrangian_objective(dofs, f, equality_constraints, lag_mul, mu):
    """
    Compute the value of the augmented Lagrangian for the current optimization variables.

    The standard augmented Lagrangian is:

        .. math::

            L_A(x, \lambda, \mu) = f(x) - \lambda^T g(x) + \frac{\mu}{2} \|g(x)\|^2

    where:
        - :math:`f(x)`: objective function (e.g., squared flux)
        - :math:`g(x)`: vector of constraint functions
        - :math:`\lambda`: vector of Lagrange multipliers
        - :math:`\mu`: penalty parameter

    If ``option == 'least-squares'``, uses the least-squares form:

        .. math::

            L_{LS}(x, \lambda, \mu) = \frac{1}{2} \|f(x)\|^2 + \frac{1}{2} \left\| -\frac{\lambda}{\sqrt{\mu}} + \sqrt{\mu} g(x) \right\|^2

    Args:
        f: Objective function object (with .J() and .x attributes).
        dofs (np.ndarray): Current degrees of freedom.
        equality_constraints (list): List of constraint objects (with .J() and .dJ() methods).
        lag_mul (np.ndarray): Lagrange multipliers.
        mu (float): Penalty parameter.
        option (str, optional): If 'least-squares', use least-squares form.

    Returns:
        float: Value of the augmented Lagrangian.
    """
    f.x = dofs
    if isinstance(f, dummyObjective):
        f_val = f.J()
    else:
        try:
            with threadpool_limits(limits=1, user_api='openmp'):
                f_val = f.J()
        except Exception:
            f_val = f.J()

    # collect constraint values
    # When f is a real Optimizable, f.x = dofs already propagated the full DAG;
    # re-setting c_i.x with a tail-slice would overwrite with wrong values.
    c_vals_list = []
    for c_i in equality_constraints:
        if isinstance(f, dummyObjective):
            dof_dif = len(dofs) - len(c_i.x)
            c_i.x = dofs[dof_dif:]
        try:
            with threadpool_limits(limits=1, user_api='openmp'):
                val_ci = float(c_i.J())
        except Exception:
            val_ci = float(c_i.J())
        c_vals_list.append(val_ci)
    c_vals = np.array(c_vals_list, dtype=np.float64)

    # build Lagrangian with mu now a vector
    #         # 0.5 * sum_i mu_i * c_i^2
    L = f_val - np.dot(lag_mul, c_vals) + 0.5 * np.dot(mu, c_vals**2)

    return L


def grad_augmented_lagrangian(dofs, f, equality_constraints, lag_mul, mu):
    """
    Compute the gradient of the augmented Lagrangian with respect to the optimization variables.

    For the standard form:

        .. math::

            \nabla_x L_A(x, \lambda, \mu) = \nabla f(x) - \sum_i \lambda_i \nabla g_i(x) + \mu J_g^T g(x)

    where:
        - :math:`J_g`: Jacobian matrix of constraints (rows: constraints, columns: dofs)
        - :math:`g(x)`: vector of constraint values

    For the least-squares form (if ``option == 'least-squares'``):

        .. math::

            \nabla_x L_{LS}(x, \lambda, \mu) = f(x) \nabla f(x) + \sqrt{\mu} J_g^T g(x)

    Args:
        dofs (np.ndarray): Current degrees of freedom.
        f: Objective function object (with .J() and .dJ() methods).
        equality_constraints (list): List of constraint objects.
        lag_mul (np.ndarray): Lagrange multipliers.
        mu (float): Penalty parameter.
        option (str, optional): If 'least-squares', use least-squares form.

    Returns:
        np.ndarray: Gradient of the augmented Lagrangian.
    """
    f.x = dofs
    if isinstance(f, dummyObjective):
        grad_f = f.dJ()
    else:
        try:
            with threadpool_limits(limits=1, user_api='openmp'):
                grad_f = f.dJ()
        except Exception:
            grad_f = f.dJ()
    grad_f_arr = np.asarray(grad_f, dtype=np.float64)

    # When f is a real Optimizable, f.x = dofs already propagated the full DAG.
    # Only re-set constraint DOFs in the dummyObjective case (original behaviour).
    if isinstance(f, dummyObjective):
        for c_i in equality_constraints:
            dof_dif = len(dofs) - len(c_i.x)
            c_i.x = dofs[dof_dif:]

    # get Jacobian matrix of all constraints
    c_jac = jac_constraint(equality_constraints, f)

    # collect constraint values
    c_vals_list = []
    for c_i in equality_constraints:
        try:
            with threadpool_limits(limits=1, user_api='openmp'):
                val_ci = float(c_i.J())
        except Exception:
            val_ci = float(c_i.J())
        c_vals_list.append(val_ci)
    c_vals = np.array(c_vals_list, dtype=np.float64)

    # build gradient of Lagrangian with vector mu
    # grad_f - lag_mul^T * jac + jac^T * (mu * c_vals)
    dL = grad_f_arr
    dL -= np.dot(lag_mul, c_jac)
    dL += np.dot(c_jac.T, mu * c_vals)

    return np.asarray(dL, dtype=np.float64)


def _collect_leaves(node):
    """Recursively flatten a nested OptimizableSum tree into its leaf constraints.

    Python's built-in sum() builds a left-associative binary tree of OptimizableSum
    nodes, so a sum of N terms produces a tree of depth N-1 rather than a flat list.
    This helper traverses the whole tree and returns only the non-OptimizableSum leaves.
    """
    if type(node).__name__ == 'OptimizableSum':
        result = []
        for child in node.opts:
            result.extend(_collect_leaves(child))
        return result
    return [node]


def _constraint_name(c):
    """Return a descriptive name for a constraint, unwrapping one level of wrapper objectives.

    For direct objectives (e.g. BoozerResidual, CurveCurveDistance) returns the class name.
    For wrapper objectives that carry an inner objective in a common attribute (e.g.
    QuadraticPenalty with .obj), returns 'WrapperClass(InnerClass)'.
    For OptimizableSum, flattens the entire (potentially deeply-nested binary) tree and
    lists unique leaf types with counts, e.g. 'Sum(8×QuadraticPenalty(CurveLength))'.
    """
    cls = type(c).__name__
    if cls == 'OptimizableSum':
        leaves = _collect_leaves(c)
        leaf_names = [_constraint_name(leaf) for leaf in leaves]
        from collections import Counter
        counts = Counter(leaf_names)
        parts = [f"{n}×{name}" if n > 1 else name for name, n in counts.items()]
        return f"Sum({', '.join(parts)})"
    for attr in ('obj', 'Jobj'):
        inner = getattr(c, attr, None)
        if inner is not None:
            return f"{cls}({type(inner).__name__})"
    return cls


def augmented_lagrangian_method(
        f=None,
        equality_constraints=[],
        mu_init=10.0,
        grad_tol=1e-15,
        c_tol=1e-15,
        tau=4,
        MAXITER=50,
        MAXFUN=None,
        argmin_tol=1e-15,
        minimize_method='L-BFGS-B',
        MAXITER_LAG=10,
        verbose=False,
        penalty_type=None,
        dof_scale=None,
        callback=None,
        ):
    """
    Run the Augmented Lagrangian Method (ALM) for constrained optimization.

    This method solves:

        .. math::

            \min_x f(x) \quad \text{subject to} \quad g_i(x) = 0

    by iteratively minimizing the augmented Lagrangian and updating multipliers and penalty parameters.

    The main loop alternates between minimizing the augmented Lagrangian and updating the multipliers/penalty:
        - Minimize :math:`L_A(x, \lambda, \mu)` with respect to :math:`x`
        - Update :math:`\lambda` and :math:`\mu` based on constraint satisfaction

    Args:
        f (Optimizable, optional): Main objective function (with .J(), .dJ(), .x attributes).
            If not provided, only constraints are used in the optimization.
        equality_constraints (list of Optimizable): List of constraint objects corresponding to equality constraints.
            These are equivalent to inequality constraints if the constraint is set up so that if the value is below
            some threshold, the value is set to zero.
        mu_init (float, optional, default=10.0): Initial penalty parameter (must be > 0).
        grad_tol (float, optional, default=1e-15): Tolerance for gradient norm of Lagrangian.
        c_tol (float, optional, default=1e-15): Tolerance for constraint norm.
        MAXITER (int, optional, default=50): Max iterations for inner minimization.
        MAXFUN (int, optional, default=None): Max function evaluations for the inner
            minimization (L-BFGS-B only). If None, no explicit cap is applied beyond the
            minimizer's own defaults. Reduces the number of objective evaluations per outer
            ALM iteration.
        argmin_tol (float, optional, default=1e-15): Tolerance for inner minimization.
        minimize_method (str, optional, default='L-BFGS-B'): Optimization method for inner loop.
        MAXITER_LAG (int, optional, default=10): Max outer ALM iterations.
        lagrangian_form (str, optional, default=None): If 'least-squares', use least-squares form.
        verbose (bool, optional, default=False): Whether to print verbose output.
        dof_scale (float or array-like, optional, default=None): If set, the inner minimizer
            works in a rescaled DOF space ``y = x / scale``. A step of size ``s`` in ``y``
            corresponds to a physical step of ``s * scale`` in ``x``. The gradient returned
            to the optimizer is rescaled accordingly: ``dL/dy = dL/dx * scale``.

            In single-stage Boozer surface optimization the primary motivation is NOT to
            compensate for different DOF types (surface shape DOFs are NOT outer variables —
            they are solved implicitly by ``bsurf.run_code()`` via the adjoint method).
            Instead, a uniform ``dof_scale < 1`` limits the physical step size of each
            L-BFGS-B trial point so that coil perturbations stay small enough for the
            Boozer Newton solver to remain in its convergence basin.

            A scalar applies the same scale to all DOFs; an array of the same length as ``x``
            applies per-DOF scales. ``None`` (default) applies no rescaling.
        callback (callable, optional): If provided, a function to call after each outer iteration with signature `callback(x, f_val, c_vals)`.

    Returns:
        tuple: (x, final_Lagrangian_value, lagrange_multipliers, penalty_parameters)
            - x (np.ndarray): Optimized degrees of freedom
            - final_Lagrangian_value (float): Final value of the augmented Lagrangian
            - lagrange_multipliers (np.ndarray): Final Lagrange multipliers (λ)
            - penalty_parameters (np.ndarray): Final per-constraint penalty parameters (μ).
              The effective weight on each constraint's gradient is (μᵢcᵢ - λᵢ),
              analogous to wᵢ in a traditional weighted objective J = f + Σ wᵢgᵢ.
    """
    np.random.seed(1)
    m_eq = len(equality_constraints)

    if mu_init <= 0 or grad_tol <= 0 or c_tol <= 0:
        raise ValueError(
            "eta_init, mu_init and omega_init  must be strictly positive")
    if not penalty_type:
        if np.isscalar(mu_init):
            mu_k = np.ones(m_eq, dtype=float) * mu_init
        else:
            mu_k = np.array(mu_init, dtype=float)
            if mu_k.size != m_eq:
                raise ValueError("mu_init vector length must match number of constraints")
        if np.any(mu_k <= 1):
            raise ValueError("All components of mu_init must be > 1")
    elif penalty_type == 'scalar':
        mu_k = mu_init
    else:
        raise ValueError("Invalid penalty type")
    
    k = 1

    # Use the DOF vector from f if available, otherwise fall back to the first constraint
    if f is not None:
        x = f.x
    else:
        x = equality_constraints[0].x
    
    if f is None:
        f = dummyObjective(x)

    if verbose:
        _print('----------------------------------------------------------------')
        _print(f'METHOD {minimize_method} IS SELECTED FOR THE OPTIMIZATION')
        _print(f'INITIAL OBJECTIVE: {f.J():0.6f}')
        _print('----------------------------------------------------------------')

    # Initialize multipliers randomly -- DO NOT INITIALIZE TO ZEROS since then the
    # first iteration will not be able to improve the constraints and DO NOT
    # INITIALIZE TO -mu_init * c_vals since then some of the lagrange multipliers
    # will be set to exactly zero, turning off the corresponding constraint for 
    # all remaining iterations. However, need to make sure lag_mul matches c_vals sign
    c_vals = np.array([J.J() for J in equality_constraints], dtype=float)
    lag_mul = -np.random.rand(m_eq) * np.sign(c_vals)

    # Evaluate initial lagrangian
    c_norm = np.linalg.norm(c_vals)

    mu_k_scalar = np.mean(mu_k)
    omega_k = 1.0 / mu_k_scalar
    eta_k = 1.0 / (mu_k_scalar ** 0.1)
    aug_lag = augmented_lagrangian_objective(x, f, equality_constraints, lag_mul, mu_k)
    grad_aug_lag_norm = np.linalg.norm(grad_augmented_lagrangian(
        x, f, equality_constraints, lag_mul, mu_k))

    if verbose:
        _print("--------------------------------------------------------------------------------------------------------------------------------------------")
        # Handle mu_k formatting - it can be a scalar or array
        if np.isscalar(mu_k):
            mu_k_str = f"{mu_k:.2e}"
        else:
            mu_k_str = f"[{', '.join([f'{m:.2e}' for m in mu_k])}]"
        _print(f"Iteration {0}, \u03BC_k={mu_k_str}, \u03C9_k={omega_k:.2e}, \u03B7_k={eta_k:.2e}, \u221A║∇L_A║ = {grad_aug_lag_norm:.2e}, \u221A║g║ = {c_norm:.2e}")
        _print(f'L_A value: {aug_lag:0.5f}')
        _print("--------------------------------------------------------------------------------------------------------------------------------------------")

    options = {
        'disp': False,
        'maxiter': MAXITER,
    }
    if minimize_method == 'L-BFGS-B':
        options['maxcor'] = 100
        if MAXFUN is not None:
            options['maxfun'] = MAXFUN

    # Set up DOF scaling: the inner minimizer works in y = x / scale.
    # A unit step in y corresponds to a scale-sized step in physical x.
    # Setting scale < 1 for a DOF type slows down its effective step size,
    # which is useful when some DOF types have much larger gradient magnitudes.
    if dof_scale is not None:
        scale = np.broadcast_to(np.asarray(dof_scale, dtype=float), x.shape).copy()
    else:
        scale = np.ones(len(x), dtype=float)

    # If the penalty parameter is too large, stop the optimization
    while (grad_aug_lag_norm > grad_tol or c_norm > c_tol) and k < MAXITER_LAG:
        # Solve arg min of the augmented lagrangian
        dofs_before = x.copy()

        def fun(dofs):
            # Set DOFs exactly ONCE so the Optimizable DAG is only invalidated
            # once per function evaluation.  Calling f.x = dofs a second time
            # (as the old two-function approach did) unconditionally re-triggers
            # recompute_bell() → need_to_run_code = True even when dofs haven't
            # changed, causing a redundant (iter=0) Boozer solve per call.
            f.x = dofs

            # ── objective value ───────────────────────────────────────────────
            if isinstance(f, dummyObjective):
                f_val = f.J()
            else:
                try:
                    with threadpool_limits(limits=1, user_api='openmp'):
                        f_val = f.J()
                except Exception:
                    f_val = f.J()

            # ── constraint values (DAG already propagated above) ──────────────
            c_vals_list = []
            for c_i in equality_constraints:
                if isinstance(f, dummyObjective):
                    dof_dif = len(dofs) - len(c_i.x)
                    c_i.x = dofs[dof_dif:]
                try:
                    with threadpool_limits(limits=1, user_api='openmp'):
                        val_ci = float(c_i.J())
                except Exception:
                    val_ci = float(c_i.J())
                c_vals_list.append(val_ci)
            c_vals = np.array(c_vals_list, dtype=np.float64)

            # ── objective gradient (cached; no second run_code triggered) ─────
            if isinstance(f, dummyObjective):
                grad_f = f.dJ()
            else:
                try:
                    with threadpool_limits(limits=1, user_api='openmp'):
                        grad_f = f.dJ()
                except Exception:
                    grad_f = f.dJ()
            grad_f_arr = np.asarray(grad_f, dtype=np.float64)

            # ── constraint Jacobian ───────────────────────────────────────────
            c_jac = jac_constraint(equality_constraints, f)

            # ── augmented Lagrangian value and gradient ───────────────────────
            aug_lag = f_val - np.dot(lag_mul, c_vals) + 0.5 * np.dot(mu_k, c_vals**2)
            grad_aug_lag = grad_f_arr - np.dot(lag_mul, c_jac) + np.dot(c_jac.T, mu_k * c_vals)

            return aug_lag, np.asarray(grad_aug_lag, dtype=np.float64)

        def fun_scaled(y):
            # Unscale to physical DOFs, evaluate L and dL/dx, then rescale gradient:
            # dL/dy = dL/dx * dx/dy = dL/dx * scale
            L, dL = fun(y * scale)
            return L, dL * scale

        # With DOF scaling, the gradient in y-space is grad_x * scale, so its
        # inf-norm is scale * ||grad_x||_inf. L-BFGS-B converges when
        # ||grad_y||_inf < gtol. To enforce the same physical convergence
        # criterion regardless of scale, set gtol = omega_k * min(scale).
        options['gtol'] = omega_k * float(np.min(scale))

        # On the first outer iteration, run a finite-difference Taylor test to verify gradient
        # correctness. The test runs in scaled space. Small eps values keep perturbations within
        # the linear regime of the forward model.
        y = x / scale
        if k == 1:
            h = np.random.uniform(size=y.shape)
            _, dJ0 = fun_scaled(y)
            dJh = sum(dJ0 * h)
            err = 1e100
            for eps in [1e-5, 1e-6, 1e-7]:
                J1, _ = fun_scaled(y + eps*h)
                J2, _ = fun_scaled(y - eps*h)
                err_new = np.abs((J1-J2)/(2*eps) - dJh)
                if not (err_new < err * 0.5) and err > 1e-10:
                    _print("WARNING: Taylor test failed, err_new = {:.2e}, err = {:.2e}. "
                          "Proceeding with optimization anyway.".format(err_new, err))
                    break
                err = err_new
            else:
                _print("Taylor test passed")
        x = dofs_before.copy()
        y = x / scale

        # Save warmstart before inner optimizer so trial-point evaluations inside
        # minimize() cannot contaminate the warm-start for the post-minimize
        # evaluation at the committed DOFs.
        warmstart_snapshot = f.save_warmstart() if hasattr(f, 'save_warmstart') else None

        res = minimize(fun_scaled, y, method=minimize_method, options=options,
                       jac=True, tol=argmin_tol)
        x = res.x * scale

        # Restore warmstart before evaluating at the committed DOFs.  After
        # minimize() the _last_valid_iota etc. reflect the last *trial* point,
        # not the committed best x.  The pre-inner warmstart (from the previous
        # outer iteration's committed point) is a better starting guess for Newton.
        if warmstart_snapshot is not None:
            f.restore_warmstart(warmstart_snapshot)

        grad_vec = grad_augmented_lagrangian(x, f, equality_constraints, lag_mul, mu_k)
        grad_aug_lag_norm = np.linalg.norm(grad_vec)
        c_vals = np.array([J.J() for J in equality_constraints], dtype=float)
        c_norm = np.linalg.norm(c_vals, ord=np.inf) if m_eq > 0 else 0

        if verbose:
            _print(f"Iteration {k}")
            _print(f"  Objective f(x) = {f.J():.6e}")
            _print(f"  Constraint norm (inf) = {c_norm:.3e}")
            _print(f"  ||∇L_A|| = {grad_aug_lag_norm:.3e}")
            _print(f"  Lagrange multipliers = {lag_mul}")
            if np.isscalar(mu_k):
                _print(f"  μ_k = {mu_k:.2e}")
            else:
                _print(f"  μ_k = [{', '.join(f'{m:.2e}' for m in mu_k)}]")
            _print(f"  ω_k = {omega_k:.2e}, η_k = {eta_k:.2e}")
            _print(f"  ||Δx|| = {np.linalg.norm(x - dofs_before):.3e}")
            _print("--------------------------------------------------")

        if callable(callback):
            callback(x, k)

        if m_eq > 0:
            # Print objective and per-constraint values labelled by constraint type name
            c_str = f"Iter {k}: Jf = {f.J():.2e}, "
            c_str += ', '.join(f'{_constraint_name(c_i)} = {abs(c):.2e}'
                               for c_i, c in zip(equality_constraints, c_vals))
            _print(c_str)
            if verbose:
                # |μ_i*c_i - λ_i| is the scalar multiplier on ∇g_i in ∇L_A; large values indicate
                # which constraints are dominating the gradient direction.
                eff_mul = mu_k * c_vals - lag_mul
                grad_str = "  |μc-λ|: " + ', '.join(
                    f'{_constraint_name(c_i)}={abs(v):.2e}'
                    for c_i, v in zip(equality_constraints, eff_mul))
                _print(grad_str)
        # increase penalty only for violated constraints

        if not penalty_type:
            for i in range(m_eq):
                if abs(c_vals[i]) > c_tol:
                    mu_k[i] *= tau
        # Constraints are making good progress, update Lagrange multipliers
        if c_norm < eta_k:
            # constraints sufficiently small: update multipliers
            if verbose:
                _print("*Constraints are satisfied*")
            lag_mul += -mu_k * c_vals
            # tighten tolerances based on current worst‐case mu
            omega_k = max(omega_k / mu_k_scalar, grad_tol)
            eta_k = max(eta_k / mu_k_scalar, c_tol)
        else:
            if verbose:
                _print("*Constraints are not satisfied*")
            if not penalty_type:
                mu_k_scalar = np.mean(mu_k)
            else:
                mu_k = tau * mu_k
            omega_k = max(1.0 / mu_k_scalar, grad_tol)
            eta_k = max(1.0 / (mu_k_scalar ** 0.1), c_tol)
        if verbose:
            _print("LAGRANGE MULTIPLIERS:", lag_mul)
            _print("--------------------------------------------------------------------------------------------------------------------------------------------")

        k += 1

    if verbose:
        _print('While loop finished because something became false: \n',
            'grad_tol_check = ', grad_aug_lag_norm > grad_tol,
            ', c_tol_check = ', c_norm > c_tol,
            ', maxiter_check = ', k < MAXITER_LAG,
        )
    try:
        return x, res.fun, lag_mul, mu_k
    except:  # while loop did not run a single time
        return x, None, lag_mul, mu_k
