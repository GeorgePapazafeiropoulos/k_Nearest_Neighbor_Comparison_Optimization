"""
k-Nearest Neighbor Comparison (k-NNC) based Rao Algorithm
Implementation based on:
"An efficient k-NN-based rao optimization method for optimal discrete sizing of truss structures"
Pham et al., Applied Soft Computing, 2024

The k-NNC mechanism judges whether a newly generated solution is worth
evaluating by comparing its k nearest neighbors (from the current population)
against the current solution being updated. If the majority of those neighbors
are worse than the current solution, the new candidate is labeled a
"Possibly Useless Solution" (PUS) and skipped without expensive evaluation.
"""

import numpy as np
from typing import Callable, Optional, Tuple

# Distance metric (Eq. 4 in [1]) – normalised Euclidean distance
def normalised_euclidean(x_new: np.ndarray,
                         x_pop: np.ndarray,
                         lower: np.ndarray,
                         upper: np.ndarray) -> np.ndarray:
    """
    Compute the normalised Euclidean distance between x_new and every row of x_pop.

    Parameters:
    x_new  : new solution x'p (without evaluation), shape (n_vars,)
    x_pop  : solutions of the population, from which xq is drawn, shape (NP, n_vars)
    lower  : lower bounds, shape (n_vars,)   
    upper  : upper bounds, shape (n_vars,)

    Output:
    distances : shape (NP,)
    """
    range_ = upper - lower
    # avoid division by zero
    range_ = np.where(range_ == 0, 1.0, range_)
    diff = (x_new - x_pop) / range_             # (NP, n_vars)
    return np.sqrt(np.sum(diff ** 2, axis=1))   # (NP,)

# Rao-1 and Rao-2 update formulas (Eqs. 2, 3 in [1] for continuous; 8, 9 in [1] for discrete)
def rao1_update(x_p: np.ndarray,
                x_best: np.ndarray,
                x_worst: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
    """Rao-1 formula (Eq. 2 / Eq. 8 in [1] for continuous and discrete index space respectively)."""
    r1 = rng.random(x_p.shape)
    return x_p + r1 * (x_best - x_worst)

def rao2_update(x_p: np.ndarray,
                x_best: np.ndarray,
                x_worst: np.ndarray,
                x_r: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
    """Rao-2 formula (Eq. 3 / Eq. 9 in [1] for continuous and discrete index space respectively)."""
    r1 = rng.random(x_p.shape)
    r2 = rng.random(x_p.shape)

    # Randomly decide whether |x_p| - |x_r| or |x_r| - |x_p|
    swap = rng.random(x_p.shape) < 0.5
    term2 = np.where(swap, np.abs(x_p) - np.abs(x_r), np.abs(x_r) - np.abs(x_p))

    return x_p + r1 * (x_best - x_worst) + r2 * term2

# Deb's constraint-handling comparison (Section 4.2)
def deb_better(w_a: float, c_a: float, w_b: float, c_b: float) -> bool:
    """
    Return True if solution A is better than solution B using Deb's rules:
      1. A feasible, B infeasible : A better
      2. Both feasible, w_a < w_b : A better
      3. Both infeasible, c_a < c_b : A better
      4. Both infeasible, c_a == c_b, w_a < w_b : A better
    w_a and w_b are the objective function values of solutions A and B respectively
    c_a and c_b are the constraint violation values of solutions A and B respectively
    (see equation (6) in [1] for the definition of constraint violation)
    """
    if c_a == 0 and c_b > 0:
        return True
    if c_a > 0 and c_b == 0:
        return False
    if c_a == 0 and c_b == 0:
        return w_a < w_b
    # both infeasible
    if c_a < c_b:
        return True
    if c_a == c_b:
        return w_a < w_b
    return False

# Main k-NNC-based Rao optimizer (Algorithm 2 of [1])
class KNNCRaoOptimizer:
    """
    Discrete / continuous optimizer combining Rao-1 or Rao-2 with k-NNC.

    Parameters:
    obj_fn        : callable(x) : (weight, constraint_violation)
                    Returns the objective value and total constraint violation
                    (C(A) from Eq. 6; 0 means feasible).
    n_vars        : number of design variables
    lower / upper : bound arrays, shape (n_vars,)
    discrete_sets : list of sorted arrays, one per variable (None for continuous)
    NP            : population size
    k             : number of nearest neighbours for k-NNC
    max_iter      : maximum number of iterations
    rao_variant   : 1 or 2
    tol           : convergence tolerance on relative weight spread (1e-6)
    seed          : random seed
    """

    def __init__(self,
                 n_vars: int,
                 lower: np.ndarray,
                 upper: np.ndarray,
                 obj_fn: Callable,
                 constraint_fn: Optional[Callable] = None,
                 discrete_sets: Optional[list] = None,
                 use_knnc: bool = True,
                 NP: int = 30,
                 k: int = 3,
                 max_iter: int = 500,
                 rao_variant: int = 1,
                 tol: float = 1e-6,
                 seed: int = None):

        self.obj_fn = obj_fn
        self.constraint_fn = constraint_fn
        self.n_vars = n_vars
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.discrete_sets = discrete_sets
        self.use_knnc = use_knnc
        self.NP = NP
        self.k = k
        self.max_iter = max_iter
        self.rao_variant = rao_variant
        self.tol = tol
        self.rng = np.random.default_rng(seed)

        # Statistics
        self.n_obj_evals = 0  # NFEs (number of objective function evaluations)
        self.n_con_evals = 0  # NCFs (or NSAs, i.e. full structural analysis including constraints)
        self.n_skipped = 0

    # Map continuous value to nearest discrete option
    def _to_discrete(self, x: np.ndarray) -> np.ndarray:
        if self.discrete_sets is None:
            return x
        result = np.empty_like(x)
        for i, val in enumerate(x):
            arr = np.asarray(self.discrete_sets[i])
            idx = np.argmin(np.abs(arr - val))
            result[i] = arr[idx]
        return result

    # Initialise population using index-based rounding (Eq. 7)
    def _init_population(self) -> np.ndarray:
        pop = np.empty((self.NP, self.n_vars))
        for p in range(self.NP):
            r = self.rng.random(self.n_vars)
            if self.discrete_sets is not None:
                for i in range(self.n_vars):
                    ns = len(self.discrete_sets[i])
                    idx = int(round(r[i] * (ns - 1)))
                    pop[p, i] = self.discrete_sets[i][idx]
            else:
                pop[p] = self.lower + r * (self.upper - self.lower)
        return pop

    # Evaluate a solution (only objective function value without constraint violation)
    def _eval_objective(self, x: np.ndarray) -> float:
        """Evaluate objective only (cheap)."""
        self.n_obj_evals += 1
        return float(self.obj_fn(x))

    # Evaluate a solution (only constraint violation value without objective value)
    def _eval_constraint(self, x: np.ndarray) -> float:
        """Evaluate constraint violation only (expensive structural analysis)."""
        if self.constraint_fn is None:
            return 0.0
        self.n_con_evals += 1
        return float(self.constraint_fn(x))

    # Evaluate a solution (objective function value and constraint violation value)
    def _eval_both(self, x: np.ndarray) -> Tuple[float, float]:
        """Evaluate objective and constraint together (used at initialisation)."""
        w = self._eval_objective(x)
        c = self._eval_constraint(x)
        return w, c

    # Core k-NNC check (using fitness tuples for Deb comparison)
    # (Steps 1-4 in Section 3.2 of [1])
    def _knc_is_pus(self,
                    x_new: np.ndarray,
                    current_idx: int,
                    population: np.ndarray,
                    weights: np.ndarray,
                    violations: np.ndarray) -> bool:
        """
        k-NNC check adapted for constrained optimization (Algorithm 2).
        Decide whether x_new is a Possibly Useless Solution (PUS).

        Steps (section 3.2 in [1]):
        1. Find k nearest neighbours of x_new in the population.
        2. Compare each neighbour's fitness with x_current's fitness.
        3. Count n_better = number of neighbours better than x_current.
        4. If n_better < k - n_better  (i.e. majority are NOT better) : PUS.

        """
        # Step 1: find k nearest neighbours of x_new in the population
        distances = normalised_euclidean(x_new, population, self.lower, self.upper)
        distances[current_idx] = np.inf  # exclude self
        nn_indices = np.argsort(distances)[:self.k]

        # Steps 2-3: count how many neighbours are better than x_current
        w_curr = weights[current_idx]
        c_curr = violations[current_idx]
        n_better = sum(
            1 for idx in nn_indices
            if deb_better(weights[idx], violations[idx], w_curr, c_curr)
        )

        # Step 4: PUS if majority are NOT better
        return n_better < (self.k - n_better)

    # Main optimization loop (Algorithm 2 in [1])
    def optimize(self) -> dict:
        """
        Run the k-NNC-based Rao optimization.

        Output:
        dict with keys:
          best_x               : best solution found
          best_weight          : best objective value
          best_violation       : constraint violation of best solution
          n_obj_evals          : total objective evaluations (NFEs)
          n_con_evals          : total constraint violation analyses (or structural analyses, NSAs)
          skip_rate            : fraction of candidates skipped by k-NNC
          history_weight       : best weight per iteration
        """
        # Initialisation
        population = self._init_population()
        weights = np.empty(self.NP)
        violations = np.empty(self.NP)
        for p in range(self.NP):
            # initial evaluation includes both objective and constraints (full analysis)
            weights[p], violations[p] = self._eval_both(population[p])

        history_weight = []
        total_candidates = 0

        # Main loop
        for iteration in range(self.max_iter):
            # Identify best and worst (by Deb's rules, feasible first)
            order = sorted(range(self.NP),
                           key=lambda i: (violations[i] > 0, weights[i]))
            best_idx = order[0]
            worst_idx = order[-1]
            x_best = population[best_idx].copy()
            x_worst = population[worst_idx].copy()

            for p in range(self.NP):
                x_p = population[p].copy()

                # Generate candidate via Rao formula
                if self.rao_variant == 1:
                    x_new_cont = rao1_update(x_p, x_best, x_worst, self.rng)
                else:
                    r_idx = self.rng.integers(0, self.NP)
                    while r_idx == p:
                        r_idx = self.rng.integers(0, self.NP)
                    x_r = population[r_idx]
                    x_new_cont = rao2_update(x_p, x_best, x_worst, x_r, self.rng)

                # Clip to bounds and discretize (rounding step, eqs. 8-9 in [1])
                x_new_cont = np.clip(x_new_cont, self.lower, self.upper)
                x_new = self._to_discrete(x_new_cont)

                total_candidates += 1

                if self.use_knnc:
                    is_pus = self._knc_is_pus(x_new, p, population, weights, violations)
                    if is_pus:
                        self.n_skipped += 1
                        continue

                # Evaluate objective only (cheap) (Step 3 of constrained k-NNC)
                w_new = self._eval_objective(x_new)

                # Decide whether constrain violation evaluation (full structural analysis) is needed (Step 4)
                # If current is feasible and new weight is already worse, skip analysis
                if violations[p] == 0 and w_new > weights[p]:
                    continue  # no calculation of constrain violation needed

                # Evaluate constraint violation only (expensive)
                c_new = self._eval_constraint(x_new)

                # Selection
                if deb_better(w_new, c_new, weights[p], violations[p]):
                    population[p] = x_new
                    weights[p] = w_new
                    violations[p] = c_new

            # Convergence check
            best_w = weights[np.argmin(
                [weights[i] if violations[i] == 0 else np.inf for i in range(self.NP)]
            )]
            mean_w = np.mean(weights)
            history_weight.append(best_w)

            if best_w > 0 and abs(mean_w / best_w - 1) < self.tol:
                break

        # Collect results
        feasible = [(weights[i], violations[i], i)
                    for i in range(self.NP) if violations[i] == 0]
        if feasible:
            best_w, best_c, best_i = min(feasible)
        else:
            best_i = int(np.argmin(violations))
            best_w = weights[best_i]
            best_c = violations[best_i]

        skip_rate = self.n_skipped / total_candidates if total_candidates > 0 else 0.0

        return {
            "best_x": population[best_i].copy(),
            "best_weight": best_w,
            "best_violation": best_c,
            "n_obj_evals": self.n_obj_evals,
            "n_con_evals": self.n_con_evals,
            "skip_rate": skip_rate,
            "history_weight": history_weight,
        }

if __name__ == "__main__":
    """
    Example: constrained continuous optimization
    
    Problem:
      Minimise   f(x) = sum(x_i^2)          (Sphere, n=5)
    
      Subject to:
        g1(x): sum(x_i) >= 3                (forces the solution away from the unconstrained minimum at the origin)
        g2(x): x1^2 + x2^2 <= 9             (stay inside a circle in (x1,x2))
        g3(x): sin(x1) + cos(x2) <= 1.2     (nonlinear trigonometric bound)
    
      Search space: x_i in [-4, 4], i = 1..5
    
    Description:
      Without g1 the unconstrained minimum is x* = (0,...,0) with f=0.
      g1 forces sum(x_i) >= 3. By symmetry the constrained optimum lies on the
      hyperplane sum(x_i) = 3, with each x_i = 3/5 = 0.6, giving f* = 5*(0.6^2)
      = 1.8, provided g2 and g3 are satisfied there.
      Check g2: x1^2+x2^2 = 0.36+0.36 = 0.72 <= 9  OK
      Check g3: sin(0.6)+cos(0.6) ≈ 0.565+0.825 = 1.39 > 1.2  NOT OK
      So g3 is active and pushes x1 away from 0.6. The true constrained minimum
      must be found numerically; the expected feasible weight is slightly above 1.8.
    
    Constraint violation follows Eq. 6 of [1]]:
      C(x) = sum_i max(0, c_i(x)/[c_i] - 1)
    where [c_i] is the permissible limit of the i-th constraint.
    """

    print("\n" + "=" * 60)
    print("Constrained continuous optimization demo")
    print("  min  f(x) = sum(x_i^2),  x in [-4,4]^5")
    print("  s.t. g1: sum(x_i) >= 3")
    print("       g2: x1^2 + x2^2 <= 9")
    print("       g3: sin(x1) + cos(x2) <= 1.2   (nonlinear)")
    print("=" * 60)


    def constrained_sphere_obj(x):
        """
        Objective: Sphere function f(x) = sum(x_i^2).
        """
        return float(np.sum(x ** 2))

    def constrained_sphere_con(x):
        """
        Constraints (paper Eq. 6 normalisation):
          g1: sum(x) >= 3               (rewritten as  3 - sum(x) <= 0)
              c1 = (3 - sum(x)) / 3     permissible limit = 0 (i.e. <=0)
          g2: x1^2+x2^2 <= 9
              c2 = (x[0]**2 + x[1]**2) / 9 - 1   <=0
          g3: sin(x1)+cos(x2) <= 1.2
              c3 = (sin(x[0])+cos(x[1])) / 1.2 - 1  <=0

        Constraint violation C(x) = sum max(0, c_i).
        C(x) = 0 means all constraints satisfied (feasible).
        """
        c1 = (3.0 - np.sum(x)) / 3.0
        c2 = (x[0]**2 + x[1]**2) / 9.0 - 1.0
        c3 = (np.sin(x[0]) + np.cos(x[1])) / 1.2 - 1.0
        return float(np.sum([max(0.0, c) for c in [c1, c2, c3]]))

    # Run both Rao variants and compare
    n_vars_c = 5
    lower_c = np.full(n_vars_c, -4.0)
    upper_c = np.full(n_vars_c,  4.0)

    print(f"\n{'Method':<20} {'Best f(x)':>12} {'Violation':>12} "
        f"{'NFEs':>8} {'NCFs':>8} {'Skip%':>8}")
    print("-" * 73)

    constrained_results = {}
    for variant in [1, 2]:
        for k_val, use_knnc in [(3, True), (5, True), (None, False)]:
            if use_knnc:
                label = f"Rao{variant}-kNNC(k={k_val})"
            else:
                label = f"Rao{variant} (no kNNC)"
            opt_c = KNNCRaoOptimizer(
                obj_fn=constrained_sphere_obj,
                constraint_fn=constrained_sphere_con,
                n_vars=n_vars_c,
                lower=lower_c,
                upper=upper_c,
                discrete_sets=None,
                NP=30,
                k=k_val if k_val else 3,   # k is irrelevant when use_knnc=False
                max_iter=500,
                rao_variant=variant,
                tol=1e-8,
                seed=None,
                use_knnc=use_knnc,
            )
            res_c = opt_c.optimize()
            constrained_results[label] = res_c
            print(f"{label:<20} {res_c['best_weight']:>12.6f} "
                f"{res_c['best_violation']:>12.6f} "
                f"{res_c['n_obj_evals']:>8d} "
                f"{res_c['n_con_evals']:>8d} "
                f"{res_c['skip_rate']*100:>7.1f}% ")

    # Print best solution details for all cases
    for variant in [1, 2]:
        for k_val, use_knnc in [(3, True), (5, True), (None, False)]:
            if use_knnc:
                label = f"Rao{variant}-kNNC(k={k_val})"
            else:
                label = f"Rao{variant} (no kNNC)"
            best_res = constrained_results[label]
            x_best_c = best_res["best_x"]
            print(f"\nBest solution detail [{label}]:")
            print(f"  x*          = {np.round(x_best_c, 6)}")
            print(f"  f(x*)       = {best_res['best_weight']:.6f}")
            print(f"  sum(x*)     = {np.sum(x_best_c):.6f}  (must be >= 3.0)")
            print(f"  x1^2+x2^2   = {x_best_c[0]**2 + x_best_c[1]**2:.6f}  (must be <= 9.0)")
            print(f"  sin+cos     = {np.sin(x_best_c[0])+np.cos(x_best_c[1]):.6f}  (must be <= 1.2)")
            print(f"  Feasible?   = {best_res['best_violation'] == 0.0}")

