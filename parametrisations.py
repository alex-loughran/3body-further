"""State builder classes for different parametrisations.

Each builder maps a parameter tuple to a 12-component state vector,
or returns None if the point should be skipped (e.g. positive energy).

These are classes rather than closures so they can be pickled by
multiprocessing.
"""

from three_body import build_state_symmetric, initial_conditions_from_params, is_negative_energy


class SymmetricBuilder:
    """Suvakov symmetric parametrisation.

    Grid coordinates: (vx, vy)
    Fixed positions: r1=(-1,0), r2=(1,0), r3=(0,0)
    Zero angular momentum.
    """
    def __call__(self, params):
        vx, vy = params
        return build_state_symmetric(vx, vy)


class BHHBuilder:
    """BHH (Jankovic) parametrisation at fixed L.

    Grid coordinates: (a, c)
    Collinear Jacobi positions, velocities determined by (a, b, c, d)
    where d = (L - a*c) / b.
    """
    def __init__(self, L, b=1.0):
        self.L = L
        self.b = b

    def __call__(self, params):
        a, c = params
        if a <= 0.001:
            return None
        if not is_negative_energy(a, c, self.L, self.b):
            return None
        return initial_conditions_from_params(a, c, self.L, self.b)
