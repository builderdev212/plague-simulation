import numpy as np
from numpy.linalg import eig, solve


class PlagueParameters:
    # Initial population
    INITIAL_HEALTHY = 200
    INITIAL_SICK = 0
    INITIAL_DEAD = 0

    # Percentage of change per each unit of time
    HEALTHY_WHO_STAY_HEALTHY = 0.3
    HEALTHY_WHO_GET_SICK = 0.6
    HEALTHY_WHO_DIE = 0.1

    SICK_WHO_GET_HEALTHY = 0.2
    SICK_WHO_STAY_SICK = 0.2
    SICK_WHO_DIE = 0.6

    DEAD_WHO_GET_HEALTHY = 0
    DEAD_WHO_GET_SICK = 0
    DEAD_WHO_STAY_DEAD = 1

    DIFFERENCE_EQUATION_MATRIX = np.array(
        [
            [HEALTHY_WHO_STAY_HEALTHY, SICK_WHO_GET_HEALTHY, DEAD_WHO_GET_HEALTHY],
            [HEALTHY_WHO_GET_SICK, SICK_WHO_STAY_SICK, DEAD_WHO_GET_SICK],
            [HEALTHY_WHO_DIE, SICK_WHO_DIE, DEAD_WHO_STAY_DEAD],
        ]
    )

    INITIAL_CONDITIONS_MATRIX = np.array(
        [
            [INITIAL_HEALTHY],
            [INITIAL_SICK],
            [INITIAL_DEAD],
        ]
    )


def sim(cycles):
    # Solve for eigenvalues and eigenvectors of difference equation matrix.
    w, e = eig(PlagueParameters.DIFFERENCE_EQUATION_MATRIX)

    # Solve for the weights on the eigenvectors.
    c = solve(e, PlagueParameters.INITIAL_CONDITIONS_MATRIX)

    # Split the eigenvectors into their own vectors.
    e1, e2, e3 = e.T

    # Multiply the eigenvectors by their weight and eigenvalue to the k value.
    e1 *= c[0] * (w[0] ** cycles)
    e2 *= c[1] * (w[1] ** cycles)
    e3 *= c[2] * (w[2] ** cycles)

    # Return the final result by adding the modified eigenvectors.
    return e1 + e2 + e3


if __name__ == "__main__":
    cycles = int(input("Number of cycles to run the simulation for: "))
    results = sim(cycles)
    print(f"Healthy: {results[0]:.0f}\nSick: {results[1]:.0f}\nDead: {results[2]:.0f}")
