import cvxpy as cp
import numpy as np

def find_optimal_allocation(t):
    # Define decision variables (resource allocation for each player)
    x_ami = cp.Variable(2)  # [steel, oil] for Ami
    x_tami = cp.Variable(2)  # [steel, oil] for Tami

    # Constraints: Ensure non-negativity and total resource allocation
    constraints = [
        x_ami >= 0,  # Non-negativity constraint for Ami
        x_tami >= 0,  # Non-negativity constraint for Tami
        x_ami[0] + x_tami[0] == 1,  # Total steel must sum to 1
        x_ami[1] + x_tami[1] == 1,  # Total oil must sum to 1
    ]

    # Preference values
    ami_values = np.array([1, 0])  # Ami: steel=1, oil=0
    tami_values = np.array([t, 1 - t])  # Tami: steel=t, oil=1-t

    # Compute the value each player receives from the allocation
    ami_value = ami_values @ x_ami
    tami_value = tami_values @ x_tami

    # This avoids approximation and directly uses the log of the value
    ami_log_val = cp.log(ami_value)  # Logarithm of Ami's value
    tami_log_val = cp.log(tami_value)  # Logarithm of Tami's value

    # Constraints to ensure positive values for the log (logarithms are only valid for positive numbers)
    constraints += [
        ami_value >= 1e-6,  # Ensure Ami's value is positive to avoid log of zero or negative
        tami_value >= 1e-6,  # Ensure Tami's value is positive to avoid log of zero or negative
    ]

    # Objective: Maximize the sum of logarithms (equivalent to maximizing product)
    objective = cp.Maximize(ami_log_val + tami_log_val)

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()

        # Print results
        print(f"For t={t}:")
        print(f"Solution status: {problem.status}")
        print(f"Solution value: {problem.value}")
        print(f"Ami's allocation: steel={x_ami[0].value:.4f}, oil={x_ami[1].value:.4f}")
        print(f"Tami's allocation: steel={x_tami[0].value:.4f}, oil={x_tami[1].value:.4f}")
        print(f"Ami's value: {(ami_values @ x_ami).value:.4f}")
        print(f"Tami's value: {(tami_values @ x_tami).value:.4f}")
        print(f"Value product: {(ami_values @ x_ami).value * (tami_values @ x_tami).value:.4f}")

        return x_ami.value, x_tami.value, problem.value

    except cp.SolverError:
        print("Solver error - problem might not be convex")
        # Analytical fallback solution
        if t >= 0.5:
            x_ami_val = np.array([1 / (2 * t), 0])
            x_tami_val = np.array([1 - 1 / (2 * t), 1])
        else:
            x_ami_val = np.array([1, 0])
            x_tami_val = np.array([0, 1])

        ami_val = ami_values @ x_ami_val
        tami_val = tami_values @ x_tami_val

        print(f"Analytical solution for t={t}:")
        print(f"Ami's allocation: steel={x_ami_val[0]:.4f}, oil={x_ami_val[1]:.4f}")
        print(f"Tami's allocation: steel={x_tami_val[0]:.4f}, oil={x_tami_val[1]:.4f}")
        print(f"Ami's value: {ami_val:.4f}")
        print(f"Tami's value: {tami_val:.4f}")
        print(f"Value product: {ami_val * tami_val:.4f}")

        return x_ami_val, x_tami_val, ami_val * tami_val
