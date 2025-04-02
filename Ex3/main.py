import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def compute_competitive_equilibrium(valuations):
    """
    Compute competitive equilibrium for resource allocation using log-sum maximization

    Parameters:
    valuations: n x m matrix where valuations[i][j] is the value of resource j to player i

    Returns:
    prices: equilibrium prices for each resource
    allocations: equilibrium allocation matrix (n x m matrix with allocation fractions)
    """
    n, m = valuations.shape  # n players, m resources

    # Step 1: Find the allocation that maximizes the sum of logarithms
    # We'll use an optimization approach with continuous allocation variables

    # Initial guess: equal allocation of each resource
    initial_allocation = np.ones((n, m)) / n

    # Flatten the allocation for the optimizer
    x0 = initial_allocation.flatten()

    # Constraint: sum of allocations for each resource = 1
    constraints = []
    for j in range(m):
        # Define the constraint: sum of x[i][j] over all i equals 1
        def resource_constraint(x, j=j):
            x_reshaped = x.reshape(n, m)
            return np.sum(x_reshaped[:, j]) - 1

        constraints.append({'type': 'eq', 'fun': resource_constraint})

    # Bounds: allocations must be between 0 and 1
    bounds = [(0, 1) for _ in range(n * m)]

    # Objective function: negative sum of logarithms of utilities
    # (we negate it because scipy.optimize.minimize minimizes)
    def objective(x):
        x_reshaped = x.reshape(n, m)
        utilities = np.sum(valuations * x_reshaped, axis=1)
        # Avoid log(0) by adding a small epsilon
        epsilon = 1e-10
        return -np.sum(np.log(utilities + epsilon))

    # Run the optimization
    result = minimize(
        objective,
        x0,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'ftol': 1e-8, 'disp': False, 'maxiter': 1000}
    )

    # Reshape the result back to n x m matrix
    allocations = result.x.reshape(n, m)

    # Round small allocation values to 0 and normalize to ensure sum equals 1
    epsilon = 1e-6
    allocations[allocations < epsilon] = 0

    # Normalize to ensure constraints are satisfied
    for j in range(m):
        if np.sum(allocations[:, j]) > 0:
            allocations[:, j] = allocations[:, j] / np.sum(allocations[:, j])

    # Step 2: Calculate equilibrium prices based on the theorem
    # p(r) = vj(r)/vj(Xj) where j is a player receiving a positive amount of resource r

    utilities = np.sum(valuations * allocations, axis=1)
    prices = np.zeros(m)

    for j in range(m):
        # Find players who receive this resource
        receiving_players = np.where(allocations[:, j] > 0)[0]
        if len(receiving_players) > 0:
            # Use the first player's valuation ratio to set the price
            # According to the theorem, this ratio should be the same for all receiving players
            player_idx = receiving_players[0]
            value_ratio = valuations[player_idx, j] / utilities[player_idx]
            prices[j] = value_ratio

    return prices, allocations


def print_results(valuations, prices, allocations):
    """Print the results in a readable format"""
    n, m = valuations.shape

    print("Valuations matrix:")
    for i in range(n):
        values_str = ", ".join(f"{v:.2f}" for v in valuations[i])
        print(f"Player {i + 1}: [{values_str}]")

    print("\nEquilibrium prices:")
    for j in range(m):
        print(f"Resource {j + 1}: {prices[j]:.4f}")

    print("\nEquilibrium allocations:")
    for i in range(n):
        alloc_str = ", ".join(f"{allocations[i][j]:.4f}" for j in range(m))
        print(f"Player {i + 1}: [{alloc_str}]")

    # Calculate and print utilities
    utilities = np.sum(valuations * allocations, axis=1)
    print("\nUtilities:")
    for i in range(n):
        print(f"Player {i + 1}: {utilities[i]:.4f}")

    # Calculate and print expenditures
    expenditures = np.sum(allocations * prices, axis=1)
    print("\nExpenditures:")
    for i in range(n):
        print(f"Player {i + 1}: {expenditures[i]:.4f}")

    # Verify optimality conditions
    print("\nVerifying optimality conditions:")
    for i in range(n):
        # Calculate "bang per buck" for each resource
        bang_per_buck = np.zeros(m)
        for j in range(m):
            if prices[j] > 0:
                bang_per_buck[j] = valuations[i][j] / prices[j]
            else:
                bang_per_buck[j] = float('inf') if valuations[i][j] > 0 else 0

        max_bang = np.max(bang_per_buck)
        print(f"Player {i + 1} bang-per-buck ratios: {', '.join(f'{b:.4f}' for b in bang_per_buck)}")

        # Check if player gets only resources with maximum bang per buck
        for j in range(m):
            if allocations[i][j] > 0.01 and bang_per_buck[j] < max_bang * 0.99:
                print(f"  Warning: Player {i + 1} gets resource {j + 1} but it's not optimal")
            if allocations[i][j] < 0.01 and bang_per_buck[j] > max_bang * 1.01:
                print(f"  Warning: Player {i + 1} doesn't get resource {j + 1} but it's optimal")


def visualize_allocation(valuations, allocations, prices):
    """Visualize the allocation and prices"""
    n, m = valuations.shape

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot allocations
    im1 = ax1.imshow(allocations, cmap='Blues')
    ax1.set_title('Resource Allocation')
    ax1.set_xlabel('Resources')
    ax1.set_ylabel('Players')
    ax1.set_xticks(range(m))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels([f'R{j + 1}' for j in range(m)])
    ax1.set_yticklabels([f'P{i + 1}' for i in range(n)])

    # Add allocation values
    for i in range(n):
        for j in range(m):
            text = ax1.text(j, i, f'{allocations[i, j]:.2f}',
                            ha='center', va='center',
                            color='black' if allocations[i, j] < 0.5 else 'white')

    fig.colorbar(im1, ax=ax1, label='Allocation Fraction')

    # Plot prices
    ax2.bar(range(m), prices)
    ax2.set_title('Equilibrium Prices')
    ax2.set_xlabel('Resources')
    ax2.set_ylabel('Price')
    ax2.set_xticks(range(m))
    ax2.set_xticklabels([f'R{j + 1}' for j in range(m)])

    # Add price values
    for j, price in enumerate(prices):
        ax2.text(j, price + 0.05, f'{price:.2f}', ha='center')

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Example 1: Simple 2x2 case
    valuations_ex1 = np.array([
        [6, 3],  # Player 1 values
        [4, 5]  # Player 2 values
    ])

    prices_ex1, allocations_ex1 = compute_competitive_equilibrium(valuations_ex1)
    print("===== Example 1 =====")
    print_results(valuations_ex1, prices_ex1, allocations_ex1)
    visualize_allocation(valuations_ex1, allocations_ex1, prices_ex1)

    # Example 2: 3x3 case
    valuations_ex2 = np.array([
        [10, 5, 3],  # Player 1 values
        [8, 12, 4],  # Player 2 values
        [6, 7, 15]  # Player 3 values
    ])

    prices_ex2, allocations_ex2 = compute_competitive_equilibrium(valuations_ex2)
    print("\n===== Example 2 =====")
    print_results(valuations_ex2, prices_ex2, allocations_ex2)
    visualize_allocation(valuations_ex2, allocations_ex2, prices_ex2)

    # Example 3: Asymmetric case (3 players, 4 resources)
    valuations_ex3 = np.array([
        [20, 10, 15, 5],  # Player 1 values
        [12, 18, 8, 15],  # Player 2 values
        [8, 12, 10, 20]  # Player 3 values
    ])

    prices_ex3, allocations_ex3 = compute_competitive_equilibrium(valuations_ex3)
    print("\n===== Example 3 =====")
    print_results(valuations_ex3, prices_ex3, allocations_ex3)
    visualize_allocation(valuations_ex3, allocations_ex3, prices_ex3)