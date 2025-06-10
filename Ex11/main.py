from typing import List, Set, Optional, Dict
from scipy.optimize import linprog


def find_decomposition(budget: List[float], preferences: List[Set[int]]) -> Optional[List[Dict[int, float]]]:
    """
    Checks if the budget is decomposable, and if so, returns a valid decomposition.

    :param budget: list of budgets for each subject (in order)
    :param preferences: for each player, the set of subjects they support
    :return: list of contributions per player (if decomposition exists), otherwise None
    """
    num_subjects = len(budget)
    num_players = len(preferences)

    # Each variable represents a contribution of player i to subject j they support
    # Do not create variables for (i,j) that citizen i does not support on topic j, thus automatically guaranteeing zero for these values.
    var_index: List[tuple[int, int]] = []
    for i, prefs in enumerate(preferences):
        for j in prefs:
            var_index.append((i, j))

    num_vars = len(var_index)

    # Objective function: zeros (no optimization objective)
    c = [0.0] * num_vars

    # Constraint for each subject: sum of contributions to subject j equals budget[j]
    A_eq: List[List[float]] = []
    b_eq: List[float] = []
    for j in range(num_subjects):
        row = [1.0 if (i_j := col[1]) == j else 0.0 for col in var_index]
        A_eq.append(row)
        b_eq.append(budget[j])

    # Constraint for each player: total contributions equals C/n
    C = sum(budget)
    for i in range(num_players):
        row = [1.0 if (i_j := col[0]) == i else 0.0 for col in var_index]
        A_eq.append(row)
        b_eq.append(C / num_players)

    # Bounds: all contributions >= 0
    bounds = [(0, None)] * num_vars

    # Solve LP
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not res.success:
        return None

    # Build decomposition from solution
    x = res.x
    decomposition: List[Dict[int, float]] = [{} for _ in range(num_players)]
    for value, (i, j) in zip(x, var_index):
        if value > 1e-6:  # filter out numerical noise
            decomposition[i][j] = round(value, 6)

    return decomposition


def main():
    # Example from the task
    budget: List[float] = [400, 50, 50, 0]
    preferences: List[Set[int]] = [
        {0, 1},  # player 0
        {0, 2},  # player 1
        {0, 3},  # player 2
        {1, 2},  # player 3
        {0}      # player 4
    ]

    result = find_decomposition(budget, preferences)

    if result is None:
        print("Budget is not decomposable.")
    else:
        print("Budget is decomposable. Found decomposition:")
        for i, contrib in enumerate(result):
            print(f"Player {i}: {contrib}")


if __name__ == "__main__":
    main()
