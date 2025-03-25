from fair_allocation_solver import find_optimal_allocation


def main():
    # Test for different t values
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t_val in t_values:
        print("\n" + "=" * 50)
        find_optimal_allocation(t_val)


if __name__ == "__main__":
    main()