"""Run the reference-count-balanced HypoSpace experiment."""
from .direct_evaluation import main

if __name__ == "__main__":
    main(hard=True, branch_examples=True, balanced=True)
