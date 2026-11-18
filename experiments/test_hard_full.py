"""Evaluate all 90 fixed hard HypoSpace cases."""
from .direct_evaluation import main

if __name__ == "__main__":
    main(hard=True, branch_examples=True)
