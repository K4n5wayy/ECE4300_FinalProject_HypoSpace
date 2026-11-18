"""Test branch formatting examples on the fixed hard Boolean case."""
from .direct_evaluation import main
from .test_hard_quick import QUICK_CASES

if __name__ == "__main__":
    main(hard=True, quick_cases={"boolean": QUICK_CASES["boolean"]}, branch_examples=True)
