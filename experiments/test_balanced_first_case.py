"""Resume the first formal causal case in an independent experiment."""
from .direct_evaluation import main


if __name__ == "__main__":
    main(hard=True, balanced=True, branch_examples=True,
         quick_cases={"causal": "n6_hard_001"})
