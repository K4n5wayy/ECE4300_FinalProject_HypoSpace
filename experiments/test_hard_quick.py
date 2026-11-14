"""Evaluate one fixed case from each hard HypoSpace domain."""
from .direct_evaluation import main

QUICK_CASES = {
    "causal": "n6_hard_029",  # 57 reference DAGs
    "3d": "g3_h3_b3_006",  # 27 height assignments
    "boolean": "n1_inputs_00_outputs_0",  # 40 reference mechanisms
}

if __name__ == "__main__":
    main(hard=True, quick_cases=QUICK_CASES)
