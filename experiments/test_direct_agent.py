"""Run three fixed representative cases with the direct agent."""
from .direct_evaluation import main

QUICK_CASES = {
    "causal": "n1_004",  # Medium: distinguish direct and indirect effects across five DAGs.
    "3d": "g3_h3_b1_005",  # Easy: only the height of one occupied column is unknown.
    "boolean": "n2_inputs_00_11_outputs_01",  # Hard for coverage: ten mechanisms, including nested expressions.
}

if __name__ == "__main__":
    main(quick_cases=QUICK_CASES)
