# Module 4: RL Fine-Tuning and Reward Hacking Safeguards
# Upstream references:
#   https://github.com/CJReinforce/PURE (min-form credit assignment)
#   https://github.com/ruc-datalab/rewardsql (GRPO + execution rewards)
#
# Components:
#   credit/pure_min_form.py   - V = min(r_t, ..., r_T) advantage override
#   rl/recode_grpo.py         - Consistency-gated Group Relative Policy Optimization
#   rl/execution_gate.py      - Nullify neural rewards on sandbox failure
#
# Status: implemented

__all__: list[str] = []
