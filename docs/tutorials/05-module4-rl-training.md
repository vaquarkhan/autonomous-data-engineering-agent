# Tutorial 5: Module 4 - ReCode GRPO + PURE

## Objective

Fine-tune the policy with Group Relative Policy Optimization, gated by sandbox execution and PURE min-form credit assignment.

## Run

```bash
genprm-train-rl --config config/phase4.yaml
```

## Configuration

```yaml
rl:
  group_size: 4
  pure_min_form: true      # V_t = min(r_t, ..., r_T)
  execution_gate: true     # ReCode: nullify rewards on sandbox failure

reward:
  process_weight: 1.0
  outcome_weight: 1.0
```

## ReCode Execution Gate

If generated SQL fails sandbox tests, all neural process rewards are zeroed:

```python
from genprm.phase4.rl.recode_grpo import TrajectoryRewards, apply_execution_gate

traj = TrajectoryRewards(step_rewards=[0.9, 0.8], execution_passed=False)
gated = apply_execution_gate(traj)
# gated.step_rewards == [0.0, 0.0]
```

## PURE Min-Form Advantages

```python
from genprm.phase4.credit.pure_min_form import pure_min_form_advantages

rewards = [0.9, 0.7, 0.5, 0.3]
advantages = pure_min_form_advantages(rewards)
# Optimizes weakest link (min future reward)
```

## Output

`data/processed/rl/grpo_updates.jsonl` - per-group advantage assignments ready for policy gradient update.
