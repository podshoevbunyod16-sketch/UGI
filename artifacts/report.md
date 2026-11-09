# UDI MVP Verification Report

- Status: **verified**
- Verification checks: **31/31 passed**
- Seed: `20261108`
- Input hash: `818e4f5e799bc723afbc59cc695e58a99f9c4b94cfa8424cbca4adabb4f411cb`

## Problem

Discrete linear system with bounded control:
`x[t+1] = a*x[t] + b*u[t] + disturbance[t]`

Parameters: `a=1.08`, `b=0.8`, `horizon=40`, `control_limit=1.0`.

## Representations

- `R-proportional`: `proportional` {'k': 0.85}
- `R-damped`: `damped` {'k': 0.62, 'd': 0.22}
- `R-bounded`: `bounded` {'k': 1.25}
- `R-proportional::increase_gain::0.08`: `proportional` {'k': 0.9299999999999999}
- `R-proportional::add_damping::0.12`: `damped` {'k': 0.85, 'd': 0.12}

## Mutations

- `M-001` `passed`: `{'operator': 'increase_gain', 'value': 0.08, 'parameters': {'k': 0.9299999999999999}}`; error=`None`
- `M-002` `rejected`: `{'operator': 'increase_gain', 'value': 3.0, 'parameters': {'k': 3.85}}`; error=`closed_loop_not_stable`
- `M-003` `passed`: `{'operator': 'add_damping', 'value': 0.12, 'parameters': {'k': 0.85, 'd': 0.12}}`; error=`None`

## Competing hypotheses

- **H1**: Увеличение gain быстрее уменьшает ошибку без насыщения. Prediction: Высокий gain даст меньшее settling_time в сценарии без сильного возмущения.
- **H2**: Демпфирование улучшает устойчивость при переменном возмущении. Prediction: Damped-контроллер даст меньший max_abs_state в сценарии alternating.
- **H3**: Ограничение актуатора повышает безопасность ценой скорости. Prediction: Bounded-контроллер уменьшит max_abs_control, но увеличит settling_time.

## Cross-model transfer

- `T-001` `passed`: `{'added_edge': ('disturbance', 'control')}`; error=`None`
- `T-002` `rejected`: `{'added_edge': ('next_state', 'state')}`; error=`causal_cycle`

## Phase 3 numerical optimization

Scenario: `alternating`
Baseline objective: `1.57266108`
Transferred objective: `1.24958478`
Absolute improvement: `0.3230763`
Relative improvement: `20.54%`
Baseline parameters: `{'k': 1.0, 'q': 0.0}`
Transferred parameters: `{'k': 1.25976562, 'q': -1.08886719}`

## Experiment

Scenario: `alternating`

| Model | Cost | Settling time | Max state | Stable |
|---|---:|---:|---:|---|
| `R-proportional` | 1.62809213 | 41 | 1.0 | True |
| `R-bounded` | 1.73608539 | 41 | 1.0 | True |
| `R-damped` | 2.20706228 | 41 | 1.0 | True |

Selection rationale: Alternating disturbance maximizes divergence between damping and direct feedback predictions.

## Phase 4 held-out evaluation

Train scenario: `alternating`
Held-out scenarios: `('nominal', 'pulse', 'ramp')`
Transfer wins: `3/3`
Mean absolute improvement: `0.25202891`
Mean relative improvement: `16.52%`

| Held-out scenario | Baseline | Transferred | Improvement | Relative |
|---|---:|---:|---:|---:|
| `nominal` | 1.154437 | 1.14314533 | 0.01129166 | 0.98% |
| `pulse` | 1.37633843 | 1.15082313 | 0.2255153 | 16.39% |
| `ramp` | 1.61297176 | 1.09369198 | 0.51927978 | 32.19% |

Parameters were frozen after training; no re-optimization was performed on held-out scenarios.

## Phase 5 dynamic-system environment suite

Source environment: `control-mvp-001`
Transfer wins: `3/3`
Mean absolute improvement: `0.26450442`
Mean relative improvement: `14.90%`

| Environment | a | b | Limit | Transfer | Baseline | Transferred | Improvement |
|---|---:|---:|---:|---|---:|---:|---:|
| `env-fast` | 1.18 | 0.9 | 1.0 | passed | 1.57266108 | 1.24377487 | 0.32888621 |
| `env-damped` | 0.94 | 0.65 | 0.8 | passed | 1.64249166 | 1.3855658 | 0.25692586 |
| `env-weak-actuator` | 1.08 | 0.45 | 0.7 | passed | 2.55122927 | 2.34352808 | 0.20770119 |

The same structural transfer operator was applied in every environment; numerical parameters were recalibrated independently per environment.

## Phase 6 zero-shot parameter transfer

Source environment: `control-mvp-001`
Scenario: `alternating`
Targets: `('env-fast', 'env-damped', 'env-weak-actuator')`
Transfer wins: `3/3`
Mean absolute improvement: `0.28680279`
Mean relative improvement: `15.78%`

| Target | Transfer | Baseline | Transferred | Improvement | Relative |
|---|---|---:|---:|---:|---:|
| `env-fast` | passed | 1.57266108 | 1.24875703 | 0.32390406 | 20.60% |
| `env-damped` | passed | 1.69173403 | 1.3940379 | 0.29769613 | 17.60% |
| `env-weak-actuator` | passed | 2.61267514 | 2.37386697 | 0.23880817 | 9.14% |

Source parameters were frozen and no optimizer was run in target environments.

## Phase 7 preregistered random zero-shot suite

Generation seed: `424242`
Generated targets: `20`
Criterion pass: **True**
Transfer win rate: `100.00%` (required ≥ `75.00%`)
Stable rate: `100.00%` (required ≥ `95.00%`)
Mean relative improvement: `20.17%` (required ≥ `0.00%`)

| Random environment | a | b | Limit | Improvement | Relative | Stable |
|---|---:|---:|---:|---:|---:|---|
| `random-env-001` | 1.04141837 | 0.51997572 | 1.02158765 | 0.18725513 | 10.74% | True |
| `random-env-002` | 0.98918766 | 0.99264804 | 1.12888888 | 0.68940465 | 38.00% | True |
| `random-env-003` | 0.99114363 | 0.65256054 | 0.87212468 | 0.1783818 | 11.61% | True |
| `random-env-004` | 1.11274106 | 0.84759305 | 0.8558123 | 0.21982528 | 14.42% | True |
| `random-env-005` | 1.07280232 | 0.36938903 | 0.93699243 | 0.28684223 | 11.29% | True |
| `random-env-006` | 1.09294329 | 1.07584408 | 0.94461451 | 0.67883161 | 36.05% | True |
| `random-env-007` | 0.93310673 | 0.72383455 | 1.14285045 | 0.3765975 | 24.58% | True |
| `random-env-008` | 1.00250612 | 0.89314678 | 0.91240784 | 0.67114914 | 35.78% | True |
| `random-env-009` | 1.13989104 | 0.82251111 | 0.68217426 | 0.3365636 | 17.46% | True |
| `random-env-010` | 1.18116337 | 0.68923006 | 0.65591686 | 0.25703439 | 11.31% | True |
| `random-env-011` | 0.90400746 | 0.91353133 | 1.08513452 | 0.3348141 | 23.69% | True |
| `random-env-012` | 1.01230153 | 0.73617099 | 1.07859049 | 0.21600695 | 15.47% | True |
| `random-env-013` | 0.88718741 | 0.91881688 | 0.82684946 | 0.7057957 | 38.13% | True |
| `random-env-014` | 1.16729605 | 0.43563911 | 1.01674744 | 0.47291494 | 16.47% | True |
| `random-env-015` | 1.1265695 | 0.48911777 | 1.0458653 | 0.279299 | 13.21% | True |
| `random-env-016` | 1.13499558 | 0.92387356 | 1.08198108 | 0.25813711 | 18.55% | True |
| `random-env-017` | 0.95478537 | 0.44280287 | 0.79124468 | 0.16797194 | 8.95% | True |
| `random-env-018` | 0.89258787 | 0.71773565 | 1.15829188 | 0.40423214 | 26.11% | True |
| `random-env-019` | 0.96480741 | 0.96430887 | 0.69108803 | 0.29329993 | 19.49% | True |
| `random-env-020` | 1.03790101 | 0.55163875 | 1.00135221 | 0.20824997 | 12.01% | True |

The criterion and parameter ranges were fixed before random generation; target parameters remained frozen and no re-optimization was performed.

## Phase 8 adversarial and boundary stress suite

Protocol pass: **True**
Failure cases detected: `6` (required ≥ `1`)

| Case | Category | a | b | Limit | Baseline | Transferred | Improvement | Failure modes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `boundary-high-a` | boundary | 1.2 | 0.35 | 0.6 | 3521.46192615 | 3521.46192615 | 0.0 | `['baseline_actuator_saturation', 'transferred_actuator_saturation']` |
| `boundary-low-b` | boundary | 1.05 | 0.35 | 1.2 | 2.48064315 | 2.05391528 | 0.42672787 | `['transferred_actuator_saturation']` |
| `boundary-low-limit` | boundary | 1.08 | 0.8 | 0.2 | 5.69882007 | 5.26483568 | 0.43398439 | `['baseline_actuator_saturation', 'transferred_actuator_saturation']` |
| `adversarial-unstable` | adversarial | 1.3 | 0.2 | 0.35 | 2424193563.0473776 | 2424193563.0473776 | 0.0 | `['baseline_unstable', 'transferred_unstable', 'baseline_actuator_saturation', 'transferred_actuator_saturation']` |
| `adversarial-saturation` | adversarial | 1.25 | 0.35 | 0.25 | 96489639.0603273 | 96489639.0603273 | 0.0 | `['baseline_unstable', 'transferred_unstable', 'baseline_actuator_saturation', 'transferred_actuator_saturation']` |
| `adversarial-low-authority` | adversarial | 0.95 | 0.1 | 0.2 | 10.88869771 | 10.69138853 | 0.19730918 | `['baseline_actuator_saturation', 'transferred_actuator_saturation']` |

Stress protocol pass means the suite exercised both boundary and adversarial categories and detected at least one explicit failure mode; it is not a claim that zero-shot transfer succeeds everywhere.

## Limitations

- Computational simulation only; no physical experiment.
- Novelty and patent status were not established.
- Benchmark is intentionally small and domain-specific.
