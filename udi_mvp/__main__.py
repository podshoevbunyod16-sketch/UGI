from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import Problem, UDIEngine
from .reporting import write_phase9_report


def markdown_report(data: dict) -> str:
    v = data["verification"]
    lines = [
        "# UDI MVP Verification Report", "",
        f"- Status: **{v['status']}**",
        f"- Verification checks: **{v['passed']}/{v['total']} passed**",
        f"- Seed: `{data['seed']}`",
        f"- Input hash: `{data['input_hash']}`", "",
        "## Problem", "",
        "Discrete linear system with bounded control:",
        "`x[t+1] = a*x[t] + b*u[t] + disturbance[t]`", "",
        f"Parameters: `a={data['problem']['state_a']}`, `b={data['problem']['input_b']}`, "
        f"`horizon={data['problem']['horizon']}`, `control_limit={data['problem']['control_limit']}`.", "",
        "## Representations", "",
    ]
    for r in data["representations"]:
        lines.append(f"- `{r['representation_id']}`: `{r['kind']}` {r['parameters']}")
    lines += ["", "## Mutations", ""]
    for m in data["mutations"]:
        lines.append(f"- `{m['mutation_id']}` `{m['validation']}`: `{m['changed_elements']}`; error=`{m['error']}`")
    lines += ["", "## Competing hypotheses", ""]
    for h in data["hypotheses"]:
        lines.append(f"- **{h['hypothesis_id']}**: {h['statement']} Prediction: {h['prediction']}")
    lines += ["", "## Cross-model transfer", ""]
    for t in data["transfers"]:
        lines.append(f"- `{t['transfer_id']}` `{t['validation']}`: `{t['changed_elements']}`; error=`{t['error']}`")
    comparison = data["optimization_comparison"]
    lines += ["", "## Phase 3 numerical optimization", "",
              f"Scenario: `{comparison['scenario']}`",
              f"Baseline objective: `{comparison['baseline']['objective']}`",
              f"Transferred objective: `{comparison['transferred']['objective']}`",
              f"Absolute improvement: `{comparison['absolute_improvement']}`",
              f"Relative improvement: `{comparison['relative_improvement']:.2%}`",
              f"Baseline parameters: `{comparison['baseline']['parameters']}`",
              f"Transferred parameters: `{comparison['transferred']['parameters']}`"]
    e = data["experiment"]
    lines += ["", "## Experiment", "", f"Scenario: `{e['selected_scenario']}`", "", "| Model | Cost | Settling time | Max state | Stable |", "|---|---:|---:|---:|---|"]
    for row in e["ranking"]:
        lines.append(f"| `{row['representation_id']}` | {row['cost']} | {row['settling_time']} | {row['max_abs_state']} | {row['stable']} |")
    lines += ["", f"Selection rationale: {e['rationale']}"]
    held_out = data["held_out_evaluation"]
    lines += ["", "## Phase 4 held-out evaluation", "",
              f"Train scenario: `{held_out['train_scenario']}`",
              f"Held-out scenarios: `{held_out['held_out_scenarios']}`",
              f"Transfer wins: `{held_out['transfer_wins']}/{len(held_out['cases'])}`",
              f"Mean absolute improvement: `{held_out['mean_absolute_improvement']}`",
              f"Mean relative improvement: `{held_out['mean_relative_improvement']:.2%}`",
              "", "| Held-out scenario | Baseline | Transferred | Improvement | Relative |", "|---|---:|---:|---:|---:|"]
    for case in held_out["cases"]:
        lines.append(f"| `{case['scenario']}` | {case['baseline_objective']} | {case['transferred_objective']} | {case['absolute_improvement']} | {case['relative_improvement']:.2%} |")
    lines += ["", "Parameters were frozen after training; no re-optimization was performed on held-out scenarios."]
    suite = data["environment_suite"]
    lines += ["", "## Phase 5 dynamic-system environment suite", "",
              f"Source environment: `{suite['source_environment_id']}`",
              f"Transfer wins: `{suite['transfer_wins']}/{len(suite['cases'])}`",
              f"Mean absolute improvement: `{suite['mean_absolute_improvement']}`",
              f"Mean relative improvement: `{suite['mean_relative_improvement']:.2%}`",
              "", "| Environment | a | b | Limit | Transfer | Baseline | Transferred | Improvement |", "|---|---:|---:|---:|---|---:|---:|---:|"]
    for case in suite["cases"]:
        lines.append(f"| `{case['environment_id']}` | {case['state_a']} | {case['input_b']} | {case['control_limit']} | {case['transfer_validation']} | {case['baseline_objective']} | {case['transferred_objective']} | {case['absolute_improvement']} |")
    lines += ["", "The same structural transfer operator was applied in every environment; numerical parameters were recalibrated independently per environment."]
    zero_shot = data["zero_shot_evaluation"]
    lines += ["", "## Phase 6 zero-shot parameter transfer", "",
              f"Source environment: `{zero_shot['source_environment_id']}`",
              f"Scenario: `{zero_shot['scenario']}`",
              f"Targets: `{zero_shot['environment_ids']}`",
              f"Transfer wins: `{zero_shot['transfer_wins']}/{len(zero_shot['cases'])}`",
              f"Mean absolute improvement: `{zero_shot['mean_absolute_improvement']}`",
              f"Mean relative improvement: `{zero_shot['mean_relative_improvement']:.2%}`",
              "", "| Target | Transfer | Baseline | Transferred | Improvement | Relative |", "|---|---|---:|---:|---:|---:|"]
    for case in zero_shot["cases"]:
        lines.append(f"| `{case['environment_id']}` | {case['transfer_validation']} | {case['baseline_objective']} | {case['transferred_objective']} | {case['absolute_improvement']} | {case['relative_improvement']:.2%} |")
    lines += ["", "Source parameters were frozen and no optimizer was run in target environments."]
    random_suite = data["random_zero_shot_suite"]
    criterion = random_suite["preregistered_criterion"]
    lines += ["", "## Phase 7 preregistered random zero-shot suite", "",
              f"Generation seed: `{random_suite['generation_seed']}`",
              f"Generated targets: `{random_suite['generated_count']}`",
              f"Criterion pass: **{random_suite['passed']}**",
              f"Transfer win rate: `{random_suite['transfer_win_rate']:.2%}` (required ≥ `{criterion['min_transfer_win_rate']:.2%}`)",
              f"Stable rate: `{random_suite['stable_rate']:.2%}` (required ≥ `{criterion['min_stable_rate']:.2%}`)",
              f"Mean relative improvement: `{random_suite['mean_relative_improvement']:.2%}` (required ≥ `{criterion['min_mean_relative_improvement']:.2%}`)",
              "", "| Random environment | a | b | Limit | Improvement | Relative | Stable |", "|---|---:|---:|---:|---:|---:|---|"]
    for case in random_suite["cases"]:
        stable = case["baseline_stable"] and case["transferred_stable"]
        lines.append(f"| `{case['environment_id']}` | {case['state_a']} | {case['input_b']} | {case['control_limit']} | {case['absolute_improvement']} | {case['relative_improvement']:.2%} | {stable} |")
    lines += ["", "The criterion and parameter ranges were fixed before random generation; target parameters remained frozen and no re-optimization was performed."]
    stress = data["stress_suite"]
    stress_criterion = stress["preregistered_criterion"]
    lines += ["", "## Phase 8 adversarial and boundary stress suite", "",
              f"Protocol pass: **{stress['passed']}**",
              f"Failure cases detected: `{stress['failure_case_count']}` (required ≥ `{stress_criterion['min_failure_cases_to_detect']}`)",
              "", "| Case | Category | a | b | Limit | Baseline | Transferred | Improvement | Failure modes |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for case in stress["cases"]:
        lines.append(f"| `{case['environment_id']}` | {case['category']} | {case['state_a']} | {case['input_b']} | {case['control_limit']} | {case['baseline_objective']} | {case['transferred_objective']} | {case['absolute_improvement']} | `{case['failure_modes']}` |")
    lines += ["", "Stress protocol pass means the suite exercised both boundary and adversarial categories and detected at least one explicit failure mode; it is not a claim that zero-shot transfer succeeds everywhere.", "", "## Limitations", ""]
    lines += [f"- {x}" for x in data["limitations"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the reproducible UDI MVP")
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--output", type=Path, default=Path("artifacts/report.json"))
    args = parser.parse_args()

    problem = Problem("control-mvp-001", state_a=1.08, input_b=0.8, target=0.0,
                      horizon=40, control_limit=1.0, disturbance_scale=1.0)
    engine = UDIEngine(problem, seed=args.seed)
    engine.generate_representations()
    engine.generate_cross_domain_representations()
    engine.mutate("R-proportional", "increase_gain", 0.08)  # valid mutation
    engine.mutate("R-proportional", "increase_gain", 3.00)  # rejected: unstable
    engine.mutate("R-proportional", "add_damping", 0.12)    # valid structural mutation
    engine.transfer_causal_mutation("disturbance", "control")  # valid transfer
    engine.transfer_causal_mutation("next_state", "state")      # rejected: causal cycle
    engine.generate_hypotheses()
    engine.choose_and_run_experiment()
    engine.optimize_and_compare("alternating")
    engine.run_held_out_evaluation("alternating", ("nominal", "pulse", "ramp"))
    environment_suite = (
        Problem("env-fast", 1.18, 0.90, 0.0, 40, 1.0, 1.0),
        Problem("env-damped", 0.94, 0.65, 0.0, 40, 0.8, 1.0),
        Problem("env-weak-actuator", 1.08, 0.45, 0.0, 40, 0.7, 1.0),
    )
    engine.run_environment_suite(environment_suite)
    engine.run_zero_shot_environment_suite(environment_suite, "alternating")
    engine.run_preregistered_random_zero_shot_suite(requested_count=20, generation_seed=424242)
    engine.run_adversarial_boundary_suite("alternating")
    data = engine.report()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")
    args.output.with_suffix(".md").write_text(markdown_report(data))
    phase9_json = args.output.with_name("phase9_report.json")
    phase9_md = args.output.with_name("phase9_report.md")
    phase9 = write_phase9_report(data, phase9_json, phase9_md)
    print(json.dumps({"status": data["verification"]["status"],
                      "checks": f"{data['verification']['passed']}/{data['verification']['total']}",
                      "json": str(args.output), "markdown": str(args.output.with_suffix('.md')),
                      "phase9_json": str(phase9_json), "phase9_markdown": str(phase9_md),
                      "recommendations": len(phase9["recommendations"])}, ensure_ascii=False))
    return 0 if data["verification"]["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
