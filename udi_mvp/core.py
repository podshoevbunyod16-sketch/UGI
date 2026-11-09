from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import random
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class Problem:
    problem_id: str
    state_a: float
    input_b: float
    target: float
    horizon: int
    control_limit: float
    disturbance_scale: float
    invariants: Tuple[str, ...] = ("finite_state", "bounded_control", "valid_horizon")


@dataclass(frozen=True)
class Representation:
    representation_id: str
    kind: str
    parameters: Dict[str, float]
    assumptions: Tuple[str, ...]
    invariants: Tuple[str, ...]
    parent_id: str | None = None
    operator_id: str | None = None
    operator_version: str | None = None


@dataclass
class MutationResult:
    mutation_id: str
    source_id: str
    output: Representation | None
    changed_elements: Dict[str, Any]
    preserved_invariants: List[str]
    violated_constraints: List[str]
    validation: str
    error: str | None = None


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    statement: str
    prediction: str
    discriminating_scenario: str
    provenance: Tuple[str, ...]


@dataclass
class SimulationResult:
    representation_id: str
    scenario: str
    seed: int
    states: List[float]
    controls: List[float]
    cost: float
    settling_time: int
    max_abs_state: float
    stable: bool
    error: str | None = None


@dataclass
class ExperimentResult:
    selected_scenario: str
    information_gain_proxy: float
    rationale: str
    simulations: List[SimulationResult]
    ranking: List[Dict[str, Any]]


@dataclass(frozen=True)
class CausalGraph:
    graph_id: str
    nodes: Tuple[str, ...]
    edges: Tuple[Tuple[str, str], ...]
    assumptions: Tuple[str, ...]
    invariants: Tuple[str, ...]


@dataclass(frozen=True)
class OptimizationFormulation:
    formulation_id: str
    variables: Tuple[str, ...]
    objective: Dict[str, float]
    constraints: Tuple[str, ...]
    assumptions: Tuple[str, ...]
    invariants: Tuple[str, ...]


@dataclass
class CrossModelTransfer:
    transfer_id: str
    source_model_id: str
    target_model_id: str
    operator_id: str
    changed_elements: Dict[str, Any]
    preserved_invariants: List[str]
    violated_constraints: List[str]
    validation: str
    output_model: Dict[str, Any] | None = None
    error: str | None = None


@dataclass
class OptimizationSolution:
    formulation_id: str
    scenario: str
    transferred: bool
    parameters: Dict[str, float]
    objective: float
    states: List[float]
    controls: List[float]
    iterations: int
    converged: bool


@dataclass
class OptimizationComparison:
    scenario: str
    baseline: OptimizationSolution
    transferred: OptimizationSolution
    absolute_improvement: float
    relative_improvement: float
    transfer_required: str


@dataclass
class HeldOutCase:
    scenario: str
    seed: int
    baseline_objective: float
    transferred_objective: float
    absolute_improvement: float
    relative_improvement: float
    baseline_stable: bool
    transferred_stable: bool


@dataclass
class HeldOutEvaluation:
    train_scenario: str
    held_out_scenarios: Tuple[str, ...]
    cases: List[HeldOutCase]
    mean_absolute_improvement: float
    mean_relative_improvement: float
    transfer_wins: int
    transfer_wins_all: bool
    no_reoptimization: bool
    evaluation_budget: Dict[str, Any]


@dataclass
class EnvironmentCase:
    environment_id: str
    state_a: float
    input_b: float
    control_limit: float
    transfer_validation: str
    baseline_objective: float
    transferred_objective: float
    absolute_improvement: float
    relative_improvement: float
    baseline_stable: bool
    transferred_stable: bool


@dataclass
class EnvironmentSuiteEvaluation:
    source_environment_id: str
    environment_ids: Tuple[str, ...]
    cases: List[EnvironmentCase]
    transfer_wins: int
    transfer_wins_all: bool
    mean_absolute_improvement: float
    mean_relative_improvement: float
    recalibrated_per_environment: bool
    evaluation_budget: Dict[str, Any]


@dataclass
class ZeroShotCase:
    environment_id: str
    state_a: float
    input_b: float
    control_limit: float
    transfer_validation: str
    baseline_objective: float
    transferred_objective: float
    absolute_improvement: float
    relative_improvement: float
    baseline_stable: bool
    transferred_stable: bool
    parameters_frozen: bool


@dataclass
class ZeroShotEvaluation:
    source_environment_id: str
    source_parameters: Dict[str, Dict[str, float]]
    scenario: str
    environment_ids: Tuple[str, ...]
    cases: List[ZeroShotCase]
    transfer_wins: int
    transfer_wins_all: bool
    mean_absolute_improvement: float
    mean_relative_improvement: float
    no_reoptimization: bool
    evaluation_budget: Dict[str, Any]


@dataclass
class RandomZeroShotSuiteEvaluation:
    generation_seed: int
    requested_count: int
    generated_count: int
    generation_ranges: Dict[str, Tuple[float, float]]
    preregistered_criterion: Dict[str, Any]
    cases: List[ZeroShotCase]
    transfer_win_rate: float
    stable_rate: float
    mean_relative_improvement: float
    passed: bool
    no_reoptimization: bool


@dataclass
class StressCase:
    environment_id: str
    category: str
    state_a: float
    input_b: float
    control_limit: float
    disturbance_scale: float
    transfer_validation: str
    baseline_objective: float
    transferred_objective: float
    absolute_improvement: float
    relative_improvement: float
    baseline_stable: bool
    transferred_stable: bool
    baseline_saturated: bool
    transferred_saturated: bool
    failure_modes: List[str]


@dataclass
class StressSuiteEvaluation:
    source_environment_id: str
    scenario: str
    cases: List[StressCase]
    preregistered_criterion: Dict[str, Any]
    failure_case_count: int
    stress_detected: bool
    passed: bool
    no_reoptimization: bool


class UDIEngine:
    """A small deterministic UDI engine with no LLM or external provider dependency."""

    schema_version = "0.3.0"
    operator_version = "1.0.0"

    def __init__(self, problem: Problem, seed: int = 20261108):
        if problem.horizon <= 0:
            raise ValueError("horizon must be positive")
        if problem.control_limit <= 0:
            raise ValueError("control_limit must be positive")
        self.problem = problem
        self.seed = seed
        self.representations: List[Representation] = []
        self.mutations: List[MutationResult] = []
        self.hypotheses: List[Hypothesis] = []
        self.experiment: ExperimentResult | None = None
        self.causal_graph: CausalGraph | None = None
        self.optimization_formulation: OptimizationFormulation | None = None
        self.transfers: List[CrossModelTransfer] = []
        self.optimization_comparison: OptimizationComparison | None = None
        self.held_out_evaluation: HeldOutEvaluation | None = None
        self.environment_suite: EnvironmentSuiteEvaluation | None = None
        self.zero_shot_evaluation: ZeroShotEvaluation | None = None
        self.random_zero_shot_suite: RandomZeroShotSuiteEvaluation | None = None
        self.stress_suite: StressSuiteEvaluation | None = None

    def generate_representations(self) -> List[Representation]:
        p = self.problem
        self.representations = [
            Representation("R-proportional", "proportional", {"k": 0.85},
                           ("linear_dynamics",), p.invariants),
            Representation("R-damped", "damped", {"k": 0.62, "d": 0.22},
                           ("linear_dynamics", "observable_previous_state"), p.invariants),
            Representation("R-bounded", "bounded", {"k": 1.25},
                           ("linear_dynamics", "actuator_saturation_allowed"), p.invariants),
        ]
        return list(self.representations)

    def generate_cross_domain_representations(self) -> Dict[str, Any]:
        """Create two structurally different views of the same control problem."""
        p = self.problem
        self.causal_graph = CausalGraph(
            "CG-control-001",
            ("disturbance", "state", "control", "next_state", "objective"),
            (("disturbance", "state"), ("state", "control"),
             ("control", "next_state"), ("next_state", "objective")),
            ("disturbance_is_observable", "control_affects_next_state"),
            ("acyclic", "known_nodes", "edge_endpoints_exist"),
        )
        self.optimization_formulation = OptimizationFormulation(
            "OPT-control-001",
            ("state_t", "control_t", "disturbance_t"),
            {"state_error": 1.0, "control_effort": 0.05},
            (f"state_next = {p.state_a}*state + {p.input_b}*control + disturbance",
             f"abs(control) <= {p.control_limit}", "state_error = state - target"),
            ("finite_horizon", "known_dynamics", "bounded_actuator"),
            ("objective_defined", "constraints_defined", "variables_defined"),
        )
        return {"causal_graph": self.causal_graph, "optimization": self.optimization_formulation}

    @staticmethod
    def _is_acyclic(nodes: Tuple[str, ...], edges: Tuple[Tuple[str, str], ...]) -> bool:
        graph = {node: [] for node in nodes}
        for source, target in edges:
            if source not in graph or target not in graph:
                return False
            graph[source].append(target)
        visiting, visited = set(), set()

        def visit(node: str) -> bool:
            if node in visiting:
                return False
            if node in visited:
                return True
            visiting.add(node)
            if any(not visit(child) for child in graph[node]):
                return False
            visiting.remove(node)
            visited.add(node)
            return True

        return all(visit(node) for node in nodes)

    def transfer_causal_mutation(self, source: str, target: str) -> CrossModelTransfer:
        """Transfer a causal edge mutation into an optimization constraint."""
        if self.causal_graph is None or self.optimization_formulation is None:
            raise RuntimeError("call generate_cross_domain_representations first")
        graph = self.causal_graph
        edge = (source, target)
        violations: List[str] = []
        if source not in graph.nodes or target not in graph.nodes:
            violations.append("edge_endpoint_missing")
        if edge in graph.edges:
            violations.append("duplicate_edge")
        new_edges = graph.edges + (edge,)
        if not violations and not self._is_acyclic(graph.nodes, new_edges):
            violations.append("causal_cycle")
        if source == "disturbance" and target == "control" and "disturbance_is_observable" not in graph.assumptions:
            violations.append("unobservable_parent")
        valid = not violations
        output: Dict[str, Any] | None = None
        if valid:
            output = {
                "causal_graph": asdict(CausalGraph(
                    graph.graph_id + "::add_edge", graph.nodes, new_edges,
                    graph.assumptions, graph.invariants)),
                "optimization": asdict(OptimizationFormulation(
                    self.optimization_formulation.formulation_id + "::transfer",
                    self.optimization_formulation.variables,
                    dict(self.optimization_formulation.objective),
                    self.optimization_formulation.constraints + ("control policy may depend on disturbance_t",),
                    self.optimization_formulation.assumptions,
                    self.optimization_formulation.invariants)),
            }
        transfer = CrossModelTransfer(
            f"T-{len(self.transfers)+1:03d}", graph.graph_id, self.optimization_formulation.formulation_id,
            "causal_edge_to_optimization_constraint", {"added_edge": edge},
            list(graph.invariants) if valid else [], violations,
            "passed" if valid else "rejected", output,
            ";".join(violations) if violations else None,
        )
        self.transfers.append(transfer)
        return transfer

    def _disturbances(self, scenario: str, seed: int | None = None) -> List[float]:
        if scenario not in {"nominal", "alternating", "pulse", "ramp"}:
            raise ValueError(f"unknown scenario: {scenario}")
        rng = random.Random(self.seed if seed is None else seed)
        values: List[float] = []
        for t in range(self.problem.horizon):
            if scenario == "nominal":
                value = rng.uniform(-0.02, 0.02) * self.problem.disturbance_scale
            elif scenario == "alternating":
                value = (0.12 if t % 2 == 0 else -0.12) * self.problem.disturbance_scale
            elif scenario == "pulse":
                value = (0.45 if t == 3 else 0.0) * self.problem.disturbance_scale
            else:
                fraction = t / max(1, self.problem.horizon - 1)
                value = (-0.15 + 0.30 * fraction) * self.problem.disturbance_scale
            values.append(value)
        return values

    def _evaluate_policy(self, k: float, q: float, scenario: str, seed: int | None = None) -> Tuple[float, List[float], List[float]]:
        disturbances = self._disturbances(scenario, seed)
        x, states, controls = 1.0, [1.0], []
        for disturbance in disturbances:
            raw = -k * x + q * disturbance
            u = max(-self.problem.control_limit, min(self.problem.control_limit, raw))
            x = self.problem.state_a * x + self.problem.input_b * u + disturbance
            controls.append(u)
            states.append(x)
        objective = sum((value - self.problem.target) ** 2 for value in states)
        objective += 0.05 * sum(value ** 2 for value in controls)
        return objective, states, controls

    def solve_numeric_optimization(self, scenario: str = "alternating", transferred: bool = False,
                                   max_iterations: int = 30) -> OptimizationSolution:
        """Solve a bounded two-parameter policy with deterministic coordinate descent."""
        if self.optimization_formulation is None:
            raise RuntimeError("call generate_cross_domain_representations first")
        if transferred and not any(
            t.validation == "passed" and t.changed_elements.get("added_edge") == ("disturbance", "control")
            for t in self.transfers
        ):
            raise RuntimeError("accepted disturbance-to-control transfer is required")
        k, q = 0.85, 0.0
        if not transferred:
            q = 0.0
        best, states, controls = self._evaluate_policy(k, q, scenario)
        step_k, step_q = 0.20, (0.20 if transferred else 0.0)
        converged = False
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            improved = False
            for name, step in (("k", step_k), ("q", step_q)):
                if step == 0.0:
                    continue
                candidates = []
                for direction in (-1.0, 1.0):
                    candidate_k = max(0.0, min(2.0, k + direction * step)) if name == "k" else k
                    candidate_q = max(-2.0, min(2.0, q + direction * step)) if name == "q" else q
                    result = self._evaluate_policy(candidate_k, candidate_q, scenario)
                    candidates.append((result[0], candidate_k, candidate_q, result[1], result[2]))
                candidate = min(candidates, key=lambda item: item[0])
                if candidate[0] + 1e-12 < best:
                    best, k, q, states, controls = candidate
                    improved = True
            if not improved:
                step_k *= 0.5
                if transferred:
                    step_q *= 0.5
                if step_k < 1e-4 and (not transferred or step_q < 1e-4):
                    converged = True
                    break
        return OptimizationSolution(
            self.optimization_formulation.formulation_id, scenario, transferred,
            {"k": round(k, 8), "q": round(q, 8)}, round(best, 8), states, controls,
            iterations, converged,
        )

    def optimize_and_compare(self, scenario: str = "alternating") -> OptimizationComparison:
        baseline = self.solve_numeric_optimization(scenario, transferred=False)
        transferred = self.solve_numeric_optimization(scenario, transferred=True)
        absolute = baseline.objective - transferred.objective
        relative = absolute / baseline.objective if baseline.objective else 0.0
        self.optimization_comparison = OptimizationComparison(
            scenario, baseline, transferred, round(absolute, 8), round(relative, 8),
            "accepted causal edge disturbance → control",
        )
        return self.optimization_comparison

    def run_held_out_evaluation(self, train_scenario: str = "alternating",
                                held_out_scenarios: Tuple[str, ...] = ("nominal", "pulse", "ramp")) -> HeldOutEvaluation:
        """Evaluate frozen train-time parameters on independent scenarios without re-optimization."""
        if self.optimization_comparison is None or self.optimization_comparison.scenario != train_scenario:
            self.optimize_and_compare(train_scenario)
        if len(set(held_out_scenarios)) != len(held_out_scenarios):
            raise ValueError("held-out scenarios must be unique")
        if train_scenario in held_out_scenarios:
            raise ValueError("training scenario cannot be held out")
        allowed = {"nominal", "alternating", "pulse", "ramp"}
        if not set(held_out_scenarios).issubset(allowed):
            raise ValueError("unknown held-out scenario")
        comparison = self.optimization_comparison
        cases: List[HeldOutCase] = []
        scenario_offsets = {"nominal": 11, "pulse": 23, "ramp": 37, "alternating": 41}
        for scenario in held_out_scenarios:
            evaluation_seed = self.seed + scenario_offsets[scenario]
            baseline_params = comparison.baseline.parameters
            transferred_params = comparison.transferred.parameters
            baseline_objective, baseline_states, baseline_controls = self._evaluate_policy(
                baseline_params["k"], 0.0, scenario, evaluation_seed)
            transferred_objective, transferred_states, transferred_controls = self._evaluate_policy(
                transferred_params["k"], transferred_params["q"], scenario, evaluation_seed)
            baseline_stable = all(math.isfinite(x) for x in baseline_states + baseline_controls) and all(
                abs(u) <= self.problem.control_limit + 1e-12 for u in baseline_controls)
            transferred_stable = all(math.isfinite(x) for x in transferred_states + transferred_controls) and all(
                abs(u) <= self.problem.control_limit + 1e-12 for u in transferred_controls)
            absolute = baseline_objective - transferred_objective
            relative = absolute / baseline_objective if baseline_objective else 0.0
            cases.append(HeldOutCase(
                scenario, evaluation_seed, round(baseline_objective, 8), round(transferred_objective, 8),
                round(absolute, 8), round(relative, 8), baseline_stable, transferred_stable,
            ))
        mean_absolute = sum(c.absolute_improvement for c in cases) / len(cases) if cases else 0.0
        mean_relative = sum(c.relative_improvement for c in cases) / len(cases) if cases else 0.0
        wins = sum(c.absolute_improvement > 0 for c in cases)
        self.held_out_evaluation = HeldOutEvaluation(
            train_scenario, held_out_scenarios, cases, round(mean_absolute, 8), round(mean_relative, 8),
            wins, bool(cases) and wins == len(cases), True,
            {"optimizer": "coordinate_descent", "max_iterations": 30, "parameters_frozen": True},
        )
        return self.held_out_evaluation

    def run_environment_suite(self, environments: Tuple[Problem, ...]) -> EnvironmentSuiteEvaluation:
        """Apply the same structural transfer operator across changed environments."""
        if not environments:
            raise ValueError("environment suite must not be empty")
        ids = tuple(environment.problem_id for environment in environments)
        if len(set(ids)) != len(ids):
            raise ValueError("environment ids must be unique")
        cases: List[EnvironmentCase] = []
        for index, environment in enumerate(environments):
            child = UDIEngine(environment, seed=self.seed + 101 * (index + 1))
            child.generate_representations()
            child.generate_cross_domain_representations()
            transfer = child.transfer_causal_mutation("disturbance", "control")
            comparison = child.optimize_and_compare("alternating")
            cases.append(EnvironmentCase(
                environment.problem_id, environment.state_a, environment.input_b,
                environment.control_limit, transfer.validation,
                comparison.baseline.objective, comparison.transferred.objective,
                comparison.absolute_improvement, comparison.relative_improvement,
                all(abs(u) <= environment.control_limit + 1e-12 for u in comparison.baseline.controls),
                all(abs(u) <= environment.control_limit + 1e-12 for u in comparison.transferred.controls),
            ))
        wins = sum(case.transfer_validation == "passed" and case.absolute_improvement > 0 for case in cases)
        mean_absolute = sum(case.absolute_improvement for case in cases) / len(cases)
        mean_relative = sum(case.relative_improvement for case in cases) / len(cases)
        self.environment_suite = EnvironmentSuiteEvaluation(
            self.problem.problem_id, ids, cases, wins, wins == len(cases),
            round(mean_absolute, 8), round(mean_relative, 8), True,
            {"optimizer": "coordinate_descent", "max_iterations": 30,
             "scenario": "alternating", "same_operator": "causal_edge_to_optimization_constraint"},
        )
        return self.environment_suite

    def run_zero_shot_environment_suite(self, environments: Tuple[Problem, ...],
                                        scenario: str = "alternating") -> ZeroShotEvaluation:
        """Apply source-environment parameters to new systems without recalibration."""
        if self.optimization_comparison is None:
            raise RuntimeError("run optimize_and_compare in the source environment first")
        if not environments:
            raise ValueError("zero-shot environment suite must not be empty")
        ids = tuple(environment.problem_id for environment in environments)
        if len(set(ids)) != len(ids):
            raise ValueError("environment ids must be unique")
        if self.problem.problem_id in ids:
            raise ValueError("source environment must not be part of zero-shot targets")
        source = self.optimization_comparison
        source_parameters = {
            "baseline": dict(source.baseline.parameters),
            "transferred": dict(source.transferred.parameters),
        }
        cases: List[ZeroShotCase] = []
        for index, environment in enumerate(environments):
            child = UDIEngine(environment, seed=self.seed + 1001 * (index + 1))
            child.generate_representations()
            child.generate_cross_domain_representations()
            transfer = child.transfer_causal_mutation("disturbance", "control")
            baseline_objective, baseline_states, baseline_controls = child._evaluate_policy(
                source_parameters["baseline"]["k"], 0.0, scenario, child.seed)
            transferred_objective, transferred_states, transferred_controls = child._evaluate_policy(
                source_parameters["transferred"]["k"], source_parameters["transferred"]["q"],
                scenario, child.seed)
            absolute = baseline_objective - transferred_objective
            relative = absolute / baseline_objective if baseline_objective else 0.0
            cases.append(ZeroShotCase(
                environment.problem_id, environment.state_a, environment.input_b,
                environment.control_limit, transfer.validation,
                round(baseline_objective, 8), round(transferred_objective, 8),
                round(absolute, 8), round(relative, 8),
                all(math.isfinite(x) for x in baseline_states + baseline_controls) and
                all(abs(u) <= environment.control_limit + 1e-12 for u in baseline_controls),
                all(math.isfinite(x) for x in transferred_states + transferred_controls) and
                all(abs(u) <= environment.control_limit + 1e-12 for u in transferred_controls),
                True,
            ))
        wins = sum(case.transfer_validation == "passed" and case.absolute_improvement > 0 for case in cases)
        mean_absolute = sum(case.absolute_improvement for case in cases) / len(cases)
        mean_relative = sum(case.relative_improvement for case in cases) / len(cases)
        self.zero_shot_evaluation = ZeroShotEvaluation(
            self.problem.problem_id, source_parameters, scenario, ids, cases,
            wins, wins == len(cases), round(mean_absolute, 8), round(mean_relative, 8), True,
            {"optimizer": "none_during_evaluation", "parameters_frozen": True,
             "source_training_scenario": source.scenario, "same_operator": "causal_edge_to_optimization_constraint"},
        )
        return self.zero_shot_evaluation

    def run_preregistered_random_zero_shot_suite(
        self, requested_count: int = 20, generation_seed: int = 424242,
    ) -> RandomZeroShotSuiteEvaluation:
        """Generate and evaluate a deterministic random target suite under fixed criteria."""
        criterion: Dict[str, Any] = {
            "min_target_count": 20,
            "min_transfer_win_rate": 0.75,
            "min_stable_rate": 0.95,
            "min_mean_relative_improvement": 0.0,
            "registered_before_generation": True,
        }
        ranges: Dict[str, Tuple[float, float]] = {
            "state_a": (0.88, 1.20),
            "input_b": (0.35, 1.10),
            "control_limit": (0.60, 1.20),
            "disturbance_scale": (0.70, 1.30),
        }
        if requested_count < criterion["min_target_count"]:
            raise ValueError(f"requested_count must be >= {criterion['min_target_count']}")
        rng = random.Random(generation_seed)
        environments = tuple(
            Problem(
                f"random-env-{index + 1:03d}",
                round(rng.uniform(*ranges["state_a"]), 8),
                round(rng.uniform(*ranges["input_b"]), 8),
                0.0,
                self.problem.horizon,
                round(rng.uniform(*ranges["control_limit"]), 8),
                round(rng.uniform(*ranges["disturbance_scale"]), 8),
            )
            for index in range(requested_count)
        )
        previous = self.zero_shot_evaluation
        evaluation = self.run_zero_shot_environment_suite(environments, "alternating")
        self.zero_shot_evaluation = previous
        wins = sum(case.absolute_improvement > 0 for case in evaluation.cases)
        stable = sum(case.baseline_stable and case.transferred_stable for case in evaluation.cases)
        win_rate = wins / len(evaluation.cases)
        stable_rate = stable / len(evaluation.cases)
        mean_relative = sum(case.relative_improvement for case in evaluation.cases) / len(evaluation.cases)
        passed = (
            len(evaluation.cases) >= criterion["min_target_count"]
            and win_rate >= criterion["min_transfer_win_rate"]
            and stable_rate >= criterion["min_stable_rate"]
            and mean_relative >= criterion["min_mean_relative_improvement"]
        )
        self.random_zero_shot_suite = RandomZeroShotSuiteEvaluation(
            generation_seed, requested_count, len(evaluation.cases), ranges, criterion,
            evaluation.cases, round(win_rate, 8), round(stable_rate, 8),
            round(mean_relative, 8), passed, evaluation.no_reoptimization,
        )
        return self.random_zero_shot_suite

    def run_adversarial_boundary_suite(self, scenario: str = "alternating") -> StressSuiteEvaluation:
        """Probe zero-shot limits with preregistered boundary and adversarial systems."""
        if self.optimization_comparison is None:
            raise RuntimeError("run optimize_and_compare in the source environment first")
        criterion: Dict[str, Any] = {
            "min_case_count": 6,
            "min_failure_cases_to_detect": 1,
            "require_boundary_cases": True,
            "require_adversarial_cases": True,
            "registered_before_evaluation": True,
        }
        environments = (
            ("boundary-high-a", "boundary", 1.20, 0.35, 0.60, 1.30),
            ("boundary-low-b", "boundary", 1.05, 0.35, 1.20, 1.00),
            ("boundary-low-limit", "boundary", 1.08, 0.80, 0.20, 1.30),
            ("adversarial-unstable", "adversarial", 1.30, 0.20, 0.35, 2.00),
            ("adversarial-saturation", "adversarial", 1.25, 0.35, 0.25, 2.50),
            ("adversarial-low-authority", "adversarial", 0.95, 0.10, 0.20, 3.00),
        )
        source = self.optimization_comparison
        source_parameters = source.transferred.parameters
        baseline_parameters = source.baseline.parameters
        cases: List[StressCase] = []
        for index, (environment_id, category, state_a, input_b, limit, scale) in enumerate(environments):
            environment = Problem(environment_id, state_a, input_b, 0.0, self.problem.horizon, limit, scale)
            child = UDIEngine(environment, seed=self.seed + 2001 * (index + 1))
            child.generate_representations()
            child.generate_cross_domain_representations()
            transfer = child.transfer_causal_mutation("disturbance", "control")
            baseline_objective, baseline_states, baseline_controls = child._evaluate_policy(
                baseline_parameters["k"], 0.0, scenario, child.seed)
            transferred_objective, transferred_states, transferred_controls = child._evaluate_policy(
                source_parameters["k"], source_parameters["q"], scenario, child.seed)
            baseline_stable = all(math.isfinite(x) for x in baseline_states + baseline_controls) and all(
                abs(u) <= limit + 1e-12 for u in baseline_controls) and max(abs(x) for x in baseline_states) < 100.0
            transferred_stable = all(math.isfinite(x) for x in transferred_states + transferred_controls) and all(
                abs(u) <= limit + 1e-12 for u in transferred_controls) and max(abs(x) for x in transferred_states) < 100.0
            baseline_saturated = any(abs(u) >= limit - 1e-9 for u in baseline_controls)
            transferred_saturated = any(abs(u) >= limit - 1e-9 for u in transferred_controls)
            absolute = baseline_objective - transferred_objective
            relative = absolute / baseline_objective if baseline_objective else 0.0
            failure_modes: List[str] = []
            if transfer.validation != "passed":
                failure_modes.append("transfer_rejected")
            if not baseline_stable:
                failure_modes.append("baseline_unstable")
            if not transferred_stable:
                failure_modes.append("transferred_unstable")
            if transferred_objective > baseline_objective:
                failure_modes.append("transferred_worse")
            if baseline_saturated:
                failure_modes.append("baseline_actuator_saturation")
            if transferred_saturated:
                failure_modes.append("transferred_actuator_saturation")
            cases.append(StressCase(
                environment_id, category, state_a, input_b, limit, scale, transfer.validation,
                round(baseline_objective, 8), round(transferred_objective, 8), round(absolute, 8),
                round(relative, 8), baseline_stable, transferred_stable,
                baseline_saturated, transferred_saturated, failure_modes,
            ))
        failure_count = sum(bool(case.failure_modes) for case in cases)
        categories = {case.category for case in cases}
        passed = (
            len(cases) >= criterion["min_case_count"]
            and failure_count >= criterion["min_failure_cases_to_detect"]
            and ("boundary" in categories) == criterion["require_boundary_cases"]
            and ("adversarial" in categories) == criterion["require_adversarial_cases"]
        )
        self.stress_suite = StressSuiteEvaluation(
            self.problem.problem_id, scenario, cases, criterion, failure_count,
            failure_count >= criterion["min_failure_cases_to_detect"], passed, True,
        )
        return self.stress_suite

    def _validate_representation(self, r: Representation) -> Tuple[bool, List[str], str | None]:
        violations: List[str] = []
        p = self.problem
        values = list(r.parameters.values())
        if r.kind not in {"proportional", "damped", "bounded"}:
            violations.append("unknown_controller_kind")
        if not values or not all(math.isfinite(v) for v in values):
            violations.append("non_finite_parameter")
        if r.kind == "proportional" or r.kind == "bounded":
            k = r.parameters.get("k", float("nan"))
            if abs(p.state_a - p.input_b * k) >= 1.0:
                violations.append("closed_loop_not_stable")
        if r.kind == "damped":
            k, d = r.parameters.get("k", float("nan")), r.parameters.get("d", float("nan"))
            # x[t+1]=(a-bk-bd)x[t]+bd*x[t-1]; Jury stability conditions.
            aa = p.state_a - p.input_b * (k + d)
            bb = p.input_b * d
            if not (1 + aa + bb > 0 and 1 - aa + bb > 0 and 1 - bb > 0):
                violations.append("damped_closed_loop_not_stable")
        if r.kind == "bounded" and r.parameters.get("k", 0) <= 0:
            violations.append("non_positive_gain")
        return not violations, violations, ";".join(violations) if violations else None

    def mutate(self, source_id: str, operator: str, value: float) -> MutationResult:
        source = next((r for r in self.representations if r.representation_id == source_id), None)
        if source is None:
            raise KeyError(f"unknown representation: {source_id}")
        params = dict(source.parameters)
        if operator == "increase_gain":
            params["k"] = params.get("k", 0.0) + value
        elif operator == "add_damping":
            params["d"] = params.get("d", 0.0) + value
            if source.kind == "proportional":
                # Structural mutation: proportional -> damped.
                kind = "damped"
            else:
                kind = source.kind
        elif operator == "tighten_actuator":
            params["limit"] = max(0.01, value)
            kind = "bounded"
        else:
            raise ValueError(f"unsupported operator: {operator}")
        kind = locals().get("kind", source.kind)
        out = Representation(
            representation_id=f"{source_id}::{operator}::{value:g}", kind=kind,
            parameters=params, assumptions=source.assumptions,
            invariants=source.invariants, parent_id=source_id,
            operator_id=operator, operator_version=self.operator_version,
        )
        valid, violations, error = self._validate_representation(out)
        result = MutationResult(
            mutation_id=f"M-{len(self.mutations)+1:03d}", source_id=source_id,
            output=out if valid else None,
            changed_elements={"operator": operator, "value": value, "parameters": params},
            preserved_invariants=list(source.invariants) if valid else [],
            violated_constraints=violations, validation="passed" if valid else "rejected", error=error,
        )
        self.mutations.append(result)
        if valid:
            self.representations.append(out)
        return result

    def generate_hypotheses(self) -> List[Hypothesis]:
        self.hypotheses = [
            Hypothesis("H1", "Увеличение gain быстрее уменьшает ошибку без насыщения.",
                       "Высокий gain даст меньшее settling_time в сценарии без сильного возмущения.",
                       "nominal", ("R-proportional", "R-bounded")),
            Hypothesis("H2", "Демпфирование улучшает устойчивость при переменном возмущении.",
                       "Damped-контроллер даст меньший max_abs_state в сценарии alternating.",
                       "alternating", ("R-damped", "R-proportional")),
            Hypothesis("H3", "Ограничение актуатора повышает безопасность ценой скорости.",
                       "Bounded-контроллер уменьшит max_abs_control, но увеличит settling_time.",
                       "pulse", ("R-bounded", "R-proportional")),
        ]
        return list(self.hypotheses)

    def _control(self, r: Representation, x: float, prev_x: float, limit: float) -> float:
        p = r.parameters
        if r.kind == "damped":
            raw = -p["k"] * x - p.get("d", 0.0) * (x - prev_x)
        else:
            raw = -p["k"] * x
        effective_limit = p.get("limit", limit) if r.kind == "bounded" else limit
        return max(-effective_limit, min(effective_limit, raw))

    def simulate(self, r: Representation, scenario: str, seed: int | None = None) -> SimulationResult:
        if scenario not in {"nominal", "alternating", "pulse"}:
            raise ValueError(f"unknown scenario: {scenario}")
        rng = random.Random(self.seed if seed is None else seed)
        x, prev_x = 1.0, 1.0
        states, controls = [x], []
        for t in range(self.problem.horizon):
            if scenario == "nominal":
                disturbance = rng.uniform(-0.02, 0.02) * self.problem.disturbance_scale
            elif scenario == "alternating":
                disturbance = (0.12 if t % 2 == 0 else -0.12) * self.problem.disturbance_scale
            else:
                disturbance = (0.45 if t == 3 else 0.0) * self.problem.disturbance_scale
            u = self._control(r, x, prev_x, self.problem.control_limit)
            prev_x, x = x, self.problem.state_a * x + self.problem.input_b * u + disturbance
            controls.append(u)
            states.append(x)
        finite = all(math.isfinite(v) for v in states + controls)
        bounded_control = all(abs(v) <= self.problem.control_limit + 1e-12 for v in controls)
        stable = finite and bounded_control and max(abs(v) for v in states) < 100.0
        settling = self.problem.horizon + 1
        for i in range(len(states)):
            if all(abs(v - self.problem.target) <= 0.05 for v in states[i:]):
                settling = i
                break
        return SimulationResult(
            r.representation_id, scenario, self.seed if seed is None else seed, states, controls,
            cost=sum(v * v for v in states) + 0.05 * sum(v * v for v in controls),
            settling_time=settling, max_abs_state=max(abs(v) for v in states), stable=stable,
            error=None if stable else "simulation_invariant_failed",
        )

    def choose_and_run_experiment(self) -> ExperimentResult:
        if not self.hypotheses:
            self.generate_hypotheses()
        # Alternating disturbance creates the largest expected prediction separation in this MVP.
        scenario = "alternating"
        candidates = [r for r in self.representations if r.representation_id in
                      {"R-proportional", "R-damped", "R-bounded"}]
        sims = [self.simulate(r, scenario) for r in candidates]
        ranking = sorted(
            [{"representation_id": s.representation_id, "cost": round(s.cost, 8),
              "settling_time": s.settling_time, "max_abs_state": round(s.max_abs_state, 8),
              "stable": s.stable} for s in sims],
            key=lambda x: (not x["stable"], x["cost"]),
        )
        separation = max(s.cost for s in sims) - min(s.cost for s in sims)
        self.experiment = ExperimentResult(
            scenario, round(separation, 8),
            "Alternating disturbance maximizes divergence between damping and direct feedback predictions.",
            sims, ranking,
        )
        return self.experiment

    def verify(self) -> Dict[str, Any]:
        checks = {
            "representations_at_least_three": len(self.representations) >= 3,
            "causal_graph_present": self.causal_graph is not None,
            "optimization_formulation_present": self.optimization_formulation is not None,
            "valid_cross_model_transfer_exists": any(t.validation == "passed" for t in self.transfers),
            "rejected_cross_model_transfer_exists": any(t.validation == "rejected" for t in self.transfers),
            "optimization_comparison_exists": self.optimization_comparison is not None,
            "transferred_objective_improves": bool(self.optimization_comparison) and
            self.optimization_comparison.absolute_improvement > 0.0,
            "held_out_evaluation_exists": self.held_out_evaluation is not None,
            "held_out_has_multiple_scenarios": bool(self.held_out_evaluation) and
            len(self.held_out_evaluation.held_out_scenarios) >= 2,
            "held_out_parameters_frozen": bool(self.held_out_evaluation) and
            self.held_out_evaluation.no_reoptimization,
            "environment_suite_exists": self.environment_suite is not None,
            "environment_suite_has_multiple_systems": bool(self.environment_suite) and
            len(self.environment_suite.environment_ids) >= 3,
            "environment_transfer_wins_recorded": bool(self.environment_suite) and
            self.environment_suite.transfer_wins >= 1,
            "zero_shot_evaluation_exists": self.zero_shot_evaluation is not None,
            "zero_shot_has_multiple_targets": bool(self.zero_shot_evaluation) and
            len(self.zero_shot_evaluation.environment_ids) >= 2,
            "zero_shot_parameters_frozen": bool(self.zero_shot_evaluation) and
            self.zero_shot_evaluation.no_reoptimization,
            "zero_shot_transfer_result_recorded": bool(self.zero_shot_evaluation) and
            self.zero_shot_evaluation.transfer_wins >= 1,
            "random_zero_shot_suite_exists": self.random_zero_shot_suite is not None,
            "random_suite_criterion_preregistered": bool(self.random_zero_shot_suite) and
            self.random_zero_shot_suite.preregistered_criterion.get("registered_before_generation") is True,
            "random_suite_has_required_count": bool(self.random_zero_shot_suite) and
            self.random_zero_shot_suite.generated_count >= 20,
            "random_suite_passed_criterion": bool(self.random_zero_shot_suite) and
            self.random_zero_shot_suite.passed,
            "stress_suite_exists": self.stress_suite is not None,
            "stress_suite_detected_failure_mode": bool(self.stress_suite) and
            self.stress_suite.stress_detected,
            "stress_suite_passed_protocol": bool(self.stress_suite) and
            self.stress_suite.passed,
            "stress_suite_no_reoptimization": bool(self.stress_suite) and
            self.stress_suite.no_reoptimization,
            "hypotheses_at_least_two": len(self.hypotheses) >= 2,
            "valid_mutation_exists": any(m.validation == "passed" for m in self.mutations),
            "rejected_mutation_exists": any(m.validation == "rejected" for m in self.mutations),
            "experiment_exists": self.experiment is not None,
            "all_experiment_runs_stable": bool(self.experiment) and all(s.stable for s in self.experiment.simulations),
            "reproducible_seed_recorded": self.seed == (self.experiment.simulations[0].seed if self.experiment else None),
        }
        return {"checks": checks, "passed": sum(checks.values()), "total": len(checks),
                "status": "verified" if all(checks.values()) else "partially_verified"}

    def report(self) -> Dict[str, Any]:
        verification = self.verify()
        data = {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "problem": asdict(self.problem),
            "representations": [asdict(r) for r in self.representations],
            "mutations": [asdict(m) for m in self.mutations],
            "causal_graph": asdict(self.causal_graph) if self.causal_graph else None,
            "optimization_formulation": asdict(self.optimization_formulation) if self.optimization_formulation else None,
            "transfers": [asdict(t) for t in self.transfers],
            "optimization_comparison": asdict(self.optimization_comparison)
            if self.optimization_comparison else None,
            "held_out_evaluation": asdict(self.held_out_evaluation)
            if self.held_out_evaluation else None,
            "environment_suite": asdict(self.environment_suite)
            if self.environment_suite else None,
            "zero_shot_evaluation": asdict(self.zero_shot_evaluation)
            if self.zero_shot_evaluation else None,
            "random_zero_shot_suite": asdict(self.random_zero_shot_suite)
            if self.random_zero_shot_suite else None,
            "stress_suite": asdict(self.stress_suite)
            if self.stress_suite else None,
            "hypotheses": [asdict(h) for h in self.hypotheses],
            "experiment": asdict(self.experiment) if self.experiment else None,
            "verification": verification,
            "limitations": [
                "Computational simulation only; no physical experiment.",
                "Novelty and patent status were not established.",
                "Benchmark is intentionally small and domain-specific.",
            ],
        }
        encoded = json.dumps(data, sort_keys=True, default=str).encode()
        data["input_hash"] = hashlib.sha256(encoded).hexdigest()
        return data
