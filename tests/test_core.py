import unittest

from udi_mvp.core import Problem, UDIEngine
from udi_mvp.reporting import generate_phase9_report, markdown_phase9_report


class UDIMVPTests(unittest.TestCase):
    def make_engine(self, seed=7):
        p = Problem("test", 1.08, 0.8, 0.0, 30, 1.0, 1.0)
        e = UDIEngine(p, seed=seed)
        e.generate_representations()
        return e

    def test_three_representations_and_real_mutations(self):
        e = self.make_engine()
        self.assertEqual(len(e.representations), 3)
        valid = e.mutate("R-proportional", "increase_gain", 0.08)
        rejected = e.mutate("R-proportional", "increase_gain", 3.0)
        self.assertEqual(valid.validation, "passed")
        self.assertIsNotNone(valid.output)
        self.assertEqual(rejected.validation, "rejected")
        self.assertIsNone(rejected.output)
        self.assertIn("closed_loop_not_stable", rejected.violated_constraints)

    def test_invalid_operator_and_input_are_not_hidden(self):
        e = self.make_engine()
        with self.assertRaises(ValueError):
            e.mutate("R-proportional", "invent_physics", 1.0)
        with self.assertRaises(KeyError):
            e.mutate("missing", "increase_gain", 1.0)

    def test_hypotheses_experiment_and_verification(self):
        e = self.make_engine()
        e.generate_cross_domain_representations()
        e.mutate("R-proportional", "increase_gain", 0.08)
        e.mutate("R-proportional", "increase_gain", 3.0)
        e.generate_hypotheses()
        e.transfer_causal_mutation("disturbance", "control")
        e.transfer_causal_mutation("next_state", "state")
        e.choose_and_run_experiment()
        e.optimize_and_compare("alternating")
        e.run_held_out_evaluation("alternating", ("nominal", "pulse", "ramp"))
        environments = (
            Problem("env-fast", 1.18, 0.90, 0.0, 40, 1.0, 1.0),
            Problem("env-damped", 0.94, 0.65, 0.0, 40, 0.8, 1.0),
            Problem("env-weak-actuator", 1.08, 0.45, 0.0, 40, 0.7, 1.0),
        )
        e.run_environment_suite(environments)
        e.run_zero_shot_environment_suite(environments)
        e.run_preregistered_random_zero_shot_suite(requested_count=20, generation_seed=424242)
        e.run_adversarial_boundary_suite("alternating")
        self.assertGreaterEqual(len(generate_phase9_report(e.report())["recommendations"]), 1)
        report = e.report()
        self.assertGreaterEqual(len(e.hypotheses), 2)
        self.assertEqual(report["verification"]["status"], "verified")
        self.assertEqual(report["verification"]["passed"], report["verification"]["total"])

    def test_reproducibility(self):
        def run():
            e = self.make_engine(seed=99)
            e.generate_hypotheses()
            e.choose_and_run_experiment()
            return e.report()["experiment"]
        self.assertEqual(run(), run())

    def test_unknown_scenario_is_rejected(self):
        e = self.make_engine()
        with self.assertRaises(ValueError):
            e.simulate(e.representations[0], "unbounded_real_world")

    def test_phase_1_contracts_create_distinct_models(self):
        e = self.make_engine()
        models = e.generate_cross_domain_representations()
        graph = models["causal_graph"]
        optimization = models["optimization"]
        self.assertEqual(graph.graph_id, "CG-control-001")
        self.assertIn(("state", "control"), graph.edges)
        self.assertIn("objective_defined", optimization.invariants)
        self.assertIn("abs(control) <= 1.0", optimization.constraints)
        self.assertTrue(e._is_acyclic(graph.nodes, graph.edges))

    def test_phase_2_transfer_is_valid_and_preserves_contract(self):
        e = self.make_engine()
        e.generate_cross_domain_representations()
        valid = e.transfer_causal_mutation("disturbance", "control")
        self.assertEqual(valid.validation, "passed")
        self.assertIn("acyclic", valid.preserved_invariants)
        self.assertIn("control policy may depend on disturbance_t",
                      valid.output_model["optimization"]["constraints"])

    def test_phase_2_cycle_transfer_is_rejected_and_not_promoted(self):
        e = self.make_engine()
        e.generate_cross_domain_representations()
        rejected = e.transfer_causal_mutation("next_state", "state")
        self.assertEqual(rejected.validation, "rejected")
        self.assertIn("causal_cycle", rejected.violated_constraints)
        self.assertIsNone(rejected.output_model)

    def test_transfer_requires_phase_1_models(self):
        e = self.make_engine()
        with self.assertRaises(RuntimeError):
            e.transfer_causal_mutation("disturbance", "control")

    def test_numeric_optimizer_improves_objective_after_transfer(self):
        e = self.make_engine(seed=2026)
        e.generate_cross_domain_representations()
        e.transfer_causal_mutation("disturbance", "control")
        comparison = e.optimize_and_compare("alternating")
        self.assertGreater(comparison.absolute_improvement, 0.0)
        self.assertGreater(comparison.relative_improvement, 0.0)
        self.assertTrue(comparison.baseline.converged)
        self.assertTrue(comparison.transferred.converged)
        self.assertTrue(all(abs(u) <= e.problem.control_limit + 1e-12
                             for u in comparison.transferred.controls))

    def test_numeric_optimizer_requires_accepted_transfer(self):
        e = self.make_engine()
        e.generate_cross_domain_representations()
        with self.assertRaises(RuntimeError):
            e.solve_numeric_optimization(transferred=True)

    def test_numeric_optimizer_is_reproducible(self):
        def run():
            e = self.make_engine(seed=2026)
            e.generate_cross_domain_representations()
            e.transfer_causal_mutation("disturbance", "control")
            return e.optimize_and_compare("alternating")
        self.assertEqual(run().__dict__, run().__dict__)

    def test_held_out_evaluation_uses_frozen_parameters(self):
        e = self.make_engine(seed=2026)
        e.generate_cross_domain_representations()
        e.transfer_causal_mutation("disturbance", "control")
        e.optimize_and_compare("alternating")
        evaluation = e.run_held_out_evaluation("alternating", ("nominal", "pulse", "ramp"))
        self.assertEqual(evaluation.held_out_scenarios, ("nominal", "pulse", "ramp"))
        self.assertEqual(len(evaluation.cases), 3)
        self.assertTrue(evaluation.no_reoptimization)
        self.assertTrue(all(case.baseline_stable and case.transferred_stable
                            for case in evaluation.cases))
        self.assertEqual(evaluation.cases[0].seed, 2037)

    def test_held_out_protocol_rejects_training_overlap_and_duplicates(self):
        e = self.make_engine()
        e.generate_cross_domain_representations()
        e.transfer_causal_mutation("disturbance", "control")
        e.optimize_and_compare("alternating")
        with self.assertRaises(ValueError):
            e.run_held_out_evaluation("alternating", ("alternating", "pulse"))
        with self.assertRaises(ValueError):
            e.run_held_out_evaluation("alternating", ("pulse", "pulse"))

    def test_held_out_evaluation_is_reproducible(self):
        def run():
            e = self.make_engine(seed=2026)
            e.generate_cross_domain_representations()
            e.transfer_causal_mutation("disturbance", "control")
            e.optimize_and_compare("alternating")
            return e.run_held_out_evaluation().__dict__
        self.assertEqual(run(), run())

    def test_phase_5_transfer_survives_changed_environments(self):
        e = self.make_engine(seed=2026)
        suite = e.run_environment_suite((
            Problem("fast", 1.18, 0.90, 0.0, 30, 1.0, 1.0),
            Problem("damped", 0.94, 0.65, 0.0, 30, 0.8, 1.0),
            Problem("weak", 1.08, 0.45, 0.0, 30, 0.7, 1.0),
        ))
        self.assertEqual(len(suite.cases), 3)
        self.assertEqual(suite.transfer_wins, 3)
        self.assertTrue(suite.transfer_wins_all)
        self.assertTrue(all(case.transfer_validation == "passed" for case in suite.cases))
        self.assertTrue(all(case.baseline_stable and case.transferred_stable for case in suite.cases))
        self.assertTrue(suite.recalibrated_per_environment)

    def test_phase_5_rejects_duplicate_environment_ids(self):
        e = self.make_engine()
        with self.assertRaises(ValueError):
            e.run_environment_suite((
                Problem("duplicate", 1.0, 0.8, 0.0, 20, 1.0, 1.0),
                Problem("duplicate", 1.1, 0.7, 0.0, 20, 1.0, 1.0),
            ))

    def test_phase_5_is_reproducible(self):
        def run():
            e = self.make_engine(seed=2026)
            return e.run_environment_suite((
                Problem("fast", 1.18, 0.90, 0.0, 30, 1.0, 1.0),
                Problem("weak", 1.08, 0.45, 0.0, 30, 0.7, 1.0),
            )).__dict__
        self.assertEqual(run(), run())

    def test_phase_6_zero_shot_uses_source_parameters_without_reoptimization(self):
        e = self.make_engine(seed=2026)
        e.generate_cross_domain_representations()
        e.transfer_causal_mutation("disturbance", "control")
        source = e.optimize_and_compare("alternating")
        targets = (
            Problem("zero-fast", 1.18, 0.90, 0.0, 30, 1.0, 1.0),
            Problem("zero-weak", 1.08, 0.45, 0.0, 30, 0.7, 1.0),
        )
        result = e.run_zero_shot_environment_suite(targets)
        self.assertEqual(result.source_parameters["transferred"], source.transferred.parameters)
        self.assertEqual(len(result.cases), 2)
        self.assertTrue(result.no_reoptimization)
        self.assertTrue(all(case.parameters_frozen for case in result.cases))
        self.assertTrue(all(case.transfer_validation == "passed" for case in result.cases))
        self.assertTrue(all(case.baseline_stable and case.transferred_stable for case in result.cases))
        self.assertGreaterEqual(result.transfer_wins, 1)

    def test_phase_6_requires_source_optimization_and_rejects_source_target(self):
        e = self.make_engine()
        target = (Problem("target", 1.1, 0.8, 0.0, 20, 1.0, 1.0),)
        with self.assertRaises(RuntimeError):
            e.run_zero_shot_environment_suite(target)
        e.generate_cross_domain_representations()
        e.transfer_causal_mutation("disturbance", "control")
        e.optimize_and_compare("alternating")
        with self.assertRaises(ValueError):
            e.run_zero_shot_environment_suite((e.problem,))

    def test_phase_6_zero_shot_is_reproducible(self):
        def run():
            e = self.make_engine(seed=2026)
            e.generate_cross_domain_representations()
            e.transfer_causal_mutation("disturbance", "control")
            e.optimize_and_compare("alternating")
            return e.run_zero_shot_environment_suite((
                Problem("fast", 1.18, 0.90, 0.0, 30, 1.0, 1.0),
                Problem("weak", 1.08, 0.45, 0.0, 30, 0.7, 1.0),
            )).__dict__
        self.assertEqual(run(), run())

    def test_phase_7_random_suite_has_preregistered_success_criterion(self):
        e = self.make_engine(seed=2026)
        e.generate_cross_domain_representations()
        e.transfer_causal_mutation("disturbance", "control")
        e.optimize_and_compare("alternating")
        suite = e.run_preregistered_random_zero_shot_suite(20, 424242)
        self.assertEqual(suite.generated_count, 20)
        self.assertTrue(suite.preregistered_criterion["registered_before_generation"])
        self.assertEqual(suite.preregistered_criterion["min_target_count"], 20)
        self.assertGreaterEqual(suite.transfer_win_rate, suite.preregistered_criterion["min_transfer_win_rate"])
        self.assertGreaterEqual(suite.stable_rate, suite.preregistered_criterion["min_stable_rate"])
        self.assertTrue(suite.no_reoptimization)
        self.assertTrue(suite.passed)

    def test_phase_7_rejects_underpowered_suite_before_generation(self):
        e = self.make_engine()
        e.generate_cross_domain_representations()
        e.transfer_causal_mutation("disturbance", "control")
        e.optimize_and_compare("alternating")
        with self.assertRaises(ValueError):
            e.run_preregistered_random_zero_shot_suite(19, 424242)

    def test_phase_7_random_suite_is_reproducible(self):
        def run():
            e = self.make_engine(seed=2026)
            e.generate_cross_domain_representations()
            e.transfer_causal_mutation("disturbance", "control")
            e.optimize_and_compare("alternating")
            return e.run_preregistered_random_zero_shot_suite(20, 424242).__dict__
        self.assertEqual(run(), run())

    def test_phase_8_detects_zero_shot_failure_modes(self):
        e = self.make_engine(seed=2026)
        e.generate_cross_domain_representations()
        e.transfer_causal_mutation("disturbance", "control")
        e.optimize_and_compare("alternating")
        stress = e.run_adversarial_boundary_suite()
        self.assertEqual(len(stress.cases), 6)
        self.assertTrue(stress.stress_detected)
        self.assertTrue(stress.passed)
        self.assertTrue(any(case.failure_modes for case in stress.cases))
        self.assertTrue(all(case.transfer_validation == "passed" for case in stress.cases))
        self.assertTrue(stress.no_reoptimization)

    def test_phase_8_has_both_preregistered_categories(self):
        e = self.make_engine()
        e.generate_cross_domain_representations()
        e.transfer_causal_mutation("disturbance", "control")
        e.optimize_and_compare("alternating")
        stress = e.run_adversarial_boundary_suite()
        self.assertEqual({case.category for case in stress.cases}, {"boundary", "adversarial"})
        self.assertTrue(stress.preregistered_criterion["registered_before_evaluation"])

    def test_phase_8_requires_source_optimization(self):
        e = self.make_engine()
        with self.assertRaises(RuntimeError):
            e.run_adversarial_boundary_suite()

    def test_phase_9_synthesizes_stress_based_recommendations(self):
        e = self.make_engine(seed=2026)
        e.generate_cross_domain_representations()
        e.mutate("R-proportional", "increase_gain", 0.08)
        e.mutate("R-proportional", "increase_gain", 3.0)
        e.transfer_causal_mutation("disturbance", "control")
        e.transfer_causal_mutation("next_state", "state")
        e.generate_hypotheses()
        e.choose_and_run_experiment()
        e.optimize_and_compare("alternating")
        e.run_held_out_evaluation("alternating", ("nominal", "pulse", "ramp"))
        environments = (
            Problem("phase9-fast", 1.18, 0.90, 0.0, 40, 1.0, 1.0),
            Problem("phase9-damped", 0.94, 0.65, 0.0, 40, 0.8, 1.0),
            Problem("phase9-weak", 1.08, 0.45, 0.0, 40, 0.7, 1.0),
        )
        e.run_environment_suite(environments)
        e.run_zero_shot_environment_suite(environments)
        e.run_preregistered_random_zero_shot_suite(20, 424242)
        e.run_adversarial_boundary_suite()
        report = generate_phase9_report(e.report())
        self.assertEqual(report["status"], "verified")
        self.assertGreaterEqual(report["executive_summary"]["failure_modes"].get("transferred_unstable", 0), 1)
        self.assertTrue(any(item["priority"] == "critical" for item in report["recommendations"]))
        self.assertIn("Verification experiment", markdown_phase9_report(report))

    def test_phase_9_is_deterministic_for_same_input(self):
        e = self.make_engine(seed=2026)
        e.generate_cross_domain_representations()
        e.transfer_causal_mutation("disturbance", "control")
        e.generate_hypotheses()
        e.choose_and_run_experiment()
        e.optimize_and_compare("alternating")
        e.run_held_out_evaluation("alternating", ("nominal", "pulse", "ramp"))
        e.run_preregistered_random_zero_shot_suite(20, 424242)
        e.run_adversarial_boundary_suite()
        source = e.report()
        self.assertEqual(generate_phase9_report(source), generate_phase9_report(source))


if __name__ == "__main__":
    unittest.main()
