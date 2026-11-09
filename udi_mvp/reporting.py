"""Deterministic Phase 9 reporting and resilience recommendation synthesis."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List


def _recommendation(priority: str, title: str, evidence: str, action: str, verification: str) -> Dict[str, str]:
    return {
        "priority": priority,
        "title": title,
        "evidence": evidence,
        "action": action,
        "verification_experiment": verification,
    }


def synthesize_recommendations(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Create recommendations only from recorded results; no LLM or hidden judgment."""
    recommendations: List[Dict[str, str]] = []
    stress = data.get("stress_suite") or {}
    stress_cases = stress.get("cases", [])
    failure_modes = Counter(mode for case in stress_cases for mode in case.get("failure_modes", []))
    unstable = failure_modes.get("baseline_unstable", 0) + failure_modes.get("transferred_unstable", 0)
    saturation = failure_modes.get("baseline_actuator_saturation", 0) + failure_modes.get("transferred_actuator_saturation", 0)
    worse = failure_modes.get("transferred_worse", 0)

    if unstable:
        recommendations.append(_recommendation(
            "critical", "Добавить safe fallback и gain scheduling для неустойчивых режимов",
            f"В stress suite зафиксировано {unstable} событий instability: {dict(failure_modes)}.",
            "Перед применением transferred policy проверять оценку closed-loop pole и переключаться на conservative controller при выходе за безопасный диапазон.",
            "Сгенерировать boundary grid по state_a/input_b и проверить, что fallback удерживает max_abs_state < 100 без скрытия rejected cases.",
        ))
    if saturation:
        recommendations.append(_recommendation(
            "high", "Сделать policy actuator-aware и добавить anti-saturation режим",
            f"Зафиксировано {saturation} событий actuator saturation в {len(stress_cases)} stress cases.",
            "Добавить в objective явный penalty за saturation, ограничивать q по доступной authority и сохранять saturation ratio как отдельную метрику.",
            "Сравнить текущую policy с actuator-aware policy на том же preregistered stress suite; критерий — снижение saturation rate без роста нестабильности.",
        ))
    if worse:
        recommendations.append(_recommendation(
            "high", "Ввести автоматический отказ от переноса при отрицательном zero-shot эффекте",
            f"Transferred policy была хуже baseline в {worse} stress cases.",
            "Добавить runtime gate: не активировать transfer, если pilot simulation даёт absolute_improvement <= 0.",
            "На новых environments провести bounded pilot rollout и сравнить decision accuracy отказа с полным запуском.",
        ))

    random_suite = data.get("random_zero_shot_suite") or {}
    if random_suite.get("passed"):
        recommendations.append(_recommendation(
            "medium", "Расширить random suite за пределы текущих диапазонов",
            f"Random suite прошёл: {random_suite.get('generated_count')} systems, win rate={random_suite.get('transfer_win_rate')}, stable rate={random_suite.get('stable_rate')}.",
            "Увеличить количество систем и preregistered диапазоны в сторону меньшего input_b и меньшего control_limit, сохраняя критерий до генерации.",
            "Запустить новый независимый seed и сравнить win rate, stable rate и worst-case objective с текущим suite.",
        ))

    held_out = data.get("held_out_evaluation") or {}
    if held_out.get("mean_relative_improvement", 0) > 0:
        recommendations.append(_recommendation(
            "medium", "Добавить uncertainty-aware evaluation для held-out сценариев",
            f"Среднее held-out улучшение составило {held_out.get('mean_relative_improvement'):.2%}.",
            "Ввести интервалы по seed и повторять каждый сценарий на нескольких независимых disturbance realizations.",
            "Проверить, сохраняется ли нижняя граница доверительного интервала improvement выше нуля.",
        ))

    if not recommendations:
        recommendations.append(_recommendation(
            "high", "Расширить диагностику до failure-aware resilience suite",
            "В текущем отчёте недостаточно evidence для конкретной рекомендации.",
            "Добавить adversarial environments, saturation telemetry и preregistered acceptance thresholds.",
            "Повторить Phase 8 и сохранить результаты как отдельный reproducible artifact.",
        ))
    return recommendations


def generate_phase9_report(data: Dict[str, Any]) -> Dict[str, Any]:
    verification = data.get("verification", {})
    stress = data.get("stress_suite") or {}
    random_suite = data.get("random_zero_shot_suite") or {}
    recommendations = synthesize_recommendations(data)
    failure_modes = Counter(mode for case in stress.get("cases", []) for mode in case.get("failure_modes", []))
    return {
        "report_schema_version": "1.0.0",
        "report_type": "phase9_resilience_synthesis",
        "source_input_hash": data.get("input_hash"),
        "status": "verified" if verification.get("status") == "verified" else "partially_verified",
        "executive_summary": {
            "verification": verification.get("status", "not_run"),
            "random_suite_passed": random_suite.get("passed", False),
            "stress_protocol_passed": stress.get("passed", False),
            "failure_modes": dict(failure_modes),
            "recommendation_count": len(recommendations),
        },
        "recommendations": recommendations,
        "next_phase_experiments": [
            "Actuator-aware controller with explicit saturation penalty.",
            "Closed-loop pole guard and conservative fallback controller.",
            "Multi-seed uncertainty intervals for held-out and zero-shot suites.",
            "Independent environment family with changed disturbance dynamics.",
        ],
        "limitations": [
            "Recommendations are deterministic rule-based synthesis, not expert certification.",
            "Evidence is limited to the computational simulations recorded in the source report.",
            "No physical, safety-critical or production deployment decision is implied.",
        ],
    }


def markdown_phase9_report(report: Dict[str, Any]) -> str:
    summary = report["executive_summary"]
    lines = [
        "# Phase 9 — Resilience Synthesis Report", "",
        f"- Status: **{report['status']}**",
        f"- Source input hash: `{report.get('source_input_hash')}`",
        f"- Verification: `{summary['verification']}`",
        f"- Random suite passed: `{summary['random_suite_passed']}`",
        f"- Stress protocol passed: `{summary['stress_protocol_passed']}`",
        f"- Recommendations: `{summary['recommendation_count']}`", "",
        "## Observed failure modes", "",
    ]
    if summary["failure_modes"]:
        lines.extend(f"- `{name}`: {count}" for name, count in sorted(summary["failure_modes"].items()))
    else:
        lines.append("- None recorded.")
    lines += ["", "## Recommendations", ""]
    for index, item in enumerate(report["recommendations"], 1):
        lines += [
            f"### {index}. [{item['priority']}] {item['title']}",
            f"**Evidence:** {item['evidence']}",
            f"**Action:** {item['action']}",
            f"**Verification experiment:** {item['verification_experiment']}", "",
        ]
    lines += ["## Next experiments", ""]
    lines.extend(f"- {item}" for item in report["next_phase_experiments"])
    lines += ["", "## Limitations", ""]
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def write_phase9_report(data: Dict[str, Any], output_json: Path, output_md: Path) -> Dict[str, Any]:
    report = generate_phase9_report(data)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    output_md.write_text(markdown_phase9_report(report))
    return report
