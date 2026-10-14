"""CuongDanh team entry points.

Gateway controls outgoing tool calls. Agent is the opt-in guarded loop.
See agent/README.md for the host integration contract and local limitations.
"""

from __future__ import annotations

from agent.gateway import Command, Decision, Gateway, GatewayContext
from agent.runtime import Agent
from agent.guardrails import (
    ArithmeticCheckResult,
    AnswerSafetyResult,
    GroundingResult,
    InjectionScanResult,
    RedactionResult,
    abstention_policy,
    check_grounding,
    prepare_answer,
    redact,
    scan_for_injected_instructions,
    verify_arithmetic,
)
from agent.strategy import (
    BudgetPacer,
    ReplicaChoice,
    ResultCache,
    cheap_mask,
    disciplined_round_cost,
    careless_round_cost,
    is_catalog_trap,
    pick_replica,
    should_delegate,
    successor_of,
)
from agent.telemetry import RecordingGatewayContext, Telemetry

__all__ = [
    "Agent",
    # gateway.py
    "Command",
    "Decision",
    "Gateway",
    "GatewayContext",
    # guardrails.py
    "GroundingResult",
    "check_grounding",
    "InjectionScanResult",
    "scan_for_injected_instructions",
    "RedactionResult",
    "redact",
    "ArithmeticCheckResult",
    "verify_arithmetic",
    "abstention_policy",
    "AnswerSafetyResult",
    "prepare_answer",
    # strategy.py
    "BudgetPacer",
    "ReplicaChoice",
    "ResultCache",
    "cheap_mask",
    "disciplined_round_cost",
    "careless_round_cost",
    "is_catalog_trap",
    "pick_replica",
    "should_delegate",
    "successor_of",
    # telemetry.py
    "RecordingGatewayContext",
    "Telemetry",
]


if __name__ == "__main__":
    print("=== agent package: public surface ===")
    for name in __all__:
        print(f"  {name}")
    assert {"Command", "Decision", "Gateway", "GatewayContext"} <= set(__all__)
    assert {"BudgetPacer", "ResultCache", "should_delegate"} <= set(__all__)
    assert {"GroundingResult", "check_grounding", "abstention_policy"} <= set(__all__)
    assert {"Telemetry", "RecordingGatewayContext"} <= set(__all__)
    print(f"\n  {len(__all__)} public names, all import cleanly from `agent`.")

    print("\n=== one Gateway, built through the package's own surface ===")
    ctx = RecordingGatewayContext(
        act="learner:sv-0401", sub="agent:demo-team", scopes=frozenset({"wiki.read"}),
        credits=100, round=1, call_index=0,
    )
    gw = Gateway(ctx)
    cmd = Command(
        cmd_id="cmd:0000", kind="discover", raw="DISCOVER registry.list_servers fields=name",
        server="registry", tool="list_servers", args={}, fields=("name",), headers={},
        lease_id=None, call_index=0,
    )
    decision = gw.decide(cmd)
    print(f"  gw.decide(cmd) -> verdict={decision.verdict!r}")
    assert decision.verdict == "forward"

    print("\nagent/__init__.py import-and-export check passed.")
