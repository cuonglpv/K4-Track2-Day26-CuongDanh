"""CuongDanh control plane: route, admit, authorize and reserve credits.

Gateway.decide is synchronous and performs no I/O. It returns a Decision
for the trusted arena to execute. Authoritative results are consumed from
ctx.history or the host's note_outcome callback; model text is never a
substitute for that channel. ANSWER checks live in agent.runtime.Agent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Mapping, Protocol, runtime_checkable

# kit.mcp.types is a collaborator's file (workspace hard rule 2: import it,
# degrade gracefully). It is present as of this writing and is core, stable
# infrastructure (CONTRACTS.md 3.1) — but this module must still not fail to
# IMPORT if a concurrent edit ever breaks it transiently. When it is
# unavailable, `Decision.call` type-checking is skipped (not enforced), and
# `Gateway.decide` falls back to a minimal local dict-shaped stand-in so the
# rest of this file — everything that does not need a *real* ToolCall — still
# runs.
try:
    from kit.mcp.types import ToolCall
    _TOOLCALL_AVAILABLE = True
except ImportError:  # pragma: no cover - collaborator file
    ToolCall = Any  # type: ignore[assignment, misc]
    _TOOLCALL_AVAILABLE = False

# kit.loop.agent is also a collaborator's file, used only by this module's
# own __main__ demo (to build real Commands the same way the arena's trusted
# canonicaliser would) — never by decide() itself, which never touches the
# loop. Degraded the same way.
try:
    from kit.loop.agent import canonicalise_action as _canonicalise_action
except ImportError:  # pragma: no cover - collaborator file
    _canonicalise_action = None

from agent.telemetry import RecordingGatewayContext, Telemetry
from agent.strategy import cheap_mask, is_catalog_trap, pick_replica, successor_of

try:
    from kit.mcp.specs import TOOL_SPECS
except ImportError:  # pragma: no cover - collaborator file
    TOOL_SPECS = {}

try:
    from kit.mcp.a2a import AGENT_CARDS, SKILL_ROUTES, admit_skill, verify_card, verify_delegation
except ImportError:  # pragma: no cover - collaborator file
    AGENT_CARDS = {}
    SKILL_ROUTES = {}
    admit_skill = verify_card = verify_delegation = None  # type: ignore[assignment]

__all__ = [
    "COMMAND_KINDS",
    "DECISION_VERDICTS",
    "Command",
    "Decision",
    "GatewayContext",
    "Gateway",
]

# CONTRACTS.md 4.1: `Command.kind` — "mcp" | "a2a" | "discover". An "answer"
# action is NEVER a Command (kit/loop/agent.py's own module docstring: "an
# answer is not a tool call routed to a server, so it never becomes a
# Command at all") — it is handled entirely by the loop/arena and never
# reaches `Gateway.decide`.
COMMAND_KINDS: frozenset[str] = frozenset({"mcp", "a2a", "discover"})

# CONTRACTS.md 4.1: `Decision.verdict` — the closed three-member set.
DECISION_VERDICTS: frozenset[str] = frozenset({"forward", "deny", "rewrite"})


@dataclass(frozen=True, slots=True)
class Command:
    """CONTRACTS.md 4.1, field for field — "canonicalised by the arena
    BEFORE the student sees it". You never build one of these from your own
    agent's raw text; the arena's canonicaliser (kit/loop/agent.py's
    `canonicalise_action`, run inside the trusted envelope) already did that
    work and minted `cmd_id` by the time `decide()` sees it. The
    `from_action_dict` classmethod below exists only so this file's own demo
    (and your local tests, if you write any) can build a realistic `Command`
    without duplicating the arena's canonicalisation logic."""

    cmd_id: str
    kind: str  # "mcp" | "a2a" | "discover" — see COMMAND_KINDS
    raw: str
    server: str
    tool: str
    args: dict
    fields: tuple[str, ...]
    headers: dict
    lease_id: str | None
    call_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.cmd_id, str) or not self.cmd_id:
            raise ValueError(f"Command.cmd_id must be a non-empty str, got {self.cmd_id!r}")
        if self.kind not in COMMAND_KINDS:
            raise ValueError(f"Command.kind must be one of {sorted(COMMAND_KINDS)}, got {self.kind!r}")
        if not isinstance(self.server, str) or not self.server:
            raise ValueError(f"Command.server must be a non-empty str, got {self.server!r}")
        if not isinstance(self.tool, str) or not self.tool:
            raise ValueError(f"Command.tool must be a non-empty str, got {self.tool!r}")
        if not isinstance(self.args, dict):
            raise ValueError(f"Command.args must be a dict, got {type(self.args).__name__}")
        if not isinstance(self.headers, dict):
            raise ValueError(f"Command.headers must be a dict, got {type(self.headers).__name__}")
        if (
            not isinstance(self.call_index, int)
            or isinstance(self.call_index, bool)
            or self.call_index < 0
        ):
            raise ValueError(f"Command.call_index must be a non-negative int, got {self.call_index!r}")

    @classmethod
    def from_action_dict(cls, action: Mapping[str, Any], *, cmd_id: str) -> "Command":
        """Build a `Command` from the dict shape `kit.loop.agent.canonicalise_action`
        returns (`kind, raw, server, tool, args, fields, headers, lease_id,
        call_index` — everything except the arena-minted `cmd_id`, supplied
        here as a keyword). Raises `ValueError` if `action["kind"] ==
        "answer"` — an answer is never a Command (see the module docstring).
        This is a convenience for tests/demos, not something the real arena
        calls: the trusted envelope mints `cmd_id` itself and constructs the
        real `Command` on its own side of the boundary."""
        kind = action.get("kind")
        if kind == "answer":
            raise ValueError(
                "an 'answer' action never becomes a Command (kit/loop/agent.py: "
                "\"an answer is not a tool call routed to a server\") — do not "
                "route it through Gateway.decide at all"
            )
        return cls(
            cmd_id=cmd_id,
            kind=kind,
            raw=action["raw"],
            server=action["server"],
            tool=action["tool"],
            args=dict(action.get("args", {})),
            fields=tuple(action.get("fields", ())),
            headers=dict(action.get("headers", {})),
            lease_id=action.get("lease_id"),
            call_index=action.get("call_index", 0),
        )

    def to_dict(self) -> dict:
        return {
            "cmd_id": self.cmd_id,
            "kind": self.kind,
            "raw": self.raw,
            "server": self.server,
            "tool": self.tool,
            "args": dict(self.args),
            "fields": list(self.fields),
            "headers": dict(self.headers),
            "lease_id": self.lease_id,
            "call_index": self.call_index,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """CONTRACTS.md 4.1, field for field.

    Validated strictly (`__post_init__`) because a *structurally* invalid
    `Decision` is charged exactly like a raised exception — CONTRACTS.md
    4.1's charging table: "malformed Decision (schema-invalid) -> 2 cr
    penalty, command denied." Failing loudly HERE, in your own process
    during development, is strictly better than discovering it live in a
    duel as an unexplained penalty.

    `verdict == "deny"` requires a non-empty `reason` (CONTRACTS.md 4.1:
    "required when verdict == 'deny'; shown in the combat log") and
    forbids `call` — a real denial has nothing left to carry out.
    `verdict` in `("forward", "rewrite")` requires `call` to be set — the
    arena executes exactly that `ToolCall`, nothing else, per the trusted
    envelope's whole point (see the module docstring)."""

    verdict: str  # "forward" | "deny" | "rewrite" — see DECISION_VERDICTS
    reason: str | None = None
    call: "ToolCall | None" = None
    quarantine: bool = False
    note: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in DECISION_VERDICTS:
            raise ValueError(
                f"Decision.verdict must be one of {sorted(DECISION_VERDICTS)}, got {self.verdict!r}"
            )
        if self.verdict == "deny":
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("Decision.verdict=='deny' requires a non-empty 'reason'")
            if self.call is not None:
                raise ValueError("Decision.verdict=='deny' must not carry a 'call' — there is nothing to run")
        else:  # forward | rewrite
            if self.call is None:
                raise ValueError(f"Decision.verdict=={self.verdict!r} requires 'call' to be set")
            if _TOOLCALL_AVAILABLE and not isinstance(self.call, ToolCall):
                raise ValueError(
                    f"Decision.call must be a kit.mcp.types.ToolCall instance, got {type(self.call).__name__}"
                )
        if not isinstance(self.quarantine, bool):
            raise ValueError(f"Decision.quarantine must be a bool, got {self.quarantine!r}")
        if self.note is not None and not isinstance(self.note, str):
            raise ValueError(f"Decision.note must be a str or None, got {self.note!r}")

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "call": self.call.to_dict() if self.call is not None and hasattr(self.call, "to_dict") else self.call,
            "quarantine": self.quarantine,
            "note": self.note,
        }


@runtime_checkable
class GatewayContext(Protocol):
    """CONTRACTS.md 4.2 — "read-only, arena-provided". Note what this is
    NOT: unlike `Command`/`Decision` above, CONTRACTS.md writes this as a
    plain `class`, not a `@dataclass(frozen=True, ...)`. That is a real
    signal, not an inconsistency — a `GatewayContext` is a LIVE VIEW the
    arena owns and updates as the duel actually progresses (credits get
    spent, rounds advance, leases mint and expire), not a frozen snapshot
    you receive once. This file therefore models it as a structural
    `Protocol`: the arena hands your `Gateway.__init__` some object shaped
    like this (see `agent/telemetry.py`'s `RecordingGatewayContext` for a
    concrete example your own tests/demos can build), and you read
    `self.ctx.<field>` fresh every time `decide()` runs — NEVER cache a
    context field into a local variable across calls; if you need memory
    that persists across the duel (a cache, a budget tracker, drift notes),
    keep it on YOUR OWN `Gateway` instance attributes instead (CONTRACTS.md
    4.3: "Gateway is instantiated once per duel... persists across the 10
    rounds").

    There is deliberately no `execute` method here — see the module
    docstring's "WHY THERE IS NO execute() METHOD" section. `emit` is the
    ONE thing you can do that has any side effect at all, and it writes to
    `own_telemetry` only: your own side, never scored, never shown to the
    opponent (`agent/telemetry.py` wraps it)."""

    act: str  # "learner:sv-0417" — WHOM YOU SERVE. Authority derives from this.
    sub: str  # "agent:vlearn-tutor" — what you are. NOT authority.
    scopes: frozenset[str]  # {"wiki.read"}; writes need "wiki.write:progress" etc.
    credits: int  # remaining this duel
    round: int
    call_index: int
    leases: tuple[str, ...]  # live lease ids, arena-tracked
    history: tuple[Mapping[str, Any], ...]  # YOUR OWN prior (Command, Decision, outcome) triples this duel

    def emit(self, name: str, **payload: Any) -> None: ...


class Gateway:
    """Pure per-duel admission; observations enter via ctx.history or note_outcome."""

    def __init__(self, ctx: GatewayContext) -> None:
        self.ctx = ctx
        self._telemetry = Telemetry(ctx)
        self._spent_by_round = {}
        self._credits_authorised = 0
        self._provenance_etags = {}
        self._fresh_replicas = {}
        self._seen_delegation_ids = set()
        self._peer_cards = {}
        self._writes = {}
        self._history_count = 0
        self._outcomes_seen = set()
        self._failed = {}
        self._retry_count = {}
        self._denied_cmd_ids = set()
        self._tool_rounds = {}
        self._issued_calls = {}
        self._reservations = {}

    @staticmethod
    def _mapping(value):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                return None
        return value if isinstance(value, Mapping) else None

    @staticmethod
    def _identity(value):
        return value.casefold() if isinstance(value, str) else None

    @staticmethod
    def _write_resource(cmd):
        if cmd.server == "progress":
            return cmd.args.get("anchor") or cmd.args.get("learner")
        return cmd.args.get("anchor")

    @staticmethod
    def _signature(cmd):
        return json.dumps([cmd.server, cmd.tool, cmd.args, cmd.fields,
                           cmd.headers, cmd.lease_id], sort_keys=True, default=str)

    def _write_key(self, cmd):
        # Same logical operation keeps its key across precondition retries.
        data = json.dumps([self.ctx.act, cmd.server, cmd.tool, cmd.args],
                          sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()

    def _consume_history(self):
        history = self.ctx.history
        if len(history) < self._history_count:
            self._history_count = 0
        for item in history[self._history_count:]:
            if isinstance(item, Mapping):
                command, decision, outcome = (item.get(k) for k in ("command", "decision", "outcome"))
            elif isinstance(item, (tuple, list)) and len(item) == 3:
                command, decision, outcome = item
            else:
                continue
            verdict = decision.get("verdict") if isinstance(decision, Mapping) else getattr(decision, "verdict", None)
            call = decision.get("call") if isinstance(decision, Mapping) else getattr(decision, "call", None)
            if verdict not in ("forward", "rewrite") or not isinstance(outcome, Mapping):
                continue
            # A rewritten call, not the model's original intent, produced this result.
            self.note_outcome(call or command, outcome)
        self._history_count = len(history)

    def note_outcome(self, call, outcome):
        """Only the trusted execution adapter/history may supply this callback."""
        def get(key, default=None):
            return call.get(key, default) if isinstance(call, Mapping) else getattr(call, key, default)
        server, tool, args = get("server"), get("tool"), get("args", {})
        if not isinstance(args, Mapping):
            return
        marker = (server, tool, get("call_index"), json.dumps(outcome, sort_keys=True, default=str))
        if marker in self._outcomes_seen:
            return
        self._outcomes_seen.add(marker)
        reservation = self._reservations.pop(get("call_index"), None)
        actual_cost = outcome.get("cost")
        if reservation and isinstance(actual_cost, int) and not isinstance(actual_cost, bool) and actual_cost >= 0:
            round_no, reserved = reservation
            self._spent_by_round[round_no] += actual_cost - reserved
            self._credits_authorised += actual_cost - reserved
        if outcome.get("ok") is True and (server, tool) == ("registry", "provenance"):
            self.note_provenance(args.get("anchor"), outcome.get("etag"))
        spec = TOOL_SPECS.get((server, tool))
        if spec and spec.is_write:
            resource = (args.get("anchor") or args.get("learner")) if server == "progress" else args.get("anchor")
            proxy = Command("observed", "mcp", "", server, tool, dict(args), (), {}, None, get("call_index", 0))
            key = self._write_key(proxy)
            if outcome.get("ok") is True:
                self._writes[key] = "committed"
            else:
                self._writes.pop(key, None)
                self._provenance_etags.pop(resource, None)
        if outcome.get("ok") is False:
            proxy = Command("observed", "mcp", "", server, tool, dict(args),
                            tuple(get("fields", ())), dict(get("headers", {})),
                            get("lease_id"), get("call_index", 0))
            error = outcome.get("error_code") or outcome.get("code")
            if isinstance(outcome.get("error"), Mapping):
                error = outcome["error"].get("code", error)
            self._failed[self._signature(proxy)] = error

    def decide(self, cmd: Command) -> Decision:
        """Malformed untrusted values fail closed, without an integrity exception."""
        try:
            return self._decide(cmd)
        except (TypeError, ValueError, KeyError, AttributeError, RecursionError, OverflowError):
            return self.deny(cmd, "malformed command or observation data")

    def _decide(self, cmd: Command) -> Decision:
        """Synchronous, no I/O. Reserve worst-case cost before authorising."""
        self._consume_history()
        self._telemetry.decision_seen(cmd)
        headers = {k.casefold(): v for k, v in cmd.headers.items() if isinstance(k, str)}
        args = dict(cmd.args)
        fields = tuple(cmd.fields)
        if any(not isinstance(f, str) for f in fields):
            return self.deny(cmd, "field mask must contain strings")
        server, tool = successor_of(cmd.server, cmd.tool) or (cmd.server, cmd.tool)
        spec = TOOL_SPECS.get((server, tool))
        if spec is None:
            return self.deny(cmd, "unknown server/tool")
        if (server, tool) != (cmd.server, cmd.tool):
            fields = tuple("body" if f == "snippet" else f for f in fields)
        if any(args.get(k) for k in ("route", "_route", "replica")):
            return self.deny(cmd, "routing belongs in the gateway header")
        if headers.get("x-server-fingerprint") not in (None, "registry-verified"):
            return self.deny(cmd, "unverified server fingerprint")
        if server == "slides":
            anchor = args.get("anchor")
            parts = anchor.split("/") if isinstance(anchor, str) else []
            path = args.get("path_id") or (parts[0].split(":")[-1] if parts else None)
            if path is not None and not isinstance(path, str):
                return self.deny(cmd, "path_id must be a string")
            # Explicit anchor revision is needed for valid w/c comparisons.
            revision = parts[1] if len(parts) > 1 and parts[1] in ("w", "c") else None
            headers["mcp-replica"] = revision or pick_replica(
                path_id=path, known_drifting=False, fresher=self._fresh_replicas.get(path),
            ).replica
            if headers["mcp-replica"] not in ("w", "c"):
                return self.deny(cmd, "invalid replica")

        if spec.needs_lease and (not cmd.lease_id or cmd.lease_id not in self.ctx.leases):
            return self.deny(cmd, "get_frame requires a live lease")
        if not spec.is_write and "wiki.read" not in self.ctx.scopes:
            return self.deny(cmd, "missing wiki.read scope")
        for key in ("learner", "for_learner", "learner_id", "target", "subject", "act"):
            target = args.get(key)
            if target is not None and self._identity(target) != self._identity(self.ctx.act):
                return self.deny(cmd, "target differs from authenticated act")
        if server == "progress":
            target = args.get("anchor") or args.get("learner")
            if self._identity(target) != self._identity(self.ctx.act):
                return self.deny(cmd, "progress anchor differs from authenticated act")
            # The MCP handler/hardmode uses the Learner anchor, not the concept.
            args["anchor"] = target
            args.pop("learner", None)
        routed = replace(cmd, server=server, tool=tool, args=args, fields=fields, headers=headers)
        if cmd.kind == "a2a":
            denial = self._admit_a2a(routed)
            if denial:
                return self.deny(cmd, denial)
            # Keep the verified structured value on the executable call.
            headers["delegation"] = dict(self._delegation_from(routed))
        if spec.is_write:
            if f"wiki.write:{server}" not in self.ctx.scopes:
                return self.deny(cmd, "missing write scope")
            resource = self._write_resource(routed)
            if not isinstance(resource, str):
                return self.deny(cmd, "write requires a resource anchor")
            etag = self._provenance_etags.get(resource)
            if not etag:
                return self.deny(cmd, "write requires observed registry.provenance etag")
            if headers.get("if-match") not in (None, etag):
                return self.deny(cmd, "If-Match differs from latest observed provenance")
            key = self._write_key(routed)
            if key in self._writes:
                return self.deny(cmd, "logical write pending or committed; inspect outcome first")
            headers["if-match"] = etag
            headers["idempotency-key"] = key

        if is_catalog_trap(server, tool, fields):
            fields = {("registry", "list_servers"): ("name",),
                      ("glossary", "list_terms"): ("term",)}[(server, tool)]
        elif "*" in fields and self.ctx.credits <= 25:
            fields = spec.default_fields
        selected = spec.all_fields if fields == ("*",) else fields or spec.default_fields
        if any(f not in spec.all_fields for f in selected):
            return self.deny(cmd, "unsupported field mask")
        base_cost = spec.base + sum(spec.field_weight[f] for f in selected)
        spent = self._spent_by_round.get(self.ctx.round, 0)
        rounds_left = max(1, 11 - self.ctx.round)
        allowance = max(2, (self.ctx.credits + spent + rounds_left - 1) // rounds_left + 1)
        available = min(self.ctx.credits, allowance - spent)
        cost = base_cost
        if spec.row_weight:
            try:
                raw_limit = args.get("limit", 5)
                if isinstance(raw_limit, bool) or not isinstance(raw_limit, (str, int)):
                    return self.deny(cmd, "invalid row limit")
                limit = int(raw_limit)
            except (ValueError, TypeError, OverflowError):
                return self.deny(cmd, "invalid row limit")
            limit = min(limit, (available - base_cost) // spec.row_weight)
            if limit < 1:
                return self.deny(cmd, "not enough credits for one result row")
            args["limit"] = limit
            cost += limit * spec.row_weight
        if cost > available:
            return self.deny(cmd, "call exceeds remaining/round credit allowance")
        prior_rounds = self._tool_rounds.get((server, tool), [])
        if (server, tool) == ("registry", "list_servers") and prior_rounds:
            return self.deny(cmd, "catalogue already discovered this duel")
        if server == "citation-checker" and sum(r >= self.ctx.round - 2 for r in prior_rounds) >= 2:
            return self.deny(cmd, "citation-checker rate limit: 2 calls per 3 rounds")
        routed = replace(routed, args=args, fields=fields, headers=headers)
        signature = self._signature(routed)
        if signature in self._failed and self._failed[signature] != "unavailable":
            return self.deny(cmd, "unchanged failed call needs corrected inputs")
        if signature in self._failed:
            if self._retry_count.get(signature, 0) >= 1:
                return self.deny(cmd, "unavailable retry already exhausted")
            self._retry_count[signature] = self._retry_count.get(signature, 0) + 1
        call = self._to_tool_call(routed)
        self._issued_calls[cmd.call_index] = call
        decision = Decision(verdict="forward" if routed == cmd else "rewrite", call=call)
        self._credits_authorised += cost
        self._spent_by_round[self.ctx.round] = spent + cost
        self._reservations[cmd.call_index] = (self.ctx.round, cost)
        self._tool_rounds.setdefault((server, tool), []).append(self.ctx.round)
        if spec.is_write:
            self._writes[self._write_key(routed)] = "pending"
        if cmd.kind == "a2a":
            self._seen_delegation_ids.add(self._delegation_from(routed)["token_id"])
        self._telemetry.decision_made(cmd, decision)
        return decision

    def _delegation_from(self, cmd):
        for key in ("delegation", "delegation-token", "delegation_token"):
            if key in cmd.headers:
                return self._mapping(cmd.headers[key])
        return self._mapping(cmd.args.get("delegation_token"))

    def _admit_a2a(self, cmd):
        if admit_skill is None:
            return "A2A verifier unavailable"
        raw_card = cmd.headers.get("agent-card", cmd.headers.get("agent_card"))
        card = self._mapping(raw_card) if raw_card is not None else self._peer_cards.get(cmd.server, AGENT_CARDS.get(cmd.server))
        if card is None:
            return "unknown or malformed peer card"
        skill = cmd.args.get("skill") or next((s for s, route in SKILL_ROUTES.items()
                                              if route == (cmd.server, cmd.tool)), cmd.tool)
        if not isinstance(skill, str):
            return "malformed skill"
        admission = admit_skill(card, skill)
        if not admission.admitted or admission.peer != cmd.server or SKILL_ROUTES.get(skill) != (cmd.server, cmd.tool):
            return "card, peer and declared skill must match the executed tool"
        if cmd.headers.get("aud") not in (None, f"a2a:{cmd.server}"):
            return "delegation audience mismatch"
        if cmd.headers.get("x-card-signature") not in (None, "registry-verified"):
            return "unverified card signature"
        token = self._delegation_from(cmd)
        if token is None:
            return "A2A requires a signed delegation token (JSON object)"
        try:
            if token.get("minted_at_call_index", 0) > cmd.call_index:
                return "delegation is not yet valid"
            result = verify_delegation(token, aud=f"a2a:{cmd.server}", call_index=cmd.call_index,
                                       expected_act=self.ctx.act, seen_token_ids=self._seen_delegation_ids)
        except (TypeError, ValueError, KeyError):
            return "malformed delegation token"
        return None if result.admitted else f"delegation rejected: {result.reason}"

    def note_provenance(self, anchor, etag, *, path_id=None, drifting=False, fresher=None):
        """Drift alone is NOT freshness. Only an explicit trusted signal selects it."""
        if isinstance(anchor, str) and isinstance(etag, str) and etag:
            self._provenance_etags[anchor] = etag
        if isinstance(path_id, str) and fresher in ("w", "c"):
            self._fresh_replicas[path_id] = fresher

    def note_result(self, anchor, etag):
        self.note_provenance(anchor, etag)

    def note_card(self, server, card):
        verdict = verify_card(card)
        if verdict.admitted and verdict.peer == server:
            self._peer_cards[server] = card
            return True
        self._peer_cards.pop(server, None)
        return False

    def deny(self, cmd, reason):
        self._denied_cmd_ids.add(cmd.cmd_id)
        decision = Decision(verdict="deny", reason=reason)
        self._telemetry.decision_made(cmd, decision)
        return decision

    def _to_tool_call(self, cmd):
        return ToolCall(server=cmd.server, tool=cmd.tool, args=dict(cmd.args),
                        fields=cmd.fields, headers=dict(cmd.headers),
                        lease_id=cmd.lease_id, call_index=cmd.call_index)

if __name__ == "__main__":
    print("=== agent.gateway: Command / Decision validation ===\n")

    good_cmd = Command(
        cmd_id="cmd:0000",
        kind="mcp",
        raw="MCP slides.get_frame anchor=Frame:3f2a9c11/w/041 fields=title,body lease=lse_7f21",
        server="slides",
        tool="get_frame",
        args={"anchor": "Frame:3f2a9c11/w/041"},
        fields=("body", "title"),
        headers={},
        lease_id="lse_7f21",
        call_index=0,
    )
    print(f"  Command constructed: {good_cmd}")
    assert good_cmd.kind == "mcp"

    print("\n  Rejection demo (each must raise ValueError):")

    def _expect_value_error(label: str, fn) -> None:
        try:
            fn()
        except ValueError as exc:
            print(f"    [{label:38}] -> ValueError: {exc}")
        else:
            raise AssertionError(f"expected ValueError for case {label!r}")

    _expect_value_error("Command.kind == 'answer'", lambda: Command(
        cmd_id="cmd:0001", kind="answer", raw="x", server="slides", tool="get_frame",
        args={}, fields=(), headers={}, lease_id=None, call_index=0,
    ))
    _expect_value_error("Decision verdict='deny' with no reason", lambda: Decision(verdict="deny"))
    _expect_value_error(
        "Decision verdict='forward' with no call", lambda: Decision(verdict="forward")
    )
    _expect_value_error(
        "Decision verdict='deny' carrying a call",
        lambda: Decision(verdict="deny", reason="nope", call={"server": "x", "tool": "y"}),
    )
    _expect_value_error("Decision verdict='?' unknown", lambda: Decision(verdict="???"))

    print("\n=== Command.from_action_dict — real canonicaliser integration ===\n")
    if _canonicalise_action is None:
        print("  kit.loop.agent not importable yet — skipping the live canonicaliser demo")
        demo_commands: list[Command] = [good_cmd]
    else:
        raw_actions = [
            "MCP registry.provenance anchor=Frame:3f2a9c11/w/041 fields=etag",
            'MCP slides.query q="streamable http replaces http+sse" fields=title,body',
            "A2A curriculum-analyst.which_days_cover concept=Concept:streamable-http fields=anchor,course_day,track",
            "DISCOVER registry.list_servers fields=name",
        ]
        demo_commands = []
        for i, raw in enumerate(raw_actions):
            action = _canonicalise_action(raw, call_index=i)
            cmd = Command.from_action_dict(action, cmd_id=f"cmd:{i:04d}")
            print(f"  {raw!r}\n    -> {cmd.kind}: {cmd.server}.{cmd.tool} fields={cmd.fields}")
            demo_commands.append(cmd)
        assert {c.kind for c in demo_commands} == {"mcp", "a2a", "discover"}

        answer_action = _canonicalise_action(
            'ANSWER {"text": "day 26, track P2T2"}', call_index=None
        )
        try:
            Command.from_action_dict(answer_action, cmd_id="cmd:9999")
        except ValueError as exc:
            print(f"\n  an 'answer' action correctly refuses to become a Command: {exc}")
        else:
            raise AssertionError("expected ValueError for an 'answer' action")

    print("\n=== Gateway.decide — route, admission, authorization, budget ===\n")
    ctx = RecordingGatewayContext(
        act="learner:sv-0401",
        sub="agent:demo-team",
        scopes=frozenset({"wiki.read"}),
        credits=100,
        round=1,
        call_index=0,
        leases=(),
        history=(),
    )
    assert isinstance(ctx, GatewayContext), "RecordingGatewayContext must structurally satisfy GatewayContext"
    gw = Gateway(ctx)
    for cmd in demo_commands:
        decision = gw.decide(cmd)
        print(f"  decide({cmd.server}.{cmd.tool}) -> verdict={decision.verdict!r} quarantine={decision.quarantine}")
        if cmd.kind == "a2a":
            # The grammar demo has no signed delegation; failing closed is
            # the expected control-plane behaviour.
            assert decision.verdict == "deny"
            continue
        assert decision.verdict in {"forward", "rewrite"}
        assert decision.call is not None
        call_dict = decision.call.to_dict() if hasattr(decision.call, "to_dict") else decision.call
        assert call_dict["server"] == cmd.server
        assert call_dict["tool"] == cmd.tool
        if cmd.server != "slides":
            assert tuple(call_dict["fields"]) == cmd.fields

    print(f"\n=== Gateway.deny — the unused-by-default free-abstention path ===\n")
    denial = gw.deny(demo_commands[0], reason="demo: withholding pending a fresher registry.provenance read")
    print(f"  gw.deny(...) -> verdict={denial.verdict!r} reason={denial.reason!r} call={denial.call!r}")
    assert denial.verdict == "deny"
    assert denial.call is None
    assert demo_commands[0].cmd_id in gw._denied_cmd_ids

    print(f"\n=== own_telemetry — recorded on YOUR side only, never shown to the opponent ===\n")
    print(f"  {len(ctx.events)} events recorded on this ctx this run:")
    for ev in ctx.events:
        print(f"    {ev['name']}: {sorted(ev['payload'].keys())}")
    assert len(ctx.events) >= len(demo_commands) * 2 + 1  # decision_seen + decision_made per call, plus the deny

    print("\nAll agent/gateway.py demos passed.")
