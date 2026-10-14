"""Opt-in answer assembly around the kit loop, without patching trusted kit.

Hosts must instantiate this Agent instead of kit.loop.Agent to enforce the
answer gate. The distributed spar.py does not instantiate either loop; its
fixed answer cannot be guarded through the Gateway contract.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from kit.loop.agent import Agent as KitAgent
from agent.guardrails import prepare_answer


class Agent(KitAgent):
    def __init__(self, *args, gateway=None, required_fields=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.gateway = gateway
        self.required_fields = tuple(required_fields)
        self.retrieved_anchors = set()
        self.source_rows = []
        self.last_safety_result = None

    def _execute(self, action):
        if action["kind"] == "answer":
            safety = prepare_answer(
                action["args"], self.retrieved_anchors,
                source_rows=self.source_rows, required_fields=self.required_fields,
            )
            self.last_safety_result = safety
            # Update before the trusted environment records the ANSWER and
            # before KitAgent.run constructs its RunResult from this action.
            action["args"] = safety.answer
            action["raw"] = "ANSWER " + json.dumps(safety.answer, ensure_ascii=False)
        observation = super()._execute(action)
        if action["kind"] != "answer":
            result = observation.get("result", observation)
            if isinstance(result, Mapping):
                if self.gateway is not None:
                    call = self.gateway._issued_calls.pop(action["call_index"], None)
                    if call is not None:
                        self.gateway.note_outcome(call, result)
                if result.get("ok") is True:
                    raw_anchors = result.get("anchors", ())
                    anchors = [a for a in raw_anchors if isinstance(a, str)] if isinstance(raw_anchors, (list, tuple)) else []
                    self.retrieved_anchors.update(anchors)
                    rows = result.get("rows", ())
                    if not isinstance(rows, (list, tuple)):
                        rows = ()
                    for index, row in enumerate(rows):
                        if not isinstance(row, Mapping):
                            continue
                        source = dict(row)
                        if "anchor" not in source and len(rows) == len(anchors):
                            source["anchor"] = anchors[index]
                        self.source_rows.append(source)
        return observation
