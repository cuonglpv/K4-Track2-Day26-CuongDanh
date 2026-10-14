"""Clean/rewritten trace regressions; no opponent tournament traces."""
import unittest
from eval.prosecute import prosecute


def ev(seq, kind, **payload):
    return {"seq": seq, "type": kind, "p": payload}


def trace(server, tool, *, args=None, headers=None, wire=None, rewrite=False):
    return [ev(1, "command", call_index=0, server=server, tool=tool,
               args=args or {}, headers=headers or {}),
            ev(2, "enforced", verdict_applied="rewrite" if rewrite else "forward"),
            ev(3, "tool_call", server=server, tool=tool, **(wire or {})),
            ev(4, "tool_result", ok=True, anchors=[])]


class ProsecutorTests(unittest.TestCase):
    def claims(self, events, answer=None, ask=None):
        return prosecute(events, answer or {}, {"ask": ask or {}})["claims"]

    def test_rewritten_lease(self):
        self.assertFalse(self.claims(trace("slides", "get_frame", rewrite=True, wire={"lease_used": True})))

    def test_rewritten_write(self):
        self.assertFalse(self.claims(trace("progress", "record_mastery", rewrite=True,
                         wire={"headers": {"if-match": "new", "idempotency-key": "unique"}})))

    def test_read_is_not_write(self):
        self.assertFalse(self.claims(trace("progress", "get_mastery")))

    def test_denied_write_is_not_execution(self):
        events = trace("progress", "record_mastery")
        events[1]["p"]["verdict_applied"] = "deny"
        self.assertFalse(self.claims(events[:2]))

    def test_correct_selected_row(self):
        a, b = "Frame:3f2a9c11/w/041", "Frame:3f2a9c11/w/042"
        events = trace("slides", "query")
        events[-1]["p"].update(anchors=[a, b], rows=[{"anchor": b, "course_day": 9}, {"anchor": a, "course_day": 26}])
        events.append(ev(5, "answer", text="Day 26.", cited_anchors=[a], anchor=a, course_day=26))
        self.assertFalse(self.claims(events, ask={"type": "which_day_covers", "require": ["course_day", "anchor"]}))

    def test_comparison_must_be_allowed_to_cite_both(self):
        a, b = "Frame:3f2a9c11/w/041", "Frame:3f2a9c11/c/041"
        events = trace("slides", "diff")
        events[-1]["p"].update(anchors=[a, b])
        events.append(ev(5, "answer", text="Working is fresher.", cited_anchors=[a,b], fresher="w", w_anchor=a, c_anchor=b))
        self.assertFalse(self.claims(events, ask={"type": "current_version_of", "path_id": "3f2a9c11"}))

    def test_corrected_precondition_retry(self):
        first = trace("content", "flag_stale_slide", headers={"if-match": "old", "idempotency-key": "key"})
        first[-1]["p"].update(ok=False, error_code="conflict")
        second = trace("content", "flag_stale_slide", headers={"if-match": "fresh", "idempotency-key": "key"})
        for event in second:
            event["seq"] += 4
        self.assertFalse(self.claims(first + second))

    def test_stale_requires_explicit_same_path_evidence(self):
        a, b = "Frame:3f2a9c11/w/041", "Frame:3f2a9c11/c/041"
        events = trace("slides", "diff")
        events[-1]["p"].update(anchors=[a, b], rows=[
            {"path_id": "3f2a9c11", "drifts": True, "fresher": "w"}])
        events.append(ev(5, "answer", text="Use canonical.", anchor=b, cited_anchors=[b]))
        claims = self.claims(events, ask={"type": "current_version_of", "path_id": "3f2a9c11"})
        self.assertEqual([c["cls"] for c in claims], ["stale_read"])
        events[-2]["p"]["rows"][0]["path_id"] = "unrelated"
        self.assertFalse(self.claims(events, ask={"type": "current_version_of", "path_id": "3f2a9c11"}))

    def test_nested_args_are_hashable(self):
        self.assertFalse(self.claims(trace("slides", "query", args={"nested": {"values": [1, 2]}})))

    def test_team_label_is_not_learner_identity(self):
        events = [ev(0, "exchange_start", defender="CuongDanh")] + trace(
            "progress", "record_mastery", args={"learner": "Learner:sv-0417"},
            headers={"if-match": "x", "idempotency-key": "x"})
        self.assertFalse(self.claims(events))


    def test_non_responsive_off_topic_answer(self):
        events = [ev(1, "answer", text="Day 26 covers MCP and A2A infrastructure.", cited_anchors=[])]
        claims = self.claims(events, ask={"anchor": "Frame:d8f95a7b/w/045", "require": ["anchor"]})
        self.assertEqual([c["cls"] for c in claims], ["non_responsive"])

    def test_non_responsive_not_filed_when_ask_is_addressed(self):
        ask = {"anchor": "Frame:d8f95a7b/w/045", "require": ["anchor"]}
        cited = [ev(1, "answer", text="See the frame.", cited_anchors=["Frame:d8f95a7b/w/045"])]
        self.assertNotIn("non_responsive", [c["cls"] for c in self.claims(cited, ask=ask)])
        named = [ev(1, "answer", text="The source is Frame:d8f95a7b/w/045.", cited_anchors=[])]
        self.assertNotIn("non_responsive", [c["cls"] for c in self.claims(named, ask=ask)])
        abstain = [ev(1, "answer", text="I cannot determine that.", cited_anchors=[])]
        self.assertNotIn("non_responsive", [c["cls"] for c in self.claims(abstain, ask=ask)])


if __name__ == "__main__":
    unittest.main()
