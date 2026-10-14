"""Deterministic answer checks, used by agent.runtime before ANSWER execution.

Source metadata must come from retrieved observations, never from the answer.
These checks do not establish semantic truth.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Any, Mapping
from kit.world.anchor import Anchor, AnchorSyntaxError


@dataclass(frozen=True, slots=True)
class GroundingResult:
    grounded: bool
    cited: tuple
    ungrounded: tuple
    malformed: tuple


def check_grounding(answer, retrieved_anchors, *, require_citation=True):
    values = answer.get("cited_anchors", ())
    if not isinstance(values, (list, tuple)):
        return GroundingResult(False, (), (), ("cited_anchors must be a list",))
    malformed, missing = [], []
    for anchor in values:
        try:
            Anchor.parse(anchor)
        except (AnchorSyntaxError, TypeError, ValueError, AttributeError):
            malformed.append(str(anchor))
            continue
        if anchor not in retrieved_anchors:
            missing.append(anchor)
    return GroundingResult(not malformed and not missing and (bool(values) or not require_citation),
                           tuple(values), tuple(missing), tuple(malformed))


@dataclass(frozen=True, slots=True)
class InjectionScanResult:
    suspicious: bool
    matched_patterns: tuple


def scan_for_injected_instructions(text):
    if not isinstance(text, str):
        return InjectionScanResult(True, ("non-text-content",))
    patterns = (
        ("ignore-prior-instructions", r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b"),
        ("fake-system-message", r"\b(?:as\s+the\s+system|new\s+system\s+prompt|system\s+override)\b"),
        ("instruction-override", r"\b(?:override|bypass|disable)\s+(?:the\s+)?(?:safety|guardrail|policy|rules?)\b"),
        ("secret-request", r"\b(?:reveal|exfiltrate|disclose|print)\b.{0,80}\b(?:system prompt|api[ _-]?key|token|secret|ctx\.(?:act|scopes)|learner data)\b"),
        ("vietnamese-override", r"(?:bỏ qua|phớt lờ).{0,50}(?:chỉ dẫn|hướng dẫn|lệnh).{0,25}(?:trước|cũ)"),
    )
    folded = " ".join(text.lower().split())
    hits = tuple(name for name, pattern in patterns if re.search(pattern, folded))
    return InjectionScanResult(bool(hits), hits)


def _educational(text):
    # Require the framing BEFORE the imperative, not appended after it.
    return bool(re.match(r"\s*(?:(?:an?|this|the)\s+)?(?:example|quoted|quotation|attack|warning|do not obey|never follow)\b|^\s*(?:ví dụ|cảnh báo|không làm theo)",
                         text, re.I))


@dataclass(frozen=True, slots=True)
class RedactionResult:
    redacted_text: str
    hits: tuple


def redact(text, *, private_values=()):
    if not isinstance(text, str):
        return RedactionResult("[redacted non-text content]", ("non-text-content",))
    hits = []
    for value in private_values:
        if not isinstance(value, str) or len(" ".join(value.split())) < 40:
            continue
        text, count = re.subn(r"\s+".join(re.escape(w) for w in value.split()),
                             "[private content redacted]", text, flags=re.I)
        if count:
            hits.append("private-content")
    for name, pattern in (
        ("email", r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
        ("api-key", r"\b(?:sk|ghp|api)[_-][A-Za-z0-9_-]{16,}\b"),
        ("learner-id", r"\b(?:Learner:)?sv-\d{4}\b"),
        ("phone", r"(?<!\w)(?:\+?\d[\d .-]{7,}\d)(?!\w)"),
    ):
        text, count = re.subn(pattern, f"[{name} redacted]", text, flags=re.I)
        if count:
            hits.append(name)
    return RedactionResult(text, tuple(dict.fromkeys(hits)))


@dataclass(frozen=True, slots=True)
class ArithmeticCheckResult:
    checked: bool
    ok: bool | None
    detail: str


_NUMBER_RE = re.compile(r"(?<![\w:/])-?\d+(?:\.\d+)?(?:\s*(?:%|million\b|billion\b|M\b|B\b))?")
_EQUATION_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")


def verify_arithmetic(text, source_texts=None):
    if not isinstance(text, str):
        return ArithmeticCheckResult(True, False, "answer is not text")
    equations = list(_EQUATION_RE.finditer(text))
    try:
        with localcontext() as context:
            context.prec = max(50, len(text) * 2)
            for match in equations:
                left, op, right, claimed = match.groups()
                a, b, c = Decimal(left), Decimal(right), Decimal(claimed)
                if op == "/" and not b:
                    return ArithmeticCheckResult(True, False, "division by zero")
                actual = {"+": lambda: a+b, "-": lambda: a-b, "*": lambda: a*b, "/": lambda: a/b}[op]()
                if not actual.is_finite() or actual != c:
                    return ArithmeticCheckResult(True, False, "unsupported equation")
    except (ArithmeticError, ValueError):
        return ArithmeticCheckResult(True, False, "invalid arithmetic")
    if source_texts is not None:
        sources = "\n".join(s for s in source_texts if isinstance(s, str))
        known = {re.sub(r"\s+", "", n).casefold() for n in _NUMBER_RE.findall(sources)}
        prose = _EQUATION_RE.sub(lambda m: m.group(1) + " " + m.group(3), text)
        numbers = {re.sub(r"\s+", "", n).casefold() for n in _NUMBER_RE.findall(prose)}
        if numbers - known:
            return ArithmeticCheckResult(True, False, "number/precision absent from cited sources")
        if numbers and re.search(r"\bexact(?:ly)?\b|chính xác", text, re.I) and re.search(r"~|\b(?:about|roughly|approximately)\b|khoảng|xấp xỉ", sources, re.I):
            return ArithmeticCheckResult(True, False, "approximate source restated as exact")
        return ArithmeticCheckResult(bool(numbers or equations), True if numbers or equations else None, "source numbers checked")
    return ArithmeticCheckResult(bool(equations), True if equations else None, "equations only; source precision not checked")


def abstention_policy(grounding):
    return not grounding.grounded


@dataclass(frozen=True, slots=True)
class AnswerSafetyResult:
    answer: dict[str, Any]
    submit: bool
    reasons: tuple


_REFERENCE_KEYS = ("anchor", "cited_anchors", "w_anchor", "c_anchor")


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            if key not in _REFERENCE_KEYS:
                yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)


def _numeric_fields(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        yield str(value)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            if key not in _REFERENCE_KEYS:
                yield from _numeric_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _numeric_fields(child)


def prepare_answer(answer, retrieved_anchors, *, untrusted_texts=(), source_rows=(), required_fields=()):
    candidate = dict(answer) if isinstance(answer, Mapping) else {}
    def abstain(reason):
        return AnswerSafetyResult({"text": "Insufficient safe, grounded evidence to answer.", "cited_anchors": []}, False, (reason,))
    if not isinstance(candidate.get("text"), str):
        return abstain("malformed-answer")
    if not check_grounding(candidate, retrieved_anchors).grounded:
        return abstain("ungrounded-or-malformed-citation")
    if any(key not in candidate or candidate[key] is None for key in required_fields):
        return abstain("missing-required-field")
    cited = set(candidate["cited_anchors"])
    private_values, sources, poisoned = [], [], set()
    rows = list(source_rows)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        anchor = row.get("anchor", "")
        meta = row.get("meta", {}) if isinstance(row.get("meta"), Mapping) else {}
        sensitive = isinstance(anchor, str) and anchor.startswith(("Note:", "Learner:"))
        private = row.get("private", meta.get("private", sensitive))
        if private:
            private_values.extend(_strings(row))
        body = row.get("body", "")
        if scan_for_injected_instructions(body).suspicious and not _educational(body):
            poisoned.add(anchor)
            continue
        if anchor in cited and not private:
            sources.extend(_strings(row))
            sources.extend(_numeric_fields(row))
    if cited and cited <= poisoned:
        return abstain("only-poisoned-evidence")
    # A poisoned unrelated result does not veto clean evidence. Educational
    # discussion of an injection is allowed; direct imperative output is not.
    for text in _strings(candidate):
        if scan_for_injected_instructions(text).suspicious and not _educational(text):
            return abstain("injection-shaped-answer")
    hits = []
    def clean(value):
        if isinstance(value, str):
            redaction = redact(value, private_values=private_values)
            hits.extend(redaction.hits)
            return redaction.redacted_text
        if isinstance(value, Mapping):
            return {k: v if k in _REFERENCE_KEYS else clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value
    candidate = clean(candidate)
    texts = list(untrusted_texts)
    numeric_sources = sources if rows else (texts if texts else None)
    numeric_text = re.sub(r"(?:Frame|Concept|Note|Learner):[\w:/.-]+", "", candidate["text"])
    numeric_text += " " + " ".join(_numeric_fields(candidate))
    arithmetic = verify_arithmetic(numeric_text, numeric_sources)
    if arithmetic.checked and arithmetic.ok is False:
        return abstain("unsupported-arithmetic-or-precision")
    return AnswerSafetyResult(candidate, True, tuple(dict.fromkeys(hits)))


if __name__ == "__main__":
    a = "Frame:3f2a9c11/w/041"
    assert not prepare_answer({"text": "2 + 2 = 5", "cited_anchors": [a]}, [a]).submit
    assert verify_arithmetic("2 + 2 = 4").ok is True
    print("Guardrail smoke checks passed.")
