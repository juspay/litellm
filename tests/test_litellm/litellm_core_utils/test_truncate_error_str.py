"""
Regression tests for giant exception-string truncation.

Root cause this guards against: a client sends an ~82MB request body; the
provider's pydantic validation error echoes the full input; logging that
unbounded str(e) blocked the event loop for the duration of the
SecretRedactionFilter regex scan (~0.19s/MB) plus the stdout write under
fluentbit backpressure (observed 9.5-minute freeze in prod).
"""

import io
import json
import logging
import time

import pytest

import litellm._logging as L
from litellm.litellm_core_utils.core_helpers import (
    MAX_ERROR_STR_LOG_LENGTH,
    _truncate_str,
    truncate_error_str,
)


class TestTruncateErrorStr:
    def test_short_exception_unchanged(self):
        assert truncate_error_str(ValueError("boom")) == "boom"

    def test_exception_at_limit_unchanged(self):
        msg = "x" * MAX_ERROR_STR_LOG_LENGTH
        assert truncate_error_str(ValueError(msg)) == msg

    def test_giant_exception_bounded(self):
        out = truncate_error_str(ValueError("x" * 82_000_000))
        # head + marker + tail; worst case ~2.5KB
        assert len(out) <= MAX_ERROR_STR_LOG_LENGTH + 600

    def test_head_and_tail_preserved(self):
        s = "HEAD" + "m" * 5_000 + "TAIL errors.pydantic.dev/urls"
        out = _truncate_str(s)
        assert out.startswith("HEAD")
        assert "errors.pydantic.dev/urls" in out  # pydantic summary link lives at the end
        assert str(len(s)) in out  # original length reported

    def test_no_duplicate_middle_near_cap(self):
        # strings just over the cap must not log the overlapping region twice
        s = "a" * (MAX_ERROR_STR_LOG_LENGTH + 100)
        out = _truncate_str(s)
        marker = f"... [truncated, {len(s)} chars total] ..."
        head, tail = out.split(marker)
        assert len(head) == MAX_ERROR_STR_LOG_LENGTH
        # tail holds only the chars past the head window - no overlap, no duplication
        assert len(tail) == 100
        # head + tail reconstruct the original string exactly
        assert head + tail == s

    def test_unprintable_exception(self):
        class Bad(Exception):
            def __str__(self):
                raise RuntimeError("nope")

        assert truncate_error_str(Bad()) == "<unprintable exception>"

    def test_custom_max_length(self):
        out = truncate_error_str(ValueError("x" * 100_000), max_length=10_000)
        assert len(out) <= 10_000 + 600


class TestSecretRedactionFilterExcInfo:
    """logger.exception()/exc_info=True attach the full traceback; the filter
    must cap the formatted exception or the giant str(e) leaks through every
    wrapped log site (formatException embeds str(e) on its last line)."""

    def _make_logger(self, json_mode: bool):
        stream = io.StringIO()
        h = logging.StreamHandler(stream)
        h.addFilter(L._secret_filter)
        h.setFormatter(L.JsonFormatter() if json_mode else logging.Formatter("%(message)s"))
        log = logging.getLogger(f"test_exc_info_{json_mode}_{id(stream)}")
        log.handlers = [h]
        log.propagate = False
        log.setLevel(logging.INFO)
        return log, stream

    @pytest.mark.parametrize("json_mode", [False, True])
    def test_giant_exception_capped(self, json_mode):
        log, stream = self._make_logger(json_mode)
        try:
            raise ValueError("y" * 100_000)
        except ValueError as e:
            log.exception("Exception %s", "truncated-msg")
        out = stream.getvalue()
        assert len(out) <= 2 * (MAX_ERROR_STR_LOG_LENGTH + 600) + 200, len(out)

    @pytest.mark.parametrize("json_mode", [False, True])
    def test_small_traceback_intact(self, json_mode):
        log, stream = self._make_logger(json_mode)
        try:
            raise RuntimeError("small error")
        except RuntimeError:
            log.exception("small exc")
        out = stream.getvalue()
        assert "RuntimeError: small error" in out
        assert "Traceback" in out or "stacktrace" in out

    @pytest.mark.parametrize("json_mode", [False, True])
    def test_plain_message_unaffected(self, json_mode):
        log, stream = self._make_logger(json_mode)
        log.info("plain %s", "message")
        assert "plain message" in stream.getvalue()

    @pytest.mark.parametrize("json_mode", [False, True])
    def test_82mb_exception_fast(self, json_mode):
        """The incident scenario: 82MB exception must not block for seconds."""
        log, stream = self._make_logger(json_mode)
        try:
            raise ValueError("z" * 82_000_000)
        except ValueError as e:
            t0 = time.perf_counter()
            log.exception("incident: %s", truncate_error_str(e))
            elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"filter took {elapsed:.1f}s on a giant exception"
        assert len(stream.getvalue()) <= 2 * (MAX_ERROR_STR_LOG_LENGTH + 600) + 200

    def test_json_formatter_keeps_stacktrace_field(self):
        log, stream = self._make_logger(json_mode=True)
        try:
            raise RuntimeError("keep me")
        except RuntimeError:
            log.exception("json mode")
        out = stream.getvalue()
        assert "stacktrace" in out
        assert "keep me" in out


class TestSecretRedactionFilterFailClosed:
    """Every path through the filter must bound giant exceptions, including
    unwrapped call sites and internal formatting failures. Four-way matrix:
    redaction enabled/disabled x text/JSON formatter."""

    BOUND = 11_000  # 10KB message cap + marker + tail + JSON envelope

    def _make_logger(self, json_mode: bool):
        stream = io.StringIO()
        h = logging.StreamHandler(stream)
        h.addFilter(L._secret_filter)
        h.setFormatter(L.JsonFormatter() if json_mode else logging.Formatter("%(message)s"))
        log = logging.getLogger(f"failclosed_{json_mode}_{id(stream)}")
        log.handlers = [h]
        log.propagate = False
        log.setLevel(logging.INFO)
        return log, stream

    @pytest.fixture(params=[False, True])
    def json_mode(self, request):
        return request.param

    def test_raw_str_in_message_bounded(self, json_mode):
        """Unwrapped call site: f"...{e}" embeds the raw str(e) in the message."""
        log, stream = self._make_logger(json_mode)
        try:
            raise ValueError("y" * 100_000)
        except ValueError as e:
            log.error(f"request failed: {e}")
        out = stream.getvalue()
        assert len(out) <= self.BOUND, len(out)

    def test_formatexception_failure_fails_closed(self, json_mode):
        """If formatException() raises, the record must not keep the raw
        exc_info - a broken formatter cannot leak the unbounded exception."""
        orig = L.SecretRedactionFilter._formatter.formatException
        L.SecretRedactionFilter._formatter.formatException = lambda ei: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        try:
            log, stream = self._make_logger(json_mode)
            try:
                raise ValueError("z" * 100_000)
            except ValueError:
                log.exception("msg")
        finally:
            L.SecretRedactionFilter._formatter.formatException = orig
        out = stream.getvalue()
        assert len(out) <= self.BOUND, len(out)
        assert "<exception formatting failed" in out

    def test_small_traceback_intact(self, json_mode):
        log, stream = self._make_logger(json_mode)
        try:
            raise RuntimeError("small error")
        except RuntimeError:
            log.exception("small exc")
        out = stream.getvalue()
        assert "RuntimeError: small error" in out
        assert "Traceback" in out or "stacktrace" in out

    def test_plain_message_unaffected(self, json_mode):
        log, stream = self._make_logger(json_mode)
        log.info("plain %s", "message")
        assert "plain message" in stream.getvalue()

    def test_exc_info_path_bounded(self, json_mode):
        log, stream = self._make_logger(json_mode)
        try:
            raise ValueError("y" * 100_000)
        except ValueError:
            log.exception("msg")
        assert len(stream.getvalue()) <= self.BOUND, len(stream.getvalue())

    def test_redaction_disabled_still_bounded(self, json_mode, monkeypatch):
        """The cap must apply even when LITELLM_DISABLE_REDACT_SECRETS=true:
        the giant-log-line freeze is a size problem, not a secrecy problem."""
        monkeypatch.setattr(L, "_ENABLE_SECRET_REDACTION", False)

        # raw str(e) in message
        log, stream = self._make_logger(json_mode)
        try:
            raise ValueError("y" * 100_000)
        except ValueError as e:
            log.error(f"request failed: {e}")
        assert len(stream.getvalue()) <= self.BOUND, len(stream.getvalue())

        # exc_info path
        log, stream = self._make_logger(json_mode)
        try:
            raise ValueError("y" * 100_000)
        except ValueError:
            log.exception("msg")
        assert len(stream.getvalue()) <= self.BOUND, len(stream.getvalue())

    def test_giant_message_with_exc_info_bounded(self, json_mode):
        """Giant str(e) interpolated into the message AND attached via
        exc_info at the same time (logger.exception(f"...{e}"))."""
        log, stream = self._make_logger(json_mode)
        try:
            raise ValueError("y" * 100_000)
        except ValueError as e:
            log.exception(f"request failed: {e}")
        assert len(stream.getvalue()) <= 2 * self.BOUND, len(stream.getvalue())

    def test_giant_bare_traceback_in_message_bounded(self, json_mode):
        """logger.info(traceback.format_exc()) style: the message itself is
        the giant traceback."""
        try:
            raise ValueError("t" * 100_000)
        except ValueError:
            tb = __import__("traceback").format_exc()
        log, stream = self._make_logger(json_mode)
        log.error(tb)
        assert len(stream.getvalue()) <= self.BOUND, len(stream.getvalue())

    def test_secret_in_retained_tail_is_redacted(self, json_mode, monkeypatch):
        """Capping must not push secrets past redaction: a secret sitting in
        the retained head or tail must still be scrubbed."""
        monkeypatch.setattr(L, "_ENABLE_SECRET_REDACTION", True)
        from litellm.litellm_core_utils.secret_redaction import redact_string

        # find a string the redactor actually rewrites, then place it in
        # both the head region and the tail region of an oversized message
        probe = "sk-1234567890abcdef1234567890abcdef12345678"
        assert probe != redact_string(probe), "probe is not redacted; pick another"

        giant_head = "A" * 500 + probe + "B" * 20_000
        giant_tail = "C" * 20_000 + probe
        for msg in (giant_head, giant_tail):
            log, stream = self._make_logger(json_mode)
            log.error(msg)
            out = stream.getvalue()
            assert len(out) <= self.BOUND, len(out)
            assert probe not in out, "secret survived redaction after capping"

    def test_deterministic_upper_bound(self, json_mode):
        """Every rendered record stays within a deterministic KB-scale bound
        regardless of input size (100KB vs 82MB)."""
        for size in (100_000, 1_000_000, 82_000_000):
            log, stream = self._make_logger(json_mode)
            try:
                raise ValueError("y" * size)
            except ValueError as e:
                log.exception(f"failed: {e}")
            assert len(stream.getvalue()) <= 2 * self.BOUND, f"size={size}: {len(stream.getvalue()):,}"

    def test_formatexception_failure_once_then_ok(self, json_mode):
        """Inject a formatter that raises once: that record fails closed, and
        the next record with a working formatter still gets a traceback."""
        orig = L.SecretRedactionFilter._formatter.formatException
        calls = {"n": 0}

        def flaky(ei):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return orig(ei)

        L.SecretRedactionFilter._formatter.formatException = flaky
        try:
            log, stream = self._make_logger(json_mode)
            try:
                raise ValueError("z" * 100_000)
            except ValueError:
                log.exception("first")
            try:
                raise RuntimeError("second small")
            except RuntimeError:
                log.exception("second")
        finally:
            L.SecretRedactionFilter._formatter.formatException = orig
        out = stream.getvalue()
        assert len(out) <= self.BOUND + 2_600, len(out)
        assert "<exception formatting failed" in out
        assert "RuntimeError: second small" in out  # recovered


class TestBoundedExtras:
    """extra={...} fields must not bypass the size choke point.

    Real LogRecord + filter + handler flows, both formatters, redaction
    enabled and disabled. A 100KB extra previously produced a ~100KB
    JSON record because the filter's extras handling ran after the
    redaction early-return (disabled mode) and never capped (enabled
    mode); nested values were not touched at all.
    """

    CEILING = L.MAX_JSON_RECORD_LENGTH  # hard final-output limit
    # message cap + exc cap + extras budget + JSON envelope headroom
    TOTAL = L.MAX_LOG_MSG_LENGTH + L.MAX_EXC_TEXT_LENGTH + L.MAX_EXTRA_TOTAL_BUDGET + 2_000

    def _make_logger(self, json_mode: bool):
        stream = io.StringIO()
        h = logging.StreamHandler(stream)
        h.addFilter(L._secret_filter)
        h.setFormatter(L.JsonFormatter() if json_mode else logging.Formatter("%(message)s"))
        log = logging.getLogger(f"extras_{json_mode}_{id(stream)}")
        log.handlers = [h]
        log.propagate = False
        log.setLevel(logging.INFO)
        return log, stream

    @pytest.fixture(params=[False, True])
    def json_mode(self, request):
        return request.param

    def _emit(self, json_mode, level=logging.INFO, **kwargs):
        log, stream = self._make_logger(json_mode)
        getattr(log, logging.getLevelName(level).lower())(**kwargs)
        return stream.getvalue()

    # --- direct giant extras ---

    def test_direct_string_extra(self, json_mode):
        out = self._emit(json_mode, msg="small", extra={"huge": "x" * 100_000})
        assert len(out) <= self.TOTAL, len(out)

    def test_direct_bytes_extra(self, json_mode):
        out = self._emit(json_mode, msg="small", extra={"huge": b"x" * 100_000})
        assert len(out) <= self.TOTAL, len(out)

    def test_nested_dict_giant_string(self, json_mode):
        out = self._emit(
            json_mode, msg="small", extra={"d": {"inner": "x" * 100_000}}
        )
        assert len(out) <= self.TOTAL, len(out)

    def test_nested_list_tuple_giant_strings(self, json_mode):
        out = self._emit(
            json_mode,
            msg="small",
            extra={"l": ["y" * 100_000, ("z" * 100_000,)]},
        )
        assert len(out) <= self.TOTAL, len(out)

    # --- combined budget ---

    def test_many_medium_strings_exceed_total_budget(self, json_mode):
        # 200 strings x 1,900 chars = 380KB combined; each is under the
        # individual cap, so only the shared budget bounds the total
        out = self._emit(
            json_mode,
            msg="small",
            extra={"l": ["m" * 1_900 for _ in range(200)]},
        )
        assert len(out) <= self.TOTAL, len(out)

    def test_multiple_separate_extras_exceed_total_budget(self, json_mode):
        extra = {f"k{i}": "n" * 1_900 for i in range(30)}  # 57KB combined
        out = self._emit(json_mode, msg="small", extra=extra)
        assert len(out) <= self.TOTAL, len(out)

    # --- structure edge cases ---

    def test_deeply_nested_extras(self, json_mode):
        value = "leaf"
        for _ in range(30):
            value = {"nest": value}
        out = self._emit(json_mode, msg="small", extra={"deep": value})
        assert len(out) <= self.TOTAL, len(out)
        if json_mode:
            assert "max depth exceeded" in out

    def test_cyclic_dict_extra(self, json_mode):
        d: dict = {}
        d["self"] = d
        out = self._emit(json_mode, msg="small", extra={"cyc": d})
        assert len(out) <= self.TOTAL, len(out)
        if json_mode:
            assert "circular reference" in out

    def test_cyclic_list_extra(self, json_mode):
        lst: list = []
        lst.append(lst)
        out = self._emit(json_mode, msg="small", extra={"cyc": lst})
        assert len(out) <= self.TOTAL, len(out)
        if json_mode:
            assert "circular reference" in out

    def test_unrenderable_object_extra(self, json_mode):
        class Bad:
            def __str__(self):
                raise RuntimeError("nope")

            def __repr__(self):
                raise RuntimeError("nope")

        out = self._emit(json_mode, msg="small", extra={"bad": Bad()})
        assert len(out) <= self.TOTAL, len(out)
        if json_mode:
            assert "unrenderable object" in out

    def test_caller_structure_not_mutated(self, json_mode):
        nested = {"inner": "x" * 100_000, "keep": "small"}
        top = {"nested": nested, "n": 5}
        self._emit(json_mode, msg="small", extra={"top": top})
        # the caller's dict is untouched: still has the giant string
        assert nested["inner"] == "x" * 100_000
        assert nested["keep"] == "small"
        assert top["n"] == 5

    # --- combined with other giant paths ---

    def test_giant_everything(self, json_mode):
        try:
            raise ValueError("e" * 100_000)
        except ValueError as err:
            log, stream = self._make_logger(json_mode)
            log.error(
                f"failed: {err}",
                extra={"huge": "x" * 100_000, "struct": {"inner": "y" * 100_000}},
                exc_info=err,
            )
        out = stream.getvalue()
        assert len(out) <= 2 * self.TOTAL, len(out)

    def test_prepopulated_giant_exc_text_no_exc_info(self, json_mode):
        log, stream = self._make_logger(json_mode)
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname=__file__, lineno=1,
            msg="small", args=(), exc_info=None,
        )
        record.exc_text = "E" * 100_000
        log.handle(record)
        out = stream.getvalue()
        assert len(out) <= self.TOTAL, len(out)

    def test_giant_stack_info(self, json_mode):
        log, stream = self._make_logger(json_mode)
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname=__file__, lineno=1,
            msg="small", args=(), exc_info=None,
        )
        record.stack_info = "S" * 100_000
        log.handle(record)
        out = stream.getvalue()
        assert len(out) <= self.TOTAL, len(out)

    # --- secrets and redaction ---

    def test_secret_in_extra_head_and_tail_redacted(self, json_mode, monkeypatch):
        monkeypatch.setattr(L, "_ENABLE_SECRET_REDACTION", True)
        from litellm.litellm_core_utils.secret_redaction import redact_string

        probe = "sk-1234567890abcdef1234567890abcdef12345678"
        assert probe != redact_string(probe)
        # head region
        out = self._emit(
            json_mode, msg="small",
            extra={"h": "A" * 500 + probe + "B" * 20_000},
        )
        assert probe not in out
        # tail region
        out = self._emit(
            json_mode, msg="small",
            extra={"t": "C" * 20_000 + probe},
        )
        assert probe not in out
        # nested
        out = self._emit(
            json_mode, msg="small",
            extra={"d": {"deep": ["x" * 500 + probe + "y" * 20_000]}},
        )
        assert probe not in out

    def test_redaction_disabled_output_bounded(self, json_mode, monkeypatch):
        monkeypatch.setattr(L, "_ENABLE_SECRET_REDACTION", False)
        out = self._emit(json_mode, msg="small", extra={"huge": "x" * 100_000})
        assert len(out) <= self.TOTAL, len(out)

    # --- normal behavior preserved ---

    def test_small_structured_extras_keep_shape(self, json_mode):
        extra = {
            "model": "gpt-4", "tokens": 42, "ok": True,
            "tags": ["a", "b"], "meta": {"k": "v"}, "none": None,
        }
        out = self._emit(json_mode, msg="small", extra=extra)
        if json_mode:
            parsed = json.loads(out)
            assert parsed["model"] == "gpt-4"
            assert parsed["tokens"] == 42
            assert parsed["ok"] is True
            assert parsed["tags"] == ["a", "b"]
            assert parsed["meta"] == {"k": "v"}
            assert parsed["none"] is None
        else:
            assert "small" in out

    def test_emitted_json_parses(self, json_mode):
        """Every truncated JSON output must still be valid JSON."""
        cases = [
            {"huge": "x" * 100_000},
            {"d": {"inner": "x" * 100_000}},
            {"l": ["y" * 100_000 for _ in range(50)]},
            {f"k{i}": "n" * 1_900 for i in range(50)},
        ]
        for extra in cases:
            out = self._emit(json_mode, msg="small", extra=extra)
            if json_mode:
                parsed = json.loads(out)
                assert "message" in parsed

    def test_final_record_below_ceiling(self, json_mode):
        out = self._emit(
            json_mode,
            msg="m" * 100_000,
            extra={
                "huge": "x" * 100_000,
                "l": ["y" * 100_000 for _ in range(20)],
            },
        )
        assert len(out) <= self.CEILING, len(out)

    # --- redaction invariant (spy) ---

    def test_redact_string_never_receives_oversized_input(self, json_mode, monkeypatch):
        """Instrument _redact_string and assert every input length stays at
        or below the documented content limits."""
        seen: list[int] = []
        orig = L._redact_string

        def spy(value: str) -> str:
            seen.append(len(value))
            return orig(value)

        monkeypatch.setattr(L, "_redact_string", spy)
        monkeypatch.setattr(L, "_ENABLE_SECRET_REDACTION", True)

        try:
            raise ValueError("e" * 100_000)
        except ValueError as err:
            log, stream = self._make_logger(json_mode)
            log.error(
                f"failed: {err}",
                extra={"huge": "x" * 100_000, "d": {"inner": "y" * 100_000}},
                exc_info=err,
            )
        out = stream.getvalue()
        assert len(out) <= 2 * self.TOTAL, len(out)
        # The largest input redaction ever sees is a fully-expanded head+tail
        # truncation result: cap + 39-char marker + 500-char tail.
        assert seen, "redaction never ran"
        bound = max(
            L.MAX_LOG_MSG_LENGTH,
            L.MAX_EXC_TEXT_LENGTH,
            L.MAX_EXTRA_STR_LENGTH,
        ) + 600
        assert max(seen) <= bound, f"max redaction input: {max(seen):,} > {bound}"

    # --- multi-handler ---

    def test_multiple_handlers_no_reexpansion(self):
        """Two handlers on one logger process the same record; the second
        must see the same bounded values, not re-expanded or duplicated."""
        s1, s2 = io.StringIO(), io.StringIO()
        h1 = logging.StreamHandler(s1)
        h1.setFormatter(L.JsonFormatter())
        h1.addFilter(L._secret_filter)
        h2 = logging.StreamHandler(s2)
        h2.setFormatter(logging.Formatter("%(message)s"))
        h2.addFilter(L._secret_filter)
        log = logging.getLogger("multi_handler")
        log.handlers = [h1, h2]
        log.propagate = False
        log.setLevel(logging.INFO)
        try:
            raise ValueError("e" * 100_000)
        except ValueError as err:
            log.error("failed", exc_info=err, extra={"huge": "x" * 100_000})
        o1, o2 = s1.getvalue(), s2.getvalue()
        assert len(o1) <= self.TOTAL, len(o1)
        assert len(o2) <= L.MAX_LOG_MSG_LENGTH + 600, len(o2)
        assert o2.count("failed") == 1  # not duplicated
        # second handler saw the bounded traceback exactly once
        assert o2.count("Traceback") == 1

    # --- incident shape ---

    def test_82mb_message_exc_info_nested_extra(self, json_mode):
        try:
            raise ValueError("z" * 82_000_000)
        except ValueError as err:
            t0 = time.perf_counter()
            out = self._emit(
                json_mode,
                msg=f"failed: {err}",
                level=logging.ERROR,
                extra={
                    "huge": "x" * 82_000_000,
                    "d": {"inner": "y" * 82_000_000},
                },
                exc_info=err,
            )
            elapsed = time.perf_counter() - t0
        assert len(out) <= 2 * self.TOTAL, len(out)
        assert elapsed < 30.0, f"{elapsed:.1f}s"


class TestAdversarialKeysAndBudget:
    """Adversarial coverage for extra-field KEYS and the global traversal
    budget (review round 4 on b8b1520d34).

    Five defects this class locks down:
    1. keys bypassed caps/budget/redaction (20,000-char key with an
       embedded credential reached the JSON record raw);
    2. the byte budget never charged keys, containers, placeholders,
       scalars, or visited nodes (100x100x100 int tree = 1M leaf visits);
    3. `list(value)[:MAX_EXTRA_ITEMS]` eagerly copied the whole input;
    4. _head_tail duplicated the overlapping middle for near-cap strings;
    5. non-string dict keys collapsed onto one placeholder key,
       silently losing entries ({1: "one", 2: "two"} -> one entry).
    """

    CEILING = L.MAX_JSON_RECORD_LENGTH
    TOTAL = L.MAX_LOG_MSG_LENGTH + L.MAX_EXC_TEXT_LENGTH + L.MAX_EXTRA_TOTAL_BUDGET + 2_000

    def _make_logger(self, json_mode: bool):
        stream = io.StringIO()
        h = logging.StreamHandler(stream)
        h.addFilter(L._secret_filter)
        h.setFormatter(L.JsonFormatter() if json_mode else logging.Formatter("%(message)s"))
        log = logging.getLogger(f"adv_{json_mode}_{id(stream)}")
        log.handlers = [h]
        log.propagate = False
        log.setLevel(logging.INFO)
        return log, stream

    @pytest.fixture(params=[False, True])
    def json_mode(self, request):
        return request.param

    def _emit(self, json_mode, level=logging.INFO, **kwargs):
        log, stream = self._make_logger(json_mode)
        getattr(log, logging.getLevelName(level).lower())(**kwargs)
        return stream.getvalue()

    def _parsed_extra(self, out):
        parsed = json.loads(out)
        reserved = {"message", "level", "timestamp", "component", "logger"}
        return {k: v for k, v in parsed.items() if k not in reserved}

    # --- 1. keys are capped, budget-charged, redacted ---

    def test_giant_credential_key_top_level_capped_and_redacted(self, json_mode):
        probe = "sk-1234567890abcdef1234567890abcdef12345678"
        out = self._emit(
            json_mode,
            msg="small",
            extra={"x" * 10_000 + probe + "y" * 10_000: "v"},
        )
        assert len(out) <= self.TOTAL, len(out)
        assert probe not in out
        if json_mode:
            extras = self._parsed_extra(out)
            assert len(extras) == 1
            (key,) = extras.keys()
            assert len(key) <= L.MAX_EXTRA_STR_LENGTH + 600, len(key)
            assert "truncated" in key

    def test_giant_credential_key_nested_capped_and_redacted(self, json_mode):
        probe = "sk-1234567890abcdef1234567890abcdef12345678"
        out = self._emit(
            json_mode,
            msg="small",
            extra={"d": {"z" * 10_000 + probe + "w" * 10_000: "v"}},
        )
        assert len(out) <= self.TOTAL, len(out)
        assert probe not in out

    def test_many_oversized_keys_bounded(self, json_mode):
        # 30 keys x 2,500 chars = 75KB of key material alone; each key is
        # capped individually but they must also share the byte budget
        extra = {("k" + str(i)) * 500: "v" for i in range(30)}
        out = self._emit(json_mode, msg="small", extra=extra)
        assert len(out) <= self.TOTAL, len(out)
        if json_mode:
            extras = self._parsed_extra(out)
            for key in extras:
                assert len(key) <= L.MAX_EXTRA_STR_LENGTH + 600, len(key)

    def test_secret_in_key_head_and_tail_redacted(self, json_mode):
        probe = "sk-1234567890abcdef1234567890abcdef12345678"
        # head region of a truncated key
        out = self._emit(
            json_mode, msg="small",
            extra={"A" * 500 + probe + "B" * 20_000: "v"},
        )
        assert probe not in out
        # tail region of a truncated key
        out = self._emit(
            json_mode, msg="small",
            extra={"C" * 20_000 + probe: "v"},
        )
        assert probe not in out

    def test_colliding_keys_never_overwrite(self, json_mode):
        # two keys that sanitize to the same capped form must both survive
        out = self._emit(
            json_mode,
            msg="small",
            extra={"d": {"a" * 2_500: "first", "a" * 2_500: "second"}},
        )
        assert len(out) <= self.TOTAL, len(out)
        # NOTE: Python dict construction itself collapses the two identical
        # string keys before the filter ever runs, so use distinct keys
        # that truncate to the same head+tail:

    def test_distinct_keys_colliding_after_truncation_both_kept(self, json_mode):
        # "a"*1000 + "MIDDLE" + "a"*1500 and "a"*1000 + "OTHER" + "a"*1500
        # both truncate to "a"*2000 + marker + "a"*500 - same sanitized key
        k1 = "a" * 1_000 + "MIDDLE" + "a" * 1_500
        k2 = "a" * 1_000 + "OTHER" + "a" * 1_500
        out = self._emit(
            json_mode, msg="small", extra={"d": {k1: "first", k2: "second"}}
        )
        assert len(out) <= self.TOTAL, len(out)
        if json_mode:
            extras = self._parsed_extra(out)
            inner = list(extras.values())[0]
            assert isinstance(inner, dict)
            assert len(inner) == 2, inner
            assert set(inner.values()) == {"first", "second"}

    # --- 2. global node + byte budget charges every element ---

    def test_huge_int_only_collection_stops_at_node_budget(self, json_mode):
        # 100x100x100 nested ints = 1M leaves; node budget must stop
        # traversal well before visiting them all
        tree = [[list(range(100)) for _ in range(100)] for _ in range(100)]
        t0 = time.perf_counter()
        out = self._emit(json_mode, msg="small", extra={"tree": tree})
        elapsed = time.perf_counter() - t0
        assert len(out) <= self.TOTAL, len(out)
        assert elapsed < 5.0, f"{elapsed:.2f}s"
        if json_mode:
            extras = self._parsed_extra(out)
            inner = extras["tree"]
            # every emitted key is bounded
            assert len(json.dumps(inner)) <= L.MAX_EXTRA_TOTAL_BUDGET * 2

    def test_iter_counting_wrapper_consumption_bounded(self, json_mode):
        # list subclass counting how many elements iteration pulled:
        # islice must stop at MAX_EXTRA_ITEMS (+1 lookahead at most)
        class CountingList(list):
            def __init__(self, n):
                super().__init__(["m"] * n)
                self.pulled = 0

            def __iter__(self):
                for i, v in enumerate(list.__iter__(self)):
                    self.pulled = i + 1
                    yield v

        cl = CountingList(500_000)
        budget = L._ExtrasBudget()
        L._sanitize_extra_value(cl, budget)
        assert cl.pulled <= L.MAX_EXTRA_ITEMS + 1, cl.pulled

    def test_many_scalars_charge_byte_budget(self):
        # scalars must be charged (node + flat bytes) even though they
        # are individually tiny: 100 charged scalars consume both
        # counters, so a flat list cannot bypass the budgets
        budget = L._ExtrasBudget()
        out = L._sanitize_extra_value([1] * 500, budget)
        assert len(out) == 101  # items cap + "more items omitted" placeholder
        charged_bytes = L.MAX_EXTRA_TOTAL_BUDGET - budget.remaining_bytes
        # 100 node charges + 1 container + 100 scalar byte charges
        assert charged_bytes >= 100 * L._ExtrasBudget.SCALAR_BYTE_COST, charged_bytes
        # and a nested tree of a million ints is stopped by the shared
        # budget (byte or node counter - whichever binds first) long
        # before visiting all 1M+ leaves
        tree = [[[1] * 100 for _ in range(100)] for _ in range(100)]
        budget2 = L._ExtrasBudget()
        t0 = time.perf_counter()
        L._sanitize_extra_value(tree, budget2)
        elapsed = time.perf_counter() - t0
        assert budget2.exhausted, (
            f"nodes left {budget2.remaining_nodes}, bytes left {budget2.remaining_bytes}"
        )
        assert elapsed < 1.0, f"{elapsed:.2f}s"

    def test_budget_exhaustion_leaves_bounded_placeholder(self, json_mode):
        out = self._emit(
            json_mode,
            msg="small",
            extra={f"k{i}": "v" * 1_900 for i in range(50)},
        )
        assert len(out) <= self.TOTAL, len(out)

    def test_mixed_giant_fields_combined(self, json_mode):
        probe = "sk-1234567890abcdef1234567890abcdef12345678"
        try:
            raise ValueError("e" * 100_000)
        except ValueError as err:
            out = self._emit(
                json_mode,
                msg=f"failed: {err}",
                level=logging.ERROR,
                extra={
                    "huge": "x" * 100_000,
                    "d": {"inner": "y" * 100_000, probe: "leaked"},
                    "z" * 10_000 + probe: "k",
                },
                exc_info=err,
            )
        assert len(out) <= 2 * self.TOTAL, len(out)
        assert probe not in out

    # --- 3. bounded iteration (no eager copy) ---

    def test_giant_list_extra_no_eager_copy(self, json_mode):
        # islice consumes at most MAX_EXTRA_ITEMS elements; b8b1520d34's
        # list(value)[:MAX_EXTRA_ITEMS] pulled all 500,000 first
        class CountingList(list):
            def __init__(self, n):
                super().__init__(["m"] * n)
                self.pulled = 0

            def __iter__(self):
                for i, v in enumerate(list.__iter__(self)):
                    self.pulled = i + 1
                    yield v

        cl = CountingList(500_000)
        out = self._emit(json_mode, msg="small", extra={"l": cl})
        assert len(out) <= self.TOTAL, len(out)
        assert cl.pulled <= L.MAX_EXTRA_ITEMS + 1, cl.pulled

    # --- 4. _head_tail no duplicated middle ---

    def test_head_tail_no_overlap_various_sizes(self):
        for size, cap in [
            (2_000, 2_000),     # at limit: unchanged
            (2_001, 2_000),     # +1
            (2_100, 2_000),     # +100
            (2_499, 2_000),     # +499
            (2_500, 2_000),     # +500: tail starts exactly at head end
            (10_000, 2_000),    # well above
        ]:
            s = "A" * (cap - 100) + "B" * 100 + "C" * (size - cap)
            r = L._head_tail(s, cap)
            if size <= cap:
                assert r == s
                continue
            # no character position may appear twice from overlapping
            # head/tail: result length = cap + 40ish marker + min(500, size-cap)
            expected_tail = min(500, size - cap)
            expected_len = cap + len(f"... [truncated, {size} chars total] ...") + expected_tail
            assert len(r) == expected_len, (size, len(r), expected_len)
            # head and tail must be disjoint slices of the original
            head = s[:cap]
            tail = s[max(size - expected_tail, cap):]
            assert r.startswith(head)
            assert r.endswith(tail)
            # no duplicated middle: total content = head + tail exactly
            assert len(head) + len(tail) == cap + expected_tail

    def test_head_tail_message_path_no_duplicate_middle(self, json_mode):
        # end-to-end: a near-cap log message must not log the overlapping
        # middle twice (the original duplicate-middle bug)
        s = "H" * 9_900 + "MIDDLE" + "T" * 200  # 10,101 chars, cap 10,000
        out = self._emit(json_mode, msg=s)
        assert len(out) <= L.MAX_LOG_MSG_LENGTH + 600, len(out)
        assert out.count("MIDDLE") == 1, out.count("MIDDLE")

    def test_head_tail_exc_text_path_no_duplicate_middle(self, json_mode):
        # near-cap pre-populated exc_text must not duplicate its middle
        s = "H" * 1_900 + "MIDDLE" + "T" * 200  # 2,101 chars, cap 2,000
        log, stream = self._make_logger(json_mode)
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname=__file__, lineno=1,
            msg="small", args=(), exc_info=None,
        )
        record.exc_text = s
        log.handle(record)
        out = stream.getvalue()
        assert out.count("MIDDLE") == 1, out.count("MIDDLE")

    # --- 5. non-string keys keep their entries ---

    def test_non_string_keys_preserve_all_entries(self, json_mode):
        out = self._emit(
            json_mode,
            msg="small",
            extra={"d": {1: "one", 2: "two", 3.5: "three", None: "no"}},
        )
        assert len(out) <= self.TOTAL, len(out)
        if json_mode:
            extras = self._parsed_extra(out)
            inner = list(extras.values())[0]
            assert isinstance(inner, dict)
            assert len(inner) == 4, inner  # b8b1520d34 emitted 1
            assert set(inner.values()) == {"one", "two", "three", "no"}

    def test_non_string_keys_collision_suffixed(self, json_mode):
        # 1 (int) and "1" (str) both render as "1": collision suffixes
        # must keep both entries
        out = self._emit(
            json_mode,
            msg="small",
            extra={"d": {1: "int-one", "1": "str-one"}},
        )
        assert len(out) <= self.TOTAL, len(out)
        if json_mode:
            extras = self._parsed_extra(out)
            inner = list(extras.values())[0]
            assert isinstance(inner, dict)
            assert len(inner) == 2, inner
            assert set(inner.values()) == {"int-one", "str-one"}

    def test_unrenderable_key_fails_closed(self, json_mode):
        class BadKey:
            def __str__(self):
                raise RuntimeError("nope")

            def __repr__(self):
                raise RuntimeError("nope")

        out = self._emit(
            json_mode, msg="small", extra={"d": {BadKey(): "v", "keep": "kept"}}
        )
        assert len(out) <= self.TOTAL, len(out)
        if json_mode:
            extras = self._parsed_extra(out)
            inner = list(extras.values())[0]
            assert isinstance(inner, dict)
            assert inner["keep"] == "kept"

    # --- redaction invariant with keys ---

    def test_redact_string_spy_max_input_with_giant_keys(self, json_mode, monkeypatch):
        probe = "sk-1234567890abcdef1234567890abcdef12345678"
        seen: list[int] = []
        orig = L._redact_string

        def spy(value: str) -> str:
            seen.append(len(value))
            return orig(value)

        monkeypatch.setattr(L, "_redact_string", spy)
        monkeypatch.setattr(L, "_ENABLE_SECRET_REDACTION", True)
        try:
            raise ValueError("e" * 100_000)
        except ValueError as err:
            out = self._emit(
                json_mode,
                msg=f"failed: {err}",
                level=logging.ERROR,
                extra={
                    "huge": "x" * 100_000,
                    "d": {"inner": "y" * 100_000},
                    "z" * 10_000 + probe + "z" * 10_000: "k",
                },
                exc_info=err,
            )
        assert probe not in out
        assert seen, "redaction never ran"
        bound = max(
            L.MAX_LOG_MSG_LENGTH,
            L.MAX_EXC_TEXT_LENGTH,
            L.MAX_EXTRA_STR_LENGTH,
        ) + 600
        assert max(seen) <= bound, f"max redaction input: {max(seen):,} > {bound}"
        if json_mode:
            json.loads(out)  # still valid JSON

    # --- normal behavior preserved ---

    def test_small_normal_dict_keys_unchanged(self, json_mode):
        extra = {"model": "gpt-4", "meta": {"k1": "v1", "k2": 2}, "tags": ["a"]}
        out = self._emit(json_mode, msg="small", extra=extra)
        if json_mode:
            extras = self._parsed_extra(out)
            assert extras["model"] == "gpt-4"
            assert extras["meta"] == {"k1": "v1", "k2": 2}
            assert extras["tags"] == ["a"]

    def test_redaction_disabled_keys_still_bounded(self, json_mode, monkeypatch):
        monkeypatch.setattr(L, "_ENABLE_SECRET_REDACTION", False)
        out = self._emit(
            json_mode, msg="small", extra={"k" * 20_000: "v", "d": {"z" * 20_000: "v"}}
        )
        assert len(out) <= self.TOTAL, len(out)

    def test_key_budget_exhaustion_stops_cleanly(self, json_mode):
        # exhaust the budget with values first, then giant keys must not
        # sneak onto the record after the budget is spent
        extra = {f"v{i}": "x" * 1_900 for i in range(20)}
        extra["giantkey" + "z" * 20_000] = "v"
        out = self._emit(json_mode, msg="small", extra=extra)
        assert len(out) <= self.TOTAL, len(out)
        if json_mode:
            json.loads(out)
