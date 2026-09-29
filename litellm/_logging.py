import ast
import logging
import os
import sys
from datetime import datetime
from itertools import islice
from logging import Formatter
from typing import Any, Dict, Optional

from litellm.litellm_core_utils.secret_redaction import redact_string
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads

set_verbose = False

if set_verbose is True:
    logging.warning(
        "`litellm.set_verbose` is deprecated. Please set `os.environ['LITELLM_LOG'] = 'DEBUG'` for debug logs."
    )

_ENABLE_SECRET_REDACTION = os.getenv("LITELLM_DISABLE_REDACT_SECRETS", "").lower() != "true"

# ---------------------------------------------------------------------------
# Central logging size limits.
#
# These exist because provider validation errors can echo a full request
# body (observed: 82MB pydantic errors from an oversized image request);
# logging such a value unbounded blocks the event loop for the duration
# of the synchronous redaction regex scan and the stdout write (observed
# 9.5-minute freeze). Every limit below is a CONTENT limit on what a log
# record may carry into redaction and serialization; the final record
# after serialization additionally must fit MAX_JSON_RECORD_LENGTH
# (final-output limit, enforced by JsonFormatter).
# ---------------------------------------------------------------------------

# Maximum rendered log message (record.getMessage()) length. Content limit.
MAX_LOG_MSG_LENGTH = 10_000
# Maximum formatted exception/traceback (formatException output, exc_text,
# or stack_info text). Content limit.
MAX_EXC_TEXT_LENGTH = 2_000
# Maximum length of any single string supplied via logger extra={...},
# at any nesting depth. Content limit.
MAX_EXTRA_STR_LENGTH = 2_000
# Maximum nesting depth accepted for structured extra values before the
# value is replaced by a bounded placeholder. Content limit.
MAX_EXTRA_DEPTH = 6
# Maximum number of items (dict entries / sequence elements) accepted per
# collection level in structured extras. Content limit.
MAX_EXTRA_ITEMS = 100
# Maximum total serialized size of all extra fields combined. Content limit
# on the sanitized extras structure before it is attached to the record.
MAX_EXTRA_TOTAL_BUDGET = 20_000
# Maximum number of elements (dict keys, dict values, sequence items,
# scalars, placeholders) visited while sanitizing extras across all extra
# fields combined. Bounds traversal time independent of content size: a
# deeply nested tree of 1M tiny values is capped by node count, not just
# by bytes. Content limit.
MAX_EXTRA_TOTAL_NODES = 20_000
# Maximum final serialized JSON log record. Final-output limit: if the
# normally-serialized record exceeds this, JsonFormatter emits a valid
# JSON fallback object instead.
MAX_JSON_RECORD_LENGTH = 100_000


def _head_tail(s: str, max_length: int) -> str:
    """Deterministic head+tail truncation with original length reported.

    The tail start is clamped to never overlap the head, so strings just
    over the cap do not log the overlapping middle twice.
    """
    if len(s) <= max_length:
        return s
    head = s[:max_length]
    tail_length = min(500, max_length)
    # never reach back into the head - otherwise strings just over the
    # cap would log the overlapping middle twice
    tail_start = max(len(s) - tail_length, max_length)
    return (
        f"{head}"
        f"... [truncated, {len(s)} chars total] ..."
        f"{s[tail_start:]}"
    )


def _redact_string(value: str) -> str:
    if not _ENABLE_SECRET_REDACTION:
        return value
    return redact_string(value)


class _ExtrasBudget:
    """Shared byte + node budget across every extra field on a record.

    Both counters are charged for every element visited while
    sanitizing: each value, each dict key, and each scalar costs one
    node (bounding traversal time for trees of millions of tiny values)
    plus its rendered size in bytes (bounding total content). Exhausting
    either counter stops traversal immediately.
    """

    __slots__ = ("remaining_bytes", "remaining_nodes")

    # flat byte cost for a non-string scalar (bool/int/float/None) or a
    # container node, approximating its JSON serialization overhead so
    # combined small values cannot bypass the byte budget
    SCALAR_BYTE_COST = 16

    def __init__(self) -> None:
        self.remaining_bytes = MAX_EXTRA_TOTAL_BUDGET
        self.remaining_nodes = MAX_EXTRA_TOTAL_NODES

    @property
    def exhausted(self) -> bool:
        return self.remaining_bytes <= 0 or self.remaining_nodes <= 0

    def charge_node(self) -> bool:
        """Charge one visited element. Returns False when exhausted."""
        if self.remaining_nodes <= 0:
            return False
        self.remaining_nodes -= 1
        return True

    def charge_bytes(self, n: int) -> bool:
        """Charge n rendered bytes. Returns False when they do not fit.

        Saturates at zero so subsequent charges fail fast.
        """
        if self.remaining_bytes - n < 0:
            self.remaining_bytes = 0
            return False
        self.remaining_bytes -= n
        return True


def _sanitize_extra_key(
    key: Any, budget: "_ExtrasBudget", seen_keys: set
) -> Optional[str]:
    """Sanitize a dict key destined for the emitted record.

    Keys bypassed every cap in earlier versions: a 20,000-char key with
    an embedded credential reached the JSON record raw. Keys are now
    rendered (str() for non-string keys, so {1: "one", 2: "two"} keeps
    both entries instead of collapsing onto one placeholder), head+tail
    capped, budget-charged (node + bytes), and redacted when redaction
    is enabled.

    Collision-safe: if the sanitized key already exists in `seen_keys`,
    a "#2"/"#3"... suffix is appended so entries are never silently
    overwritten, and a sanitized key never collides with a standard
    LogRecord attribute (which would be dropped by downstream
    formatters). Returns None when the budget is exhausted.
    """
    if not budget.charge_node():
        return None
    if isinstance(key, str):
        rendered = key
    else:
        try:
            rendered = str(key)
        except Exception:
            rendered = "<non-str key unrenderable>"
    bounded = _head_tail(rendered, MAX_EXTRA_STR_LENGTH)
    if not budget.charge_bytes(len(bounded)):
        return None
    safe_key = _redact_string(bounded)
    if safe_key in _STANDARD_RECORD_ATTRS:
        safe_key = f"extra_{safe_key}"
    if safe_key in seen_keys:
        base = safe_key
        i = 2
        while f"{base}#{i}" in seen_keys:
            i += 1
        safe_key = f"{base}#{i}"
    seen_keys.add(safe_key)
    return safe_key


def _sanitize_extra_value(
    value: Any,
    budget: "_ExtrasBudget",
    depth: int = 0,
    _seen: Optional[set] = None,
) -> Any:
    """Recursively bound an extra value into a JSON-safe, size-capped copy.

    - strings are head+tail capped to MAX_EXTRA_STR_LENGTH, then redacted
      when redaction is enabled (never handed to the redaction regex
      unbounded);
    - dict KEYS are sanitized via _sanitize_extra_key (capped, redacted,
      budget-charged, collision-safe);
    - dicts/lists/tuples/sets are copied (caller-owned objects are never
      mutated) with depth capped at MAX_EXTRA_DEPTH, item count capped
      at MAX_EXTRA_ITEMS, and iteration bounded via islice so the input
      is never eagerly copied;
    - every element visited - containers, keys, scalars, placeholders -
      is charged against the shared budget's node and byte counters, so
      a tree of a million tiny values stops traversal at
      MAX_EXTRA_TOTAL_NODES instead of walking every leaf;
    - cycles are replaced by a placeholder instead of recursing forever;
    - unrenderable objects (raising __str__/__repr__) fail closed to a
      bounded placeholder.

    Values exceeding the budget, depth, or item caps are replaced by
    deterministic bounded placeholders carrying the original type and,
    where known, the original length/size.
    """
    if _seen is None:
        _seen = set()

    def _placeholder(reason: str, detail: str = "") -> str:
        return f"<{reason}{detail}>"

    # every visited element costs one node - this is what bounds
    # traversal time for huge trees of tiny values
    if not budget.charge_node():
        return _placeholder("extra budget exceeded")

    # depth cap
    if depth > MAX_EXTRA_DEPTH:
        return _placeholder("max depth exceeded")

    # cycle guard (id-based; mirrors safe_json_dumps but on the copy path)
    if isinstance(value, (dict, list, tuple, set)):
        if id(value) in _seen:
            return _placeholder("circular reference")
        _seen = _seen | {id(value)}

    if isinstance(value, str):
        bounded = _head_tail(value, MAX_EXTRA_STR_LENGTH)
        if not budget.charge_bytes(len(bounded)):
            return _placeholder("extra budget exceeded", f", {len(value)} chars omitted")
        return _redact_string(bounded)

    if isinstance(value, bytes):
        try:
            decoded = value.decode("utf-8", errors="replace")
        except Exception:
            decoded = "<undecodable bytes>"
        bounded = _head_tail(decoded, MAX_EXTRA_STR_LENGTH)
        if not budget.charge_bytes(len(bounded)):
            return _placeholder("extra budget exceeded", f", {len(value)} bytes omitted")
        return _redact_string(bounded)

    if isinstance(value, bool) or value is None:
        # scalars are charged (node above + flat bytes here) so combined
        # small values cannot bypass the byte budget
        budget.charge_bytes(_ExtrasBudget.SCALAR_BYTE_COST)
        return value

    if isinstance(value, (int, float)):
        budget.charge_bytes(_ExtrasBudget.SCALAR_BYTE_COST)
        return value

    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        seen_keys: set = set()
        processed = 0
        # islice: never materialize the full input; stop pulling items
        # the moment the budget is exhausted
        for k, v in islice(value.items(), MAX_EXTRA_ITEMS):
            key = _sanitize_extra_key(k, budget, seen_keys)
            if key is None:
                break
            out[key] = _sanitize_extra_value(v, budget, depth + 1, _seen)
            processed += 1
            if budget.exhausted:
                break
        omitted = len(value) - processed
        if omitted > 0:
            marker = _placeholder(f"{omitted} more keys omitted")
            if "__truncated__" in out:
                out["__truncated__#2"] = marker
            else:
                out["__truncated__"] = marker
        return out

    if isinstance(value, (list, tuple, set)):
        out_list: list = []
        processed = 0
        for v in islice(iter(value), MAX_EXTRA_ITEMS):
            out_list.append(_sanitize_extra_value(v, budget, depth + 1, _seen))
            processed += 1
            if budget.exhausted:
                break
        omitted = len(value) - processed
        if omitted > 0:
            out_list.append(_placeholder(f"{omitted} more items omitted"))
        # preserve tuple type for small normal values; sets -> list (JSON)
        if isinstance(value, tuple):
            return tuple(out_list)
        return out_list

    # other object types: bound their str() representation
    try:
        rendered = str(value)
    except Exception:
        return _placeholder("unrenderable object", f" ({type(value).__name__})")
    bounded = _head_tail(rendered, MAX_EXTRA_STR_LENGTH)
    if not budget.charge_bytes(len(bounded)):
        return _placeholder("extra budget exceeded", f", {type(value).__name__} omitted")
    return _redact_string(bounded)


def redact_secrets(value: str) -> str:
    """Public API: redact known secret/credential patterns from an arbitrary string.

    Use this for code paths that bypass the logging system — e.g. Slack/Teams
    alerting, HTTP error response bodies, or any other string that may contain
    secrets and will be sent to an external sink.

    Not to be confused with redact_message_input_output_from_logging() in
    litellm_core_utils/redact_messages.py, which redacts LLM prompt/response
    content for privacy — this function redacts credential patterns (API keys,
    PEM blocks, tokens, etc.) by shape.
    """
    if not _ENABLE_SECRET_REDACTION:
        return value
    return _redact_string(value)


class SecretRedactionFilter(logging.Filter):
    """Scrubs known secret/credential patterns from log records.

    Also the logging size choke point: no field on a record passing this
    filter (message, args, exc_info, exc_text, stack_info, or extra={...}
    attributes) may remain large enough to freeze the event loop in the
    redaction regex or the stdout write.
    """

    _formatter = logging.Formatter()

    @staticmethod
    def _cap_str(s: str) -> str:
        """Cap a rendered log message (module-level MAX_LOG_MSG_LENGTH)."""
        return _head_tail(s, MAX_LOG_MSG_LENGTH)

    def _cap_exc_text(self, record: logging.LogRecord) -> None:
        """Cap the formatted exception (traceback + str(e)) on the record.

        formatException() embeds the full str(e) on its last line, so an
        unbounded exception makes an unbounded traceback that this filter
        and every downstream handler would format, regex-scan, and emit.
        Also caps a pre-populated giant record.exc_text (attached by
        caller code without exc_info) and giant record.stack_info.

        Fails closed: if formatException() itself raises, we still detach
        exc_info (and set a placeholder exc_text) so a broken formatter
        cannot make the record emit the unbounded exception.
        """
        if record.exc_info and record.exc_info[1] is not None:
            try:
                exc_text = self._formatter.formatException(record.exc_info)
                exc_text = _head_tail(exc_text, MAX_EXC_TEXT_LENGTH)
            except Exception:
                # fail closed: never leave the raw exc_info attached
                exc_text = (
                    "<exception formatting failed; traceback omitted; "
                    f"{type(record.exc_info[1]).__name__}>"
                )
            record.exc_text = exc_text
            # prevent the formatter from re-formatting/re-appending the
            # untruncated exception on top of our capped exc_text
            record.exc_info = None
        elif isinstance(record.exc_text, str):
            record.exc_text = _head_tail(record.exc_text, MAX_EXC_TEXT_LENGTH)

        if isinstance(record.stack_info, str):
            record.stack_info = _head_tail(record.stack_info, MAX_EXC_TEXT_LENGTH)

    def _sanitize_extras(self, record: logging.LogRecord) -> None:
        """Bound and redact extra={...} fields (non-standard record attrs).

        Runs before the redaction early-return so extras are bounded in
        every mode. Values are replaced with bounded copies; caller-owned
        structures are never mutated. Keys are sanitized too - a giant or
        credential-bearing key is capped, redacted, and made
        collision-safe, and the unsafe original key is removed from the
        record. Redaction (when enabled) only ever sees bounded strings -
        the redaction regex never receives input above
        MAX_EXTRA_STR_LENGTH from extras.
        """
        budget = _ExtrasBudget()
        seen_keys: set = set()
        additions: Dict[str, Any] = {}
        for key, value in list(record.__dict__.items()):
            if key in _STANDARD_RECORD_ATTRS:
                continue
            try:
                safe_key = _sanitize_extra_key(key, budget, seen_keys)
                if safe_key is None:
                    # budget exhausted before this key could be charged
                    if key in record.__dict__:
                        del record.__dict__[key]
                    continue
                additions[safe_key] = _sanitize_extra_value(value, budget)
                if safe_key != key:
                    # never leave the unsafe original key on the record
                    del record.__dict__[key]
            except Exception:
                # fail closed to a bounded placeholder
                if key in record.__dict__:
                    del record.__dict__[key]
                ph = "<extra field unrenderable>"
                if ph in additions:
                    i = 2
                    while f"{ph}#{i}" in additions:
                        i += 1
                    ph = f"{ph}#{i}"
                additions[ph] = "<extra field unrenderable>"
        record.__dict__.update(additions)

    def filter(self, record: logging.LogRecord) -> bool:
        # Bound everything BEFORE the redaction early-return: the
        # giant-log-line freeze is a size problem, not a secrecy problem,
        # so all caps apply even when secret redaction is disabled.
        self._cap_exc_text(record)
        self._sanitize_extras(record)

        # Render the message once, cap it, then redact every byte that
        # will actually be emitted. Capping BEFORE redaction keeps the
        # synchronous redaction regex off unbounded input (the incident
        # freeze) without bypassing it: redaction still scans the full
        # bounded message that reaches the log.
        try:
            bounded_msg = self._cap_str(record.getMessage())
        except Exception:
            bounded_msg = "<unrenderable log message>"

        if not _ENABLE_SECRET_REDACTION:
            record.msg = bounded_msg
            record.args = None
            return True

        try:
            record.msg = _redact_string(bounded_msg)
            record.args = None
        except Exception:
            if isinstance(bounded_msg, str):
                record.msg = _redact_string(bounded_msg)

        # Redact exception tracebacks (already capped above)
        if record.exc_text is not None:
            try:
                record.exc_text = _redact_string(record.exc_text)
            except Exception:
                pass

        if record.stack_info is not None:
            try:
                record.stack_info = _redact_string(record.stack_info)
            except Exception:
                pass

        return True


_secret_filter = SecretRedactionFilter()


json_logs = bool(os.getenv("JSON_LOGS", False))
# Create a handler for the logger (you may need to adapt this based on your needs)
log_level = os.getenv("LITELLM_LOG", "DEBUG")
numeric_level: str = getattr(logging, log_level.upper())
handler = logging.StreamHandler()
handler.setLevel(numeric_level)
handler.addFilter(_secret_filter)


def _try_parse_json_message(message: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse a log message as JSON. Returns parsed dict if valid, else None.
    Handles messages that are entirely valid JSON (e.g. json.dumps output).
    Uses shared safe_json_loads for consistent error handling.
    """
    if not message or not isinstance(message, str):
        return None
    msg_stripped = message.strip()
    if not (msg_stripped.startswith("{") or msg_stripped.startswith("[")):
        return None
    parsed = safe_json_loads(message, default=None)
    if parsed is None or not isinstance(parsed, dict):
        return None
    return parsed


def _try_parse_embedded_python_dict(message: str) -> Optional[Dict[str, Any]]:
    """
    Try to find and parse a Python dict repr (e.g. str(d) or repr(d)) embedded in
    the message. Handles patterns like:
    "get_available_deployment for model: X, Selected deployment: {'model_name': '...', ...} for model: X"
    Uses ast.literal_eval for safe parsing. Returns the parsed dict or None.
    """
    if not message or not isinstance(message, str) or "{" not in message:
        return None
    i = 0
    while i < len(message):
        start = message.find("{", i)
        if start == -1:
            break
        depth = 0
        for j in range(start, len(message)):
            c = message[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    substr = message[start : j + 1]
                    try:
                        result = ast.literal_eval(substr)
                        if isinstance(result, dict) and len(result) > 0:
                            return result
                    except (ValueError, SyntaxError, TypeError):
                        pass
                    break
        i = start + 1
    return None


# Standard LogRecord attribute names - used to identify 'extra' fields.
# Derived at runtime so we automatically include version-specific attrs (e.g. taskName).
def _get_standard_record_attrs() -> frozenset:
    """Standard LogRecord attribute names - excludes extra keys from logger.debug(..., extra={...})."""
    return frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


_STANDARD_RECORD_ATTRS = _get_standard_record_attrs()


class JsonFormatter(Formatter):
    def __init__(self):
        super(JsonFormatter, self).__init__()

    def formatTime(self, record, datefmt=None):
        # Use datetime to format the timestamp in ISO 8601 format
        dt = datetime.fromtimestamp(record.created)
        return dt.isoformat()

    def format(self, record):
        message_str = record.getMessage()
        json_record: Dict[str, Any] = {
            "message": message_str,
            "level": record.levelname,
            "timestamp": self.formatTime(record),
        }

        # Parse embedded JSON or Python dict repr in message so sub-fields become first-class properties
        parsed = _try_parse_json_message(message_str)
        if parsed is None:
            parsed = _try_parse_embedded_python_dict(message_str)
        if parsed is not None:
            for key, value in parsed.items():
                if key not in json_record:
                    json_record[key] = value

        # Include extra attributes passed via logger.debug("msg", extra={...})
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key not in json_record:
                json_record[key] = value

        # Set component/logger only if not already supplied via extra={...}
        if "component" not in json_record:
            json_record["component"] = record.name
        if "logger" not in json_record:
            json_record["logger"] = f"{record.filename}:{record.lineno}"

        # exc_info is cleared (but exc_text set) by SecretRedactionFilter
        # after truncating the formatted exception, so check both
        if record.exc_info or record.exc_text:
            json_record["stacktrace"] = (
                record.exc_text or self.formatException(record.exc_info)
            )

        rendered = safe_dumps(json_record)
        # Final-output guard: if the serialized record somehow exceeds the
        # hard ceiling (e.g. many individually-bounded extras, or a message
        # parsed into many first-class fields), emit a valid JSON fallback
        # object - never a sliced invalid JSON string.
        if len(rendered) > MAX_JSON_RECORD_LENGTH:
            fallback: Dict[str, Any] = {
                "message": _head_tail(message_str, MAX_EXC_TEXT_LENGTH),
                "level": record.levelname,
                "timestamp": self.formatTime(record),
                "component": record.name,
                "logger": f"{record.filename}:{record.lineno}",
                "record_truncated": True,
                "original_length": len(rendered),
            }
            if json_record.get("stacktrace"):
                fallback["stacktrace"] = _head_tail(
                    str(json_record["stacktrace"]), MAX_EXC_TEXT_LENGTH
                )
            extra_keys = [
                k for k in json_record if k not in fallback
            ]
            if extra_keys:
                fallback["omitted_extra_keys"] = ", ".join(
                    sorted(str(k) for k in extra_keys)
                )[:MAX_EXC_TEXT_LENGTH]
            try:
                rendered = safe_dumps(fallback)
            except Exception:
                rendered = '{"record_truncated": true}'
            # deterministic last reduction: even the fallback must fit
            if len(rendered) > MAX_JSON_RECORD_LENGTH:
                core = {
                    "message": _head_tail(message_str, 500),
                    "level": record.levelname,
                    "record_truncated": True,
                    "original_length": len(rendered),
                }
                rendered = safe_dumps(core)
        return rendered


# Function to set up exception handlers for JSON logging
def _setup_json_exception_handlers(formatter):
    # Create a handler with JSON formatting for exceptions
    error_handler = logging.StreamHandler()
    error_handler.setFormatter(formatter)
    error_handler.addFilter(_secret_filter)

    # Setup excepthook for uncaught exceptions
    def json_excepthook(exc_type, exc_value, exc_traceback):
        record = logging.LogRecord(
            name="LiteLLM",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg=str(exc_value),
            args=(),
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        error_handler.handle(record)

    sys.excepthook = json_excepthook

    # Configure asyncio exception handler if possible
    try:
        import asyncio

        def async_json_exception_handler(loop, context):
            exception = context.get("exception")
            if exception:
                exc_type = type(exception)
                record = logging.LogRecord(
                    name="LiteLLM",
                    level=logging.ERROR,
                    pathname="",
                    lineno=0,
                    msg=str(exception),
                    args=(),
                    exc_info=(exc_type, exception, exception.__traceback__),
                )
                error_handler.handle(record)
            else:
                loop.default_exception_handler(context)

        asyncio.get_event_loop().set_exception_handler(async_json_exception_handler)
    except Exception:
        pass


# Create a formatter and set it for the handler
if json_logs:
    handler.setFormatter(JsonFormatter())
    _setup_json_exception_handlers(JsonFormatter())
else:
    formatter = logging.Formatter(
        "\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    handler.setFormatter(formatter)

verbose_proxy_logger = logging.getLogger("LiteLLM Proxy")
verbose_router_logger = logging.getLogger("LiteLLM Router")
verbose_logger = logging.getLogger("LiteLLM")

# Add the handler to the loggers
verbose_router_logger.addHandler(handler)
verbose_proxy_logger.addHandler(handler)
verbose_logger.addHandler(handler)


def _suppress_loggers():
    """Suppress noisy loggers at INFO level"""
    # Suppress httpx request logging at INFO level
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    # Suppress APScheduler logging at INFO level
    apscheduler_executors_logger = logging.getLogger("apscheduler.executors.default")
    apscheduler_executors_logger.setLevel(logging.WARNING)
    apscheduler_scheduler_logger = logging.getLogger("apscheduler.scheduler")
    apscheduler_scheduler_logger.setLevel(logging.WARNING)


# Call the suppression function
_suppress_loggers()

ALL_LOGGERS = [
    logging.getLogger(),
    verbose_logger,
    verbose_router_logger,
    verbose_proxy_logger,
]


def _get_loggers_to_initialize():
    """
    Get all loggers that should be initialized with the JSON handler.

    Includes third-party integration loggers (like langfuse) if they are
    configured as callbacks.
    """
    import litellm

    loggers = list(ALL_LOGGERS)

    # Add langfuse logger if langfuse is being used as a callback
    langfuse_callbacks = {"langfuse", "langfuse_otel"}
    all_callbacks = set(litellm.success_callback + litellm.failure_callback)
    if langfuse_callbacks & all_callbacks:
        loggers.append(logging.getLogger("langfuse"))

    return loggers


def _initialize_loggers_with_handler(handler: logging.Handler):
    """
    Initialize all loggers with a handler

    - Adds a handler to each logger
    - Prevents bubbling to parent/root (critical to prevent duplicate JSON logs)
    """
    handler.addFilter(_secret_filter)
    for lg in _get_loggers_to_initialize():
        lg.handlers.clear()  # remove any existing handlers
        lg.addHandler(handler)  # add JSON formatter handler
        lg.propagate = False  # prevent bubbling to parent/root


def _get_uvicorn_json_log_config():
    """
    Generate a uvicorn log_config dictionary that applies JSON formatting to all loggers.

    This ensures that uvicorn's access logs, error logs, and all application logs
    are formatted as JSON when json_logs is enabled.
    """
    json_formatter_class = "litellm._logging.JsonFormatter"

    # Use the module-level log_level variable for consistency
    uvicorn_log_level = log_level.upper()

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": json_formatter_class,
            },
            "default": {
                "()": json_formatter_class,
            },
            "access": {
                "()": json_formatter_class,
            },
        },
        "handlers": {
            "default": {
                "formatter": "json",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": uvicorn_log_level,
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["default"],
                "level": uvicorn_log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": uvicorn_log_level,
                "propagate": False,
            },
        },
    }

    return log_config


def _turn_on_json():
    """
    Turn on JSON logging

    - Adds a JSON formatter to all loggers
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    _initialize_loggers_with_handler(handler)
    # Set up exception handlers
    _setup_json_exception_handlers(JsonFormatter())


def _turn_on_debug():
    verbose_logger.setLevel(level=logging.DEBUG)  # set package log to debug
    verbose_router_logger.setLevel(level=logging.DEBUG)  # set router logs to debug
    verbose_proxy_logger.setLevel(level=logging.DEBUG)  # set proxy logs to debug


def _disable_debugging():
    """Disable the package, router, and proxy verbose loggers."""
    verbose_logger.disabled = True
    verbose_router_logger.disabled = True
    verbose_proxy_logger.disabled = True


def _enable_debugging():
    verbose_logger.disabled = False
    verbose_router_logger.disabled = False
    verbose_proxy_logger.disabled = False


def print_verbose(print_statement):
    try:
        if set_verbose:
            print(redact_secrets(str(print_statement)))  # noqa: T201
    except Exception:
        pass


def _is_debugging_on() -> bool:
    """
    Returns True if debugging is on
    """
    return verbose_logger.isEnabledFor(logging.DEBUG) or set_verbose is True
