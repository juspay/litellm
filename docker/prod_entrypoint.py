import os
import resource
import sys

CORE_LIMIT_BYTES = 6291456 * 1024


def apply_core_limit() -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (CORE_LIMIT_BYTES, CORE_LIMIT_BYTES))
    except (ValueError, OSError):
        pass


def main(argv: tuple[str, ...]) -> None:
    apply_core_limit()
    if os.environ.get("USE_DDTRACE") == "true":
        os.environ["DD_TRACE_OPENAI_ENABLED"] = "False"
        os.execvp("ddtrace-run", ("ddtrace-run", "litellm", *argv))
    os.execvp("litellm", ("litellm", *argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
