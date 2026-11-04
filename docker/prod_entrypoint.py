import os
import resource
import sys

CORE_LIMIT_BYTES = 6291456 * 1024


def apply_core_limit() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (CORE_LIMIT_BYTES, CORE_LIMIT_BYTES))
        return
    except (ValueError, OSError) as e:
        target = hard if hard == resource.RLIM_INFINITY else min(CORE_LIMIT_BYTES, hard)
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (target, hard))
        except (ValueError, OSError):
            pass
        print(f"prod_entrypoint: RLIMIT_CORE hard={hard}, could not raise: {e}", file=sys.stderr)


def main(argv: tuple[str, ...]) -> None:
    apply_core_limit()
    if os.environ.get("USE_DDTRACE") == "true":
        os.environ["DD_TRACE_OPENAI_ENABLED"] = "False"
        os.execvp("ddtrace-run", ("ddtrace-run", "litellm", *argv))
    os.execvp("litellm", ("litellm", *argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
