import sys
import types
from types import SimpleNamespace

from litellm.llms.litellm_proxy.skills.sandbox_executor import SkillsSandboxExecutor


class _FakeSandboxSession:
    last_instance = None
    install_exit_code = 0
    install_stdout = ""
    install_stderr = ""
    exec_exit_code = 0
    exec_stdout = "ok"
    exec_stderr = ""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.copy_to_runtime_calls = []
        self.run_calls = []
        self.copied_contents = {}
        type(self).last_instance = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def copy_to_runtime(self, local_path, sandbox_path):
        self.copy_to_runtime_calls.append((local_path, sandbox_path))
        with open(local_path, "rb") as f:
            self.copied_contents[sandbox_path] = f.read()

    def run(self, code):
        self.run_calls.append(code)
        if "pip', 'install', '-r'" in code:
            return SimpleNamespace(
                exit_code=self.install_exit_code,
                stdout=self.install_stdout,
                stderr=self.install_stderr,
            )
        return SimpleNamespace(
            exit_code=self.exec_exit_code,
            stdout=self.exec_stdout,
            stderr=self.exec_stderr,
        )


def _install_fake_sandbox(monkeypatch, session_cls=_FakeSandboxSession):
    fake_module = types.SimpleNamespace(SandboxSession=session_cls)
    monkeypatch.setitem(sys.modules, "llm_sandbox", fake_module)


def test_execute_installs_inline_requirements_file(monkeypatch):
    _install_fake_sandbox(monkeypatch)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(executor, "_collect_generated_files", lambda *args, **kwargs: [])

    requirements = "requests==2.32.3\nhttpx>=0.27\n# a comment line\n\n"
    result = executor.execute(
        code="print('hello')",
        skill_files={"pkg/__init__.py": b""},
        requirements=requirements,
    )

    assert result["success"] is True

    created_session = _FakeSandboxSession.last_instance
    assert created_session.copied_contents["/sandbox/.litellm_requirements.txt"] == requirements.encode("utf-8")
    assert "pip', 'install', '-r', '.litellm_requirements.txt'" in created_session.run_calls[0]
    assert "os.chdir('/sandbox')" in created_session.run_calls[1]


def test_execute_uses_skill_requirements_txt(monkeypatch):
    """Regression (requirements swap): pip must install from the validated
    bytes staged at the canonical .litellm_requirements.txt path, never from a
    skill-supplied requirements.txt that the skill controls."""
    _install_fake_sandbox(monkeypatch)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(executor, "_collect_generated_files", lambda *args, **kwargs: [])

    requirements = b"requests==2.32.3\n"
    result = executor.execute(
        code="print('hello')",
        skill_files={
            "requirements.txt": requirements,
            "main.py": b"print('x')",
        },
    )

    assert result["success"] is True

    created_session = _FakeSandboxSession.last_instance
    assert created_session.copied_contents["/sandbox/.litellm_requirements.txt"] == requirements
    assert "pip', 'install', '-r', '.litellm_requirements.txt'" in created_session.run_calls[0]
    run_code_for_requirements = [c for c in created_session.run_calls if "requirements" in c]
    assert all("'.litellm_requirements.txt'" in c for c in run_code_for_requirements)


def test_execute_returns_install_failure(monkeypatch):
    class _FailingSandboxSession(_FakeSandboxSession):
        install_exit_code = 1
        install_stdout = "pip output"
        install_stderr = "install failed"
        last_instance = None

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            type(self).last_instance = self

    _install_fake_sandbox(monkeypatch, session_cls=_FailingSandboxSession)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(executor, "_collect_generated_files", lambda *args, **kwargs: [])

    result = executor.execute(
        code="print('hello')",
        skill_files={"main.py": b"print('x')"},
        requirements="package==1.0.0\n",
    )

    assert result == {
        "success": False,
        "output": "pip output",
        "error": "install failed",
        "files": [],
    }
    assert len(_FailingSandboxSession.last_instance.run_calls) == 1


def test_execute_rejects_unsafe_requirements(monkeypatch):
    _install_fake_sandbox(monkeypatch)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(executor, "_collect_generated_files", lambda *args, **kwargs: [])

    for bad in [
        "-e ./pkg",
        "--editable ./pkg",
        "-i https://evil.example.com/simple",
        "--index-url https://evil.example.com/simple",
        "--extra-index-url https://evil.example.com/simple",
        "--find-links /tmp/wheels",
        "-f /tmp/wheels",
        "--trusted-host evil.example.com",
        "--proxy http://evil.example.com:8080",
        "git+https://evil.example.com/repo.git",
        "svn+https://evil.example.com/repo",
        "https://evil.example.com/pkg.whl",
        "./local/pkg",
        "/abs/path/pkg",
        "pkg --index-url https://evil.example.com/simple",
        "pkg --extra-index-url https://evil.example.com/simple",
        "pkg --find-links /tmp/wheels",
        "pkg --trusted-host evil.example.com",
        "pkg --hash=sha256:deadbeef",
        "pkg -i https://evil.example.com/simple",
        "[options]",
        "[global]",
        "pkg @ https://evil.example.com/pkg.whl",
        "-r other-requirements.txt",
    ]:
        result = executor.execute(
            code="print('hello')",
            skill_files={"main.py": b"print('x')"},
            requirements=bad + "\n",
        )
        assert result["success"] is False, bad
        assert "Disallowed requirements" in result["error"], bad


def test_execute_rejects_unsafe_bundled_requirements_txt(monkeypatch):
    _install_fake_sandbox(monkeypatch)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(executor, "_collect_generated_files", lambda *args, **kwargs: [])

    result = executor.execute(
        code="print('hello')",
        skill_files={
            "requirements.txt": b"git+https://evil.example.com/repo.git\n",
            "main.py": b"print('x')",
        },
    )

    assert result["success"] is False
    assert "Disallowed requirements" in result["error"]


def test_execute_network_mode_none_by_default(monkeypatch):
    _install_fake_sandbox(monkeypatch)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(executor, "_collect_generated_files", lambda *args, **kwargs: [])
    monkeypatch.delenv("LITELLM_SANDBOX_NETWORK_MODE", raising=False)

    executor.execute(
        code="print('hello')",
        skill_files={"main.py": b"print('x')"},
    )

    assert _FakeSandboxSession.last_instance.kwargs["runtime_configs"] == {"network_mode": "none"}


def test_execute_network_mode_bridge_when_requirements_need_install(monkeypatch):
    _install_fake_sandbox(monkeypatch)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(executor, "_collect_generated_files", lambda *args, **kwargs: [])
    monkeypatch.delenv("LITELLM_SANDBOX_NETWORK_MODE", raising=False)

    executor.execute(
        code="print('hello')",
        skill_files={"main.py": b"print('x')"},
        requirements="requests==2.32.3\n",
    )

    assert _FakeSandboxSession.last_instance.kwargs["runtime_configs"] == {"network_mode": "bridge"}


def test_execute_network_mode_operator_override_wins(monkeypatch):
    _install_fake_sandbox(monkeypatch)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(executor, "_collect_generated_files", lambda *args, **kwargs: [])
    monkeypatch.setenv("LITELLM_SANDBOX_NETWORK_MODE", "bridge")

    executor.execute(
        code="print('hello')",
        skill_files={"main.py": b"print('x')"},
    )

    assert _FakeSandboxSession.last_instance.kwargs["runtime_configs"] == {"network_mode": "bridge"}


def test_execute_empty_requirements_txt_does_not_widen_network_mode(monkeypatch):
    """Regression: the requirements.txt FILENAME alone must not select bridge
    networking. An empty file means pip has nothing to install, so the
    session must stay at the default (none) — the execution container is the
    one running arbitrary skill code."""
    _install_fake_sandbox(monkeypatch)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(executor, "_collect_generated_files", lambda *args, **kwargs: [])
    monkeypatch.delenv("LITELLM_SANDBOX_NETWORK_MODE", raising=False)

    result = executor.execute(
        code="print('hello')",
        skill_files={
            "requirements.txt": b"",
            "main.py": b"print('x')",
        },
    )

    assert result["success"] is True
    assert _FakeSandboxSession.last_instance.kwargs["runtime_configs"] == {"network_mode": "none"}
    # Nothing to install: no pip invocation at all.
    assert not any("pip', 'install', '-r'" in c for c in _FakeSandboxSession.last_instance.run_calls)


def test_execute_explicit_none_not_widened_by_requirements_presence(monkeypatch):
    """Regression: an explicit LITELLM_SANDBOX_NETWORK_MODE=none must be
    honored even when requirements need installing (operator decision wins;
    the install may fail, but the mode is never silently widened)."""
    _install_fake_sandbox(monkeypatch)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(executor, "_collect_generated_files", lambda *args, **kwargs: [])
    monkeypatch.setenv("LITELLM_SANDBOX_NETWORK_MODE", "none")

    executor.execute(
        code="print('hello')",
        skill_files={"main.py": b"print('x')"},
        requirements="requests==2.32.3\n",
    )

    assert _FakeSandboxSession.last_instance.kwargs["runtime_configs"] == {"network_mode": "none"}


def test_execute_rejects_duplicate_normalized_requirements_paths(monkeypatch):
    """Regression: 'requirements.txt' and './requirements.txt' normalize to
    the same staged file, so a skill could ship an empty requirements.txt for
    validation and swap contents via the dotted variant. Duplicate normalized
    paths must be rejected outright."""
    _install_fake_sandbox(monkeypatch)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(executor, "_collect_generated_files", lambda *args, **kwargs: [])

    result = executor.execute(
        code="print('hello')",
        skill_files={
            "requirements.txt": b"",
            "./requirements.txt": b"--index-url https://evil.example.com/simple\n",
            "main.py": b"print('x')",
        },
    )

    assert result["success"] is False
    assert "Duplicate skill file path" in result["error"]


def test_execute_pip_installs_only_validated_bytes_with_path_variant_present(monkeypatch):
    """Regression: when both path variants could exist in a dict (e.g. after
    a caller normalizes keys differently), pip must read the canonical staged
    file whose bytes are exactly what host-side validation inspected."""
    _install_fake_sandbox(monkeypatch)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(executor, "_collect_generated_files", lambda *args, **kwargs: [])

    requirements = b"requests==2.32.3\n"
    result = executor.execute(
        code="print('hello')",
        skill_files={
            "requirements.txt": requirements,
            "main.py": b"print('x')",
        },
    )

    assert result["success"] is True
    created_session = _FakeSandboxSession.last_instance
    assert created_session.copied_contents["/sandbox/.litellm_requirements.txt"] == requirements
    install_calls = [c for c in created_session.run_calls if "pip', 'install', '-r'" in c]
    assert install_calls and "'.litellm_requirements.txt'" in install_calls[0]


def test_execute_generic_error_does_not_leak_host_paths(monkeypatch):
    class _ExplodingSandboxSession(_FakeSandboxSession):
        last_instance = None

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            type(self).last_instance = self

        def run(self, code):
            raise RuntimeError("/Users/someone/secret/host/path.py: boom")

    _install_fake_sandbox(monkeypatch, session_cls=_ExplodingSandboxSession)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(executor, "_collect_generated_files", lambda *args, **kwargs: [])

    result = executor.execute(
        code="print('hello')",
        skill_files={"main.py": b"print('x')"},
    )

    assert result["success"] is False
    assert result["error"] == "Sandbox execution failed"
    assert "/Users" not in result["error"]


def test_execute_drops_unsafe_skill_file_paths(monkeypatch):
    _install_fake_sandbox(monkeypatch)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(executor, "_collect_generated_files", lambda *args, **kwargs: [])

    result = executor.execute(
        code="print('hello')",
        skill_files={
            "main.py": b"print('x')",
            "pkg/helpers.py": b"h = 1",
            "../escape.py": b"bad",
            "pkg/../escape.py": b"bad",
            "/etc/passwd": b"bad",
            "C:\\\\windows\\\\evil.py": b"bad",
            "": b"bad",
        },
    )

    assert result["success"] is True
    copied_paths = {sandbox_path for _, sandbox_path in _FakeSandboxSession.last_instance.copy_to_runtime_calls}
    assert copied_paths == {"/sandbox/main.py", "/sandbox/pkg/helpers.py"}
    assert _FakeSandboxSession.last_instance.copied_contents["/sandbox/main.py"] == b"print('x')"
    assert _FakeSandboxSession.last_instance.copied_contents["/sandbox/pkg/helpers.py"] == b"h = 1"
