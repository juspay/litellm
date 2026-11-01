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
    monkeypatch.setattr(
        executor, "_collect_generated_files", lambda *args, **kwargs: []
    )

    requirements = "requests==2.32.3\nhttpx>=0.27\n# a comment line\n\n"
    result = executor.execute(
        code="print('hello')",
        skill_files={"pkg/__init__.py": b""},
        requirements=requirements,
    )

    assert result["success"] is True

    created_session = _FakeSandboxSession.last_instance
    assert created_session.copied_contents[
        "/sandbox/.litellm_requirements.txt"
    ] == requirements.encode("utf-8")
    assert (
        "pip', 'install', '-r', '.litellm_requirements.txt'"
        in created_session.run_calls[0]
    )
    assert "os.chdir('/sandbox')" in created_session.run_calls[1]


def test_execute_uses_skill_requirements_txt(monkeypatch):
    _install_fake_sandbox(monkeypatch)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(
        executor, "_collect_generated_files", lambda *args, **kwargs: []
    )

    result = executor.execute(
        code="print('hello')",
        skill_files={
            "requirements.txt": b"requests==2.32.3\n",
            "main.py": b"print('x')",
        },
    )

    assert result["success"] is True

    created_session = _FakeSandboxSession.last_instance
    copied_paths = {
        sandbox_path for _, sandbox_path in created_session.copy_to_runtime_calls
    }
    assert "/sandbox/requirements.txt" in copied_paths
    assert "/sandbox/.litellm_requirements.txt" not in copied_paths
    assert "pip', 'install', '-r', 'requirements.txt'" in created_session.run_calls[0]


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
    monkeypatch.setattr(
        executor, "_collect_generated_files", lambda *args, **kwargs: []
    )

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
    monkeypatch.setattr(
        executor, "_collect_generated_files", lambda *args, **kwargs: []
    )

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
    monkeypatch.setattr(
        executor, "_collect_generated_files", lambda *args, **kwargs: []
    )

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
    monkeypatch.setattr(
        executor, "_collect_generated_files", lambda *args, **kwargs: []
    )
    monkeypatch.delenv("LITELLM_SANDBOX_NETWORK_MODE", raising=False)

    executor.execute(
        code="print('hello')",
        skill_files={"main.py": b"print('x')"},
    )

    assert _FakeSandboxSession.last_instance.kwargs["runtime_configs"] == {"network_mode": "none"}


def test_execute_network_mode_bridge_when_requirements_need_install(monkeypatch):
    _install_fake_sandbox(monkeypatch)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(
        executor, "_collect_generated_files", lambda *args, **kwargs: []
    )
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
    monkeypatch.setattr(
        executor, "_collect_generated_files", lambda *args, **kwargs: []
    )
    monkeypatch.setenv("LITELLM_SANDBOX_NETWORK_MODE", "bridge")

    executor.execute(
        code="print('hello')",
        skill_files={"main.py": b"print('x')"},
    )

    assert _FakeSandboxSession.last_instance.kwargs["runtime_configs"] == {"network_mode": "bridge"}


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
    monkeypatch.setattr(
        executor, "_collect_generated_files", lambda *args, **kwargs: []
    )

    result = executor.execute(
        code="print('hello')",
        skill_files={"main.py": b"print('x')"},
    )

    assert result["success"] is False
    assert result["error"] == "Sandbox execution failed"
    assert "/Users" not in result["error"]
