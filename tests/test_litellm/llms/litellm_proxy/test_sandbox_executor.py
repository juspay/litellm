import os
import sys
import types
from types import SimpleNamespace

import pytest

from litellm.llms.litellm_proxy.skills.sandbox_executor import SkillsSandboxExecutor


@pytest.fixture(autouse=True)
def _reset_fake_sessions():
    _FakeSandboxSession.instances = []
    _FakeSandboxSession.last_instance = None
    yield
    _FakeSandboxSession.instances = []
    _FakeSandboxSession.last_instance = None


class _FakeSandboxSession:
    last_instance = None
    instances = []
    install_exit_code = 0
    install_stdout = ""
    install_stderr = ""
    exec_exit_code = 0
    exec_stdout = "ok"
    exec_stderr = ""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.copy_to_runtime_calls = []
        self.copy_from_runtime_calls = []
        self.run_calls = []
        self.copied_contents = {}
        type(self).last_instance = self

    def __enter__(self):
        type(self).instances.append(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def copy_to_runtime(self, local_path, sandbox_path):
        self.copy_to_runtime_calls.append((local_path, sandbox_path))
        if os.path.isdir(local_path):
            self.copied_contents[sandbox_path] = b"<directory>"
            return
        with open(local_path, "rb") as f:
            self.copied_contents[sandbox_path] = f.read()

    def copy_from_runtime(self, sandbox_path, local_path):
        self.copy_from_runtime_calls.append((sandbox_path, local_path))
        if sandbox_path.endswith(".litellm_site"):
            # Mirror llm-sandbox: the archive rooted at .litellm_site
            # extracts to local_path/.litellm_site
            site_dir = os.path.join(local_path, ".litellm_site")
            os.makedirs(site_dir, exist_ok=True)
            with open(os.path.join(site_dir, "installed_marker.txt"), "w") as f:
                f.write("site")

    def run(self, code):
        self.run_calls.append(code)
        if "'pip'," in code:
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

    install_session, exec_session = _FakeSandboxSession.instances
    assert install_session.copied_contents["/sandbox/.litellm_requirements.txt"] == b"requests==2.32.3\nhttpx>=0.27\n"
    assert "'--target'," in install_session.run_calls[0]
    assert "os.chdir('/sandbox')" in exec_session.run_calls[0]


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

    install_session, exec_session = _FakeSandboxSession.instances
    assert install_session.copied_contents["/sandbox/.litellm_requirements.txt"] == b"requests==2.32.3\n"
    assert "'pip'," in install_session.run_calls[0]
    # Skill code runs in a session that never received the requirements file
    run_code_for_requirements = [c for c in exec_session.run_calls if "requirements" in c]
    assert all("'.litellm_requirements.txt'" in c for c in run_code_for_requirements)


def test_execute_returns_install_failure(monkeypatch):
    class _FailingSandboxSession(_FakeSandboxSession):
        install_exit_code = 1
        install_stdout = "pip output"
        install_stderr = "install failed"

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
    # Install session ran pip once; the execution session was never created
    assert len(_FailingSandboxSession.instances) == 1
    assert len(_FailingSandboxSession.instances[0].run_calls) == 1


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


def test_execute_install_session_networked_exec_session_isolated(monkeypatch):
    """Regression (whole-execution egress): even genuine requirements must
    never grant egress to the session running model-generated code. pip
    installs in a dedicated bridge session; the execution session is always
    network_mode=none unless the operator overrides it."""
    _install_fake_sandbox(monkeypatch)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(executor, "_collect_generated_files", lambda *args, **kwargs: [])
    monkeypatch.delenv("LITELLM_SANDBOX_NETWORK_MODE", raising=False)

    executor.execute(
        code="print('hello')",
        skill_files={"main.py": b"print('x')"},
        requirements="requests==2.32.3\n",
    )

    install_session, exec_session = _FakeSandboxSession.instances
    assert install_session.kwargs["runtime_configs"] == {"network_mode": "bridge"}
    assert exec_session.kwargs["runtime_configs"] == {"network_mode": "none"}


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
    assert not any("'pip'," in c for c in _FakeSandboxSession.last_instance.run_calls)


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

    install_session, exec_session = _FakeSandboxSession.instances
    assert install_session.kwargs["runtime_configs"] == {"network_mode": "none"}
    assert exec_session.kwargs["runtime_configs"] == {"network_mode": "none"}


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
    file re-serialized from the validated Requirement objects, never raw
    skill bytes."""
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
    install_session, _exec_session = _FakeSandboxSession.instances
    assert install_session.copied_contents["/sandbox/.litellm_requirements.txt"] == b"requests==2.32.3\n"
    install_calls = [c for c in install_session.run_calls if "'pip'," in c]
    assert install_calls and "'.litellm_requirements.txt'" in install_calls[0]


def test_execute_comment_only_requirements_do_not_widen_network_mode(monkeypatch):
    """Regression: the network decision must come from the parsed
    requirements, not raw text truthiness. A file of only comments and blank
    lines parses to zero requirements, so pip has nothing to install and no
    networked install session is created at all."""
    _install_fake_sandbox(monkeypatch)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(executor, "_collect_generated_files", lambda *args, **kwargs: [])
    monkeypatch.delenv("LITELLM_SANDBOX_NETWORK_MODE", raising=False)

    result = executor.execute(
        code="print('hello')",
        skill_files={
            "requirements.txt": b"# just a comment\n\n# another\n",
            "main.py": b"print('x')",
        },
    )

    assert result["success"] is True
    exec_session = _FakeSandboxSession.last_instance
    assert exec_session.kwargs["runtime_configs"] == {"network_mode": "none"}
    assert not any("'pip'," in c for c in exec_session.run_calls)
    # No install session: only the execution session was created
    assert len(_FakeSandboxSession.instances) == 1


def test_execute_stages_canonical_text_not_raw_skill_bytes(monkeypatch):
    """Regression: pip decodes the requirements file itself (PEP-263 coding
    declarations, BOM sniffing) and joins backslash continuations, so raw
    skill bytes can decode differently on pip's side than the host validator
    judged. pip must only ever read canonical text re-serialized from the
    validated Requirement objects."""
    _install_fake_sandbox(monkeypatch)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(executor, "_collect_generated_files", lambda *args, **kwargs: [])
    monkeypatch.delenv("LITELLM_SANDBOX_NETWORK_MODE", raising=False)

    result = executor.execute(
        code="print('hello')",
        skill_files={"main.py": b"print('x')"},
        requirements="requests==2.32.3 # pinned for prod\nhttpx >= 0.27\n\n",
    )

    assert result["success"] is True
    install_session, _exec_session = _FakeSandboxSession.instances
    assert install_session.copied_contents["/sandbox/.litellm_requirements.txt"] == b"requests==2.32.3\nhttpx>=0.27\n"


def test_execute_rejects_pep263_coding_declaration_bypass(monkeypatch):
    """Regression (validator/pip decoding divergence): with a latin-1 coding
    declaration, the UTF-8 bytes of U+00C5 decode on pip's side as C3 85,
    where 0x85 is NEL, a line separator. The host validator sees one benign
    line (everything after # is a comment); naive staging would hand pip a
    second line carrying pip options. pip must instead receive only the
    canonical re-serialization of the parsed requirement."""
    _install_fake_sandbox(monkeypatch)
    executor = SkillsSandboxExecutor()
    monkeypatch.setattr(executor, "_collect_generated_files", lambda *args, **kwargs: [])
    monkeypatch.delenv("LITELLM_SANDBOX_NETWORK_MODE", raising=False)

    result = executor.execute(
        code="print('hello')",
        skill_files={"main.py": b"print('x')"},
        requirements="requests #x" + chr(0xC5) + "-r /sandbox/evil.txt\n",
    )

    assert result["success"] is True
    install_session, _exec_session = _FakeSandboxSession.instances
    staged = install_session.copied_contents["/sandbox/.litellm_requirements.txt"]
    assert staged == b"requests\n"
    assert b"evil" not in staged
    assert b"-r" not in staged


def test_execute_generic_error_does_not_leak_host_paths(monkeypatch):
    class _ExplodingSandboxSession(_FakeSandboxSession):
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
