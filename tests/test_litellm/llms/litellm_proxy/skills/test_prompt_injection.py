"""Regression tests for SkillPromptInjectionHandler.extract_all_files path safety.

extract_all_files unzips a skill archive into the sandbox workspace; every
entry must pass the same predicate the sandbox executor applies
(is_safe_skill_file_path) before it is written, and normalized-equality
must hold so "../" traversal cannot smuggle files outside the workspace.
"""

import io
import zipfile

from litellm.llms.litellm_proxy.skills.prompt_injection import SkillPromptInjectionHandler
from litellm.models.skills import LiteLLM_SkillsTable


def _zip_skill(entries: dict) -> LiteLLM_SkillsTable:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return LiteLLM_SkillsTable(skill_id="sk-1", file_content=buf.getvalue())


def test_extract_all_files_strips_skill_folder_prefix():
    skill = _zip_skill({"slack-gif-creator/core/main.py": b"print('hi')"})
    files = SkillPromptInjectionHandler().extract_all_files(skill)
    assert files == {"core/main.py": b"print('hi')"}


def test_extract_all_files_drops_traversal_entries():
    """Zip entries carrying ../ segments must never reach the workspace."""
    skill = _zip_skill(
        {
            "skill/../../etc/passwd": b"root",
            "skill/ok.txt": b"fine",
            "skill/a/../../escape.txt": b"out",
        }
    )
    files = SkillPromptInjectionHandler().extract_all_files(skill)
    assert "ok.txt" in files
    assert all(not k.startswith("..") for k in files)
    assert "escape.txt" not in files
    assert all(".." not in k for k in files)


def test_extract_all_files_drops_windows_style_paths():
    """Backslash and drive-letter entries must be refused (this predicate is
    shared with the sandbox executor)."""
    skill = _zip_skill(
        {
            "skill/core\\evil.py": b"x",
            "skill/C:/windows/temp/evil.bat": b"x",
            "skill/C:\\abs": b"x",
            "skill/ok.txt": b"fine",
        }
    )
    files = SkillPromptInjectionHandler().extract_all_files(skill)
    assert list(files) == ["ok.txt"]


def test_extract_all_files_drops_absolute_paths():
    skill = _zip_skill(
        {
            "skill//abs.txt": b"x",
            "skill/ok.txt": b"fine",
        }
    )
    files = SkillPromptInjectionHandler().extract_all_files(skill)
    assert list(files) == ["ok.txt"]


# --------------------------- shared path predicate ---------------------------


def testis_safe_skill_file_path_predicate():
    """The shared predicate both validation layers rely on: rejects
    backslashes, NUL bytes, absolute paths, '..' segments, and Windows
    drive-letter prefixes; accepts normal relative paths."""
    from litellm.llms.litellm_proxy.skills.sandbox_executor import is_safe_skill_file_path

    assert is_safe_skill_file_path("core/main.py")
    assert is_safe_skill_file_path("README.md")
    assert not is_safe_skill_file_path("core\\evil.py")
    assert not is_safe_skill_file_path("core\x00evil.py")
    assert not is_safe_skill_file_path("../escape.txt")
    assert not is_safe_skill_file_path("a/../../escape.txt")
    assert not is_safe_skill_file_path("/abs/path.txt")
    assert not is_safe_skill_file_path("C:/windows/temp/x.bat")
    assert not is_safe_skill_file_path("C:\\abs")
