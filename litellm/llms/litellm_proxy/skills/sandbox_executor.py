"""
Sandbox Executor for LiteLLM Skills

Executes skill code in a sandboxed environment using llm-sandbox.
Supports Docker, Podman, and Kubernetes backends.
"""

import base64
import os
import re
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional

from packaging.requirements import InvalidRequirement, Requirement

from litellm._logging import verbose_logger


def is_safe_skill_file_path(path: str) -> bool:
    if "\\" in path or "\x00" in path:
        return False
    p = PurePosixPath(path)
    if not p.parts or p.is_absolute() or ".." in p.parts:
        return False
    return not re.match(r"^[A-Za-z]:", p.parts[0])


def _validated_skill_file_paths(skill_files: dict[str, bytes]) -> dict[str, bytes]:
    safe_files = {path: content for path, content in skill_files.items() if is_safe_skill_file_path(path)}
    normalized_paths: set[str] = set()
    for path in safe_files:
        normalized = str(PurePosixPath(path))
        if normalized in normalized_paths:
            raise ValueError(f"Duplicate skill file path after normalization: {normalized}")
        normalized_paths.add(normalized)
    return safe_files


def _parse_requirements(requirements: str) -> tuple[Requirement, ...]:
    """Parse a requirements file into PEP 508 Requirements, rejecting
    everything besides named packages from the trusted index.

    pip options (-e, --index-url, --extra-index-url, --find-links, --hash,
    per-requirement options after the package name, section headers like
    [options]), VCS URLs, and local paths all fail to parse or carry ``url``,
    so a model-generated requirements file can only pull named packages from
    the configured index.

    Raises ValueError naming the offending line so the caller can surface a
    generic message instead of executing the install.
    """
    parsed: list[Requirement] = []
    for line in requirements.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        try:
            req = Requirement(stripped)
        except InvalidRequirement:
            raise ValueError(f"Disallowed requirements line: {stripped.split()[0]}")
        if req.url is not None:
            raise ValueError(f"Disallowed requirements line: {req.name}")
        parsed.append(req)
    return tuple(parsed)


def _canonical_requirements_text(parsed: tuple[Requirement, ...]) -> str:
    """Serialize parsed Requirements back to canonical form for pip.

    pip decodes the requirements file itself (PEP-263 coding declarations,
    BOM sniffing) and joins backslash continuations, so it can interpret raw
    skill-controlled bytes differently than the host-side validator judged
    the decoded string. Serializing the already-validated Requirement objects
    removes that divergence: pip reads only text generated from validated
    parse output.
    """
    return "".join(f"{req}\n" for req in parsed)


def _validate_requirements(requirements: str) -> None:
    _parse_requirements(requirements)


class SkillsSandboxExecutor:
    """
    Executes skill code in llm-sandbox Docker container.

    Responsibilities:
    - Create sandbox session with skill files
    - Install requirements
    - Execute model-generated code
    - Collect generated files (GIFs, images, etc.)
    """

    def __init__(
        self,
        timeout: int = 60,
        backend: str = "docker",
        image: Optional[str] = None,
    ):
        """
        Initialize the sandbox executor.

        Args:
            timeout: Maximum execution time in seconds
            backend: Sandbox backend ("docker", "podman", "kubernetes")
            image: Custom Docker image (default: uses llm-sandbox default)
        """
        self.timeout = timeout
        self.backend = backend
        self.image = image
        self._session = None

    def execute(
        self,
        code: str,
        skill_files: Dict[str, bytes],
        requirements: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute code with skill files in sandbox.

        Args:
            code: Python code to execute
            skill_files: Dict mapping file paths to binary content
            requirements: Optional requirements.txt content

        Returns:
            {
                "success": bool,
                "output": str,
                "error": str (if failed),
                "files": [{"name": str, "content_base64": str, "mime_type": str}]
            }
        """
        try:
            from llm_sandbox import SandboxSession
        except ImportError:
            verbose_logger.error("SkillsSandboxExecutor: llm-sandbox not installed. Install `llm-sandbox`.")
            return {
                "success": False,
                "output": "",
                "error": "llm-sandbox not installed. Install `llm-sandbox`.",
                "files": [],
            }

        # Validate requirements on the host before any sandbox resource is
        # created: a rejected install must not spin up a container at all,
        # and the caller gets the specific disallowed token instead of the
        # generic failure (the message contains only the offending line's
        # first token, never host paths). pip never sees the raw skill bytes:
        # the validated Requirement objects are re-serialized to canonical
        # text and staged as .litellm_requirements.txt below, so pip's own
        # decoding (PEP-263 coding declarations, BOM sniffing) and line
        # joining cannot reinterpret what the validator judged.
        _raw_requirements = requirements or (
            skill_files.get("requirements.txt", b"").decode("utf-8", errors="replace")
            if "requirements.txt" in skill_files
            else ""
        )
        try:
            _parsed_requirements = _parse_requirements(_raw_requirements) if _raw_requirements else ()
        except ValueError as _req_err:
            verbose_logger.error(f"SkillsSandboxExecutor: {_req_err}")
            return {
                "success": False,
                "output": "",
                "error": str(_req_err),
                "files": [],
            }
        _canonical_requirements = _canonical_requirements_text(_parsed_requirements)

        try:
            skill_files = _validated_skill_file_paths(skill_files)
        except ValueError as _dup_err:
            verbose_logger.error(f"SkillsSandboxExecutor: {_dup_err}")
            return {
                "success": False,
                "output": "",
                "error": str(_dup_err),
                "files": [],
            }

        try:
            # Create sandbox session
            base_session_kwargs: Dict[str, Any] = {
                "lang": "python",
                "verbose": False,
            }

            if self.image:
                base_session_kwargs["image"] = self.image

            # Network isolation: model-generated code runs with no outbound
            # network. Requirements installation is the only step that
            # legitimately needs the network (to reach the package index),
            # so it runs in a separate, short-lived install session whose
            # only inputs are the canonical requirements text and whose only
            # action is pip. The installed packages are copied out and the
            # install session is destroyed before any skill code runs, so
            # even a genuine dependency never grants egress to the session
            # executing model-generated code. An explicit
            # LITELLM_SANDBOX_NETWORK_MODE is authoritative for both sessions
            # and is never silently widened.
            _operator_network_mode = os.getenv("LITELLM_SANDBOX_NETWORK_MODE")
            if _operator_network_mode is not None:
                _exec_network_mode = _operator_network_mode
            else:
                _exec_network_mode = "none"

            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                # 1. Install requirements in a dedicated networked session
                _installed_requirements_dir: Optional[str] = None
                if _parsed_requirements:
                    _install_network_mode = _operator_network_mode if _operator_network_mode is not None else "bridge"
                    install_session_kwargs = {
                        **base_session_kwargs,
                        "runtime_configs": {"network_mode": _install_network_mode},
                    }
                    with SandboxSession(**install_session_kwargs) as install_session:
                        # The canonical text re-serialized from the
                        # validated Requirement objects is the only input
                        # pip ever reads; raw skill bytes are never handed to
                        # pip, so pip cannot reinterpret them via its own
                        # decoding or line joining.
                        with tempfile.NamedTemporaryFile(
                            mode="w",
                            encoding="utf-8",
                            delete=False,
                        ) as f:
                            f.write(_canonical_requirements)
                            local_requirements_path = f.name
                        install_session.copy_to_runtime(
                            local_requirements_path,
                            "/sandbox/.litellm_requirements.txt",
                        )
                        os.unlink(local_requirements_path)
                        pip_code = """
import subprocess
import sys
subprocess.run(
    [
        sys.executable,
        '-m',
        'pip',
        'install',
        '--target',
        '/sandbox/.litellm_site',
        '-r',
        '.litellm_requirements.txt',
    ],
    check=True,
    cwd='/sandbox',
)
"""
                        install_result = install_session.run(pip_code)
                        if install_result.exit_code != 0:
                            verbose_logger.debug("SkillsSandboxExecutor: Requirements installation failed")
                            return {
                                "success": False,
                                "output": install_result.stdout or "",
                                "error": install_result.stderr or "",
                                "files": [],
                            }
                        verbose_logger.debug("SkillsSandboxExecutor: Installed requirements")
                        install_session.copy_from_runtime(
                            "/sandbox/.litellm_site",
                            os.path.join(tmpdir, "installed"),
                        )
                        _installed_requirements_dir = os.path.join(tmpdir, "installed", ".litellm_site")

                # 2. Execute the code in an isolated session
                exec_session_kwargs = {
                    **base_session_kwargs,
                    "runtime_configs": {"network_mode": _exec_network_mode},
                }
                with SandboxSession(**exec_session_kwargs) as session:
                    if _installed_requirements_dir is not None:
                        session.copy_to_runtime(
                            _installed_requirements_dir,
                            "/sandbox/.litellm_site",
                        )

                    tmpdir_abs = os.path.abspath(tmpdir)
                    for path, content in skill_files.items():
                        # Create the file in temp directory
                        local_path = os.path.abspath(os.path.join(tmpdir, "skill", path))
                        if not local_path.startswith(tmpdir_abs + os.sep):
                            verbose_logger.warning(f"SkillsSandboxExecutor: Skipping file with invalid path: {path}")
                            continue
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        with open(local_path, "wb") as f:
                            f.write(content)

                        # Copy to sandbox
                        sandbox_path = f"/sandbox/{path}"
                        session.copy_to_runtime(local_path, sandbox_path)

                    verbose_logger.debug(f"SkillsSandboxExecutor: Copied {len(skill_files)} files to sandbox")

                    # Wrap code to run from /sandbox directory
                    wrapped_code = f"""
import os
os.chdir('/sandbox')
import sys
sys.path.insert(0, '/sandbox')
sys.path.insert(0, '/sandbox/.litellm_site')

{code}
"""
                    result = session.run(wrapped_code)

                success = result.exit_code == 0
                output = result.stdout or ""
                error = result.stderr or ""

                if success:
                    verbose_logger.debug("SkillsSandboxExecutor: Code execution succeeded")
                else:
                    verbose_logger.debug(
                        f"SkillsSandboxExecutor: Code execution failed with exit code {result.exit_code}"
                    )
                    verbose_logger.debug(f"SkillsSandboxExecutor: stderr: {error[:500] if error else 'No stderr'}")
                    verbose_logger.debug(f"SkillsSandboxExecutor: stdout: {output[:500] if output else 'No stdout'}")

                # 4. Collect generated files
                generated_files = self._collect_generated_files(session, skill_files)

                return {
                    "success": success,
                    "output": output,
                    "error": error,
                    "files": generated_files,
                }

        except Exception as e:
            # Log the full error server-side, but return a generic message:
            # sandbox library exceptions embed host paths (staging dirs,
            # docker socket locations) that must not reach the caller.
            verbose_logger.error(f"SkillsSandboxExecutor: Execution failed: {e}")
            return {
                "success": False,
                "output": "",
                "error": "Sandbox execution failed",
                "files": [],
            }

    def _collect_generated_files(
        self,
        session: Any,
        original_files: Dict[str, bytes],
    ) -> List[Dict[str, Any]]:
        """
        Collect files generated during execution.

        Looks for new files in /sandbox that weren't in the original skill files.
        Focuses on common output types: GIF, PNG, JPG, PDF, CSV, etc.

        Args:
            session: The sandbox session
            original_files: Original skill files (to exclude)

        Returns:
            List of generated files with base64 content
        """
        generated_files: List[Dict[str, Any]] = []

        try:
            import tempfile

            # List files in /sandbox using Python code
            list_code = """
import os
import json
files = []
for root, dirs, filenames in os.walk('/sandbox'):
    for f in filenames:
        if f.endswith(('.gif', '.png', '.jpg', '.jpeg', '.pdf', '.csv', '.json')):
            files.append(os.path.join(root, f))
print(json.dumps(files))
"""
            result = session.run(list_code)

            if result.exit_code == 0 and result.stdout:
                import json

                try:
                    filepaths = json.loads(result.stdout.strip())
                except json.JSONDecodeError:
                    filepaths = []

                for filepath in filepaths:
                    if not filepath:
                        continue

                    # Get relative path
                    rel_path = filepath.replace("/sandbox/", "")

                    # Skip if it was an original file or staged install
                    # artifacts (site-packages, requirement metadata)
                    if rel_path in original_files or rel_path.startswith(".litellm_site"):
                        continue

                    # Copy file from sandbox using copy_from_runtime
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp_path = tmp.name

                    try:
                        session.copy_from_runtime(filepath, tmp_path)

                        with open(tmp_path, "rb") as f:
                            content = f.read()

                        content_b64 = base64.b64encode(content).decode("utf-8")
                        generated_files.append(
                            {
                                "name": os.path.basename(filepath),
                                "path": rel_path,
                                "content_base64": content_b64,
                                "mime_type": self._get_mime_type(filepath),
                            }
                        )

                        verbose_logger.debug(f"SkillsSandboxExecutor: Collected generated file: {rel_path}")
                    except Exception as e:
                        verbose_logger.warning(f"SkillsSandboxExecutor: Error copying file {filepath}: {e}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)

        except Exception as e:
            verbose_logger.warning(f"SkillsSandboxExecutor: Error collecting generated files: {e}")

        return generated_files

    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type for a file based on extension."""
        ext = filename.lower().split(".")[-1]
        return {
            "gif": "image/gif",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "pdf": "application/pdf",
            "csv": "text/csv",
            "json": "application/json",
            "txt": "text/plain",
        }.get(ext, "application/octet-stream")
