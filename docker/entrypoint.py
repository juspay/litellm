import os
import subprocess
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
venv_python = os.path.join(repo_root, ".venv", "bin", "python")
migration_script = os.path.join(repo_root, "litellm", "proxy", "prisma_migration.py")


def run_migration() -> None:
    if os.access(venv_python, os.X_OK):
        subprocess.run([venv_python, migration_script], check=True)
        return
    if os.path.exists(os.path.join(repo_root, ".venv")) or shutil_which("uv"):
        subprocess.run(
            ["uv", "run", "--no-sync", "python", migration_script],
            cwd=repo_root,
            check=True,
        )
        return
    subprocess.run([sys.executable, migration_script], check=True)


def shutil_which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


if __name__ == "__main__":
    run_migration()
    print("Migration script ran successfully!")
