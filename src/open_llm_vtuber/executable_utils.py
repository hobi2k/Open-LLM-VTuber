import os
import shutil
from pathlib import Path


def resolve_executable(configured: str, command: str) -> str | None:
    value = str(configured or "auto").strip()
    if value not in {"", "auto"}:
        expanded = str(Path(value).expanduser())
        resolved = shutil.which(expanded)
        if resolved:
            return resolved
        path = Path(expanded)
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None

    resolved = shutil.which(command)
    if resolved:
        return resolved

    candidates = [path / command for path in executable_search_paths()]
    return next(
        (
            str(path)
            for path in candidates
            if path.is_file() and os.access(path, os.X_OK)
        ),
        None,
    )


def executable_environment() -> dict[str, str]:
    environment = os.environ.copy()
    paths = [str(path) for path in executable_search_paths()]
    if environment.get("PATH"):
        paths.append(environment["PATH"])
    environment["PATH"] = os.pathsep.join(dict.fromkeys(paths))
    return environment


def executable_search_paths() -> list[Path]:
    home = Path.home()
    return [
        Path("/opt/homebrew/bin"),
        Path("/usr/local/bin"),
        Path("/opt/local/bin"),
        home / ".local/bin",
        home / ".bun/bin",
        home / "Library/pnpm/bin",
        home / ".cargo/bin",
        *sorted((home / ".nvm/versions/node").glob("*/bin"), reverse=True),
    ]


def executable_version(output: str) -> str | None:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return next(
        (line for line in reversed(lines) if not line.lower().startswith("warning:")),
        lines[0] if lines else None,
    )
