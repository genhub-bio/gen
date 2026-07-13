"""Install Gen's bundled agent skills into coding-agent skills directories."""

import pathlib
import shutil

DEFAULT_SKILL_NAME = "gen-genetic-engineering-python"


def _auto_targets() -> list[tuple[pathlib.Path, pathlib.Path]]:
    # Each entry is (marker, skills_dir): `marker` existing signals the tool
    # is set up on this machine; `skills_dir` is where that tool (and any
    # compatible tools) looks for personal/global skills.
    #   - ~/.claude/skills — Claude Code; also read by OpenCode.
    #   - ~/.agents/skills — Codex CLI; also read by OpenCode.
    home = pathlib.Path.home()
    return [
        (home / ".claude", home / ".claude" / "skills"),
        (home / ".agents", home / ".agents" / "skills"),
    ]


def _bundled_skills_dir() -> pathlib.Path:
    # Ships as package data under gen-python/python/gen/_skills/, so this
    # resolves correctly whether gen was installed from a wheel or sdist,
    # or is being run from an editable/source checkout.
    return pathlib.Path(__file__).resolve().parent / "_skills"


def _install_one(
    source: pathlib.Path, link_path: pathlib.Path, force: bool
) -> pathlib.Path:
    if link_path.is_symlink() or link_path.exists():
        if not force:
            raise FileExistsError(
                f"{link_path} already exists. Pass force=True to replace it."
            )
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        else:
            shutil.rmtree(link_path)

    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(source, target_is_directory=True)
    return link_path


def install_skill(
    name: str = DEFAULT_SKILL_NAME,
    target: str | pathlib.Path | None = None,
    force: bool = False,
) -> list[pathlib.Path]:
    """Symlink one of Gen's bundled skill folders into coding-agent skills directories.

    Parameters
    name : str
        Name of the skill bundled with this package, and the symlink's
        filename when `target` is omitted.
    target : str or pathlib.Path, optional
        Exact path for the symlink. When given, installs only there. When
        omitted, auto-detects which coding agents are set up on this machine
        (by checking for `~/.claude` and `~/.agents`) and installs into every
        one found: `~/.claude/skills/<name>` (read by Claude Code and
        OpenCode) and/or `~/.agents/skills/<name>` (read by Codex CLI and
        OpenCode). Raises if neither is found.
    force : bool
        Replace an existing file, directory, or symlink at each target.

    Returns
    -------
    list[pathlib.Path]
        The path(s) of the created symlink(s).
    """
    source = _bundled_skills_dir() / name
    if not source.is_dir():
        raise FileNotFoundError(
            f"No skill named {name!r} bundled with this package (looked in {source})."
        )

    if target is not None:
        return [_install_one(source, pathlib.Path(target), force)]

    installed = []
    for marker, skills_dir in _auto_targets():
        if marker.is_dir():
            installed.append(_install_one(source, skills_dir / name, force))

    if not installed:
        raise RuntimeError(
            "No supported agent skills directory found (looked for ~/.claude and "
            "~/.agents). Pass target= to install to an explicit location."
        )
    return installed
