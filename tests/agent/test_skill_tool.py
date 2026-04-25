"""Tests for nanobot.agent.tools.skill_tool (SkillManageTool)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from nanobot.agent.skills import SkillsLoader
from nanobot.agent.tools.skill_tool import SkillManageTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loader(tmp_path: Path) -> tuple[Path, SkillsLoader, SkillManageTool]:
    """Create a workspace, SkillsLoader, and SkillManageTool wired together."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    tool = SkillManageTool(skills_loader=loader)
    return workspace, loader, tool


def _skill_md(name: str, description: str, body: str = "# Skill body") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_skill_success(tmp_path: Path) -> None:
    ws, loader, tool = _make_loader(tmp_path)
    result = json.loads(_run(tool.execute(
        action="create", name="my-skill", content=_skill_md("my-skill", "A test"),
    )))
    assert result["success"] is True
    assert "my-skill" in result["message"]
    assert (ws / "skills" / "my-skill" / "SKILL.md").exists()


def test_create_skill_with_category(tmp_path: Path) -> None:
    ws, loader, tool = _make_loader(tmp_path)
    result = json.loads(_run(tool.execute(
        action="create", name="ml-skill", content=_skill_md("ml-skill", "ML"),
        category="data-science",
    )))
    assert result["success"] is True
    assert (ws / "skills" / "data-science" / "ml-skill" / "SKILL.md").exists()


def test_create_skill_missing_content(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    result = json.loads(_run(tool.execute(action="create", name="x", content=None)))
    assert result["success"] is False
    assert "content" in result["error"]


def test_create_skill_invalid_name(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    result = json.loads(_run(tool.execute(
        action="create", name="INVALID", content=_skill_md("x", "d"),
    )))
    assert result["success"] is False
    assert "Invalid" in result["error"]


def test_create_skill_duplicate(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    _run(tool.execute(action="create", name="dup", content=_skill_md("dup", "First")))
    result = json.loads(_run(tool.execute(
        action="create", name="dup", content=_skill_md("dup", "Second"),
    )))
    assert result["success"] is False
    assert "already exists" in result["error"]


def test_create_skill_includes_scan_verdict(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    result = json.loads(_run(tool.execute(
        action="create", name="clean", content=_skill_md("clean", "Clean skill"),
    )))
    assert result["success"] is True
    assert result["scan_verdict"] == "safe"


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


def test_edit_skill_success(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    _run(tool.execute(action="create", name="editable", content=_skill_md("editable", "Original")))
    result = json.loads(_run(tool.execute(
        action="edit", name="editable", content=_skill_md("editable", "Updated", "# New body"),
    )))
    assert result["success"] is True
    assert "updated" in result["message"].lower()


def test_edit_skill_missing_content(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    result = json.loads(_run(tool.execute(action="edit", name="x", content=None)))
    assert result["success"] is False
    assert "content" in result["error"]


def test_edit_skill_not_found(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    result = json.loads(_run(tool.execute(
        action="edit", name="ghost", content=_skill_md("ghost", "desc"),
    )))
    assert result["success"] is False
    assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# patch
# ---------------------------------------------------------------------------


def test_patch_skill_success(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    _run(tool.execute(action="create", name="patchable", content=_skill_md("patchable", "Desc", "# Step\n\nDo the thing.")))
    result = json.loads(_run(tool.execute(
        action="patch", name="patchable", old_string="Do the thing.", new_string="Do the better thing.",
    )))
    assert result["success"] is True
    assert result["match_count"] == 1


def test_patch_skill_missing_old_string(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    result = json.loads(_run(tool.execute(action="patch", name="x", old_string="", new_string="y")))
    assert result["success"] is False
    assert "old_string" in result["error"]


def test_patch_skill_missing_new_string(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    result = json.loads(_run(tool.execute(action="patch", name="x", old_string="a", new_string=None)))
    assert result["success"] is False
    assert "new_string" in result["error"]


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_skill_success(tmp_path: Path) -> None:
    ws, _, tool = _make_loader(tmp_path)
    _run(tool.execute(action="create", name="to-delete", content=_skill_md("to-delete", "Desc")))
    result = json.loads(_run(tool.execute(action="delete", name="to-delete")))
    assert result["success"] is True
    assert not (ws / "skills" / "to-delete").exists()


def test_delete_skill_not_found(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    result = json.loads(_run(tool.execute(action="delete", name="ghost")))
    assert result["success"] is False
    assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


def test_write_file_success(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    _run(tool.execute(action="create", name="with-files", content=_skill_md("with-files", "Desc")))
    result = json.loads(_run(tool.execute(
        action="write_file", name="with-files",
        file_path="references/api-guide.md", file_content="# API Guide\n\nUse the API wisely.",
    )))
    assert result["success"] is True


def test_write_file_missing_path(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    result = json.loads(_run(tool.execute(action="write_file", name="x", file_path=None, file_content="c")))
    assert result["success"] is False
    assert "file_path" in result["error"]


def test_write_file_missing_content(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    result = json.loads(_run(tool.execute(action="write_file", name="x", file_path="references/a.md", file_content=None)))
    assert result["success"] is False
    assert "file_content" in result["error"]


def test_write_file_invalid_path(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    result = json.loads(_run(tool.execute(
        action="write_file", name="x", file_path="../etc/passwd", file_content="hax",
    )))
    assert result["success"] is False


def test_write_file_skill_not_found(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    result = json.loads(_run(tool.execute(
        action="write_file", name="ghost", file_path="references/a.md", file_content="c",
    )))
    assert result["success"] is False
    assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# remove_file
# ---------------------------------------------------------------------------


def test_remove_file_success(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    _run(tool.execute(action="create", name="rm-file", content=_skill_md("rm-file", "Desc")))
    _run(tool.execute(
        action="write_file", name="rm-file",
        file_path="references/to-rm.md", file_content="Remove me.",
    ))
    result = json.loads(_run(tool.execute(
        action="remove_file", name="rm-file", file_path="references/to-rm.md",
    )))
    assert result["success"] is True


def test_remove_file_missing_path(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    result = json.loads(_run(tool.execute(action="remove_file", name="x", file_path=None)))
    assert result["success"] is False
    assert "file_path" in result["error"]


def test_remove_file_not_found(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    _run(tool.execute(action="create", name="no-file", content=_skill_md("no-file", "Desc")))
    result = json.loads(_run(tool.execute(
        action="remove_file", name="no-file", file_path="references/nonexistent.md",
    )))
    assert result["success"] is False


# ---------------------------------------------------------------------------
# unknown action
# ---------------------------------------------------------------------------


def test_unknown_action(tmp_path: Path) -> None:
    _, _, tool = _make_loader(tmp_path)
    result = json.loads(_run(tool.execute(action="bogus", name="x")))
    assert result["success"] is False
    assert "Unknown action" in result["error"]


# ---------------------------------------------------------------------------
# tool name and schema
# ---------------------------------------------------------------------------


def test_tool_name_and_schema() -> None:
    from nanobot.agent.skills import SkillsLoader
    loader = SkillsLoader(Path("/tmp/fake-ws"))
    tool = SkillManageTool(skills_loader=loader)
    assert tool.name == "skill_manage"
    schema = tool.parameters
    assert schema["type"] == "object"
    assert "action" in schema["properties"]
    assert "name" in schema["properties"]
    assert schema["required"] == ["action", "name"]
