"""Tests for nanobot.agent.skills.SkillsLoader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobot.agent.skills import SkillsLoader


def _write_skill(
    base: Path,
    name: str,
    *,
    metadata_json: dict | None = None,
    body: str = "# Skill\n",
) -> Path:
    """Create ``base / name / SKILL.md`` with optional nanobot metadata JSON."""
    skill_dir = base / name
    skill_dir.mkdir(parents=True)
    lines = ["---"]
    if metadata_json is not None:
        payload = json.dumps({"nanobot": metadata_json}, separators=(",", ":"))
        lines.append(f'metadata: {payload}')
    lines.extend(["---", "", body])
    path = skill_dir / "SKILL.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_list_skills_empty_when_skills_dir_missing(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    assert loader.list_skills(filter_unavailable=False) == []


def test_list_skills_empty_when_skills_dir_exists_but_empty(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    (workspace / "skills").mkdir(parents=True)
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    assert loader.list_skills(filter_unavailable=False) == []


def test_list_skills_workspace_entry_shape_and_source(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    skills_root = workspace / "skills"
    skills_root.mkdir(parents=True)
    skill_path = _write_skill(skills_root, "alpha", body="# Alpha")
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    entries = loader.list_skills(filter_unavailable=False)
    assert entries == [
        {"name": "alpha", "path": str(skill_path), "source": "workspace"},
    ]


def test_list_skills_skips_non_directories_and_missing_skill_md(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    skills_root = workspace / "skills"
    skills_root.mkdir(parents=True)
    (skills_root / "not_a_dir.txt").write_text("x", encoding="utf-8")
    (skills_root / "no_skill_md").mkdir()
    ok_path = _write_skill(skills_root, "ok", body="# Ok")
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    entries = loader.list_skills(filter_unavailable=False)
    names = {entry["name"] for entry in entries}
    assert names == {"ok"}
    assert entries[0]["path"] == str(ok_path)


def test_list_skills_workspace_shadows_builtin_same_name(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    ws_skills = workspace / "skills"
    ws_skills.mkdir(parents=True)
    ws_path = _write_skill(ws_skills, "dup", body="# Workspace wins")

    builtin = tmp_path / "builtin"
    _write_skill(builtin, "dup", body="# Builtin")

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    entries = loader.list_skills(filter_unavailable=False)
    assert len(entries) == 1
    assert entries[0]["source"] == "workspace"
    assert entries[0]["path"] == str(ws_path)


def test_list_skills_merges_workspace_and_builtin(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    ws_skills = workspace / "skills"
    ws_skills.mkdir(parents=True)
    ws_path = _write_skill(ws_skills, "ws_only", body="# W")
    builtin = tmp_path / "builtin"
    bi_path = _write_skill(builtin, "bi_only", body="# B")

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    entries = sorted(loader.list_skills(filter_unavailable=False), key=lambda item: item["name"])
    assert entries == [
        {"name": "bi_only", "path": str(bi_path), "source": "builtin"},
        {"name": "ws_only", "path": str(ws_path), "source": "workspace"},
    ]


def test_list_skills_builtin_omitted_when_dir_missing(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    ws_skills = workspace / "skills"
    ws_skills.mkdir(parents=True)
    ws_path = _write_skill(ws_skills, "solo", body="# S")
    missing_builtin = tmp_path / "no_such_builtin"

    loader = SkillsLoader(workspace, builtin_skills_dir=missing_builtin)
    entries = loader.list_skills(filter_unavailable=False)
    assert entries == [{"name": "solo", "path": str(ws_path), "source": "workspace"}]


def test_list_skills_filter_unavailable_excludes_unmet_bin_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    skills_root = workspace / "skills"
    skills_root.mkdir(parents=True)
    _write_skill(
        skills_root,
        "needs_bin",
        metadata_json={"requires": {"bins": ["nanobot_test_fake_binary"]}},
    )
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    def fake_which(cmd: str) -> str | None:
        if cmd == "nanobot_test_fake_binary":
            return None
        return "/usr/bin/true"

    monkeypatch.setattr("nanobot.agent.skills.shutil.which", fake_which)

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    assert loader.list_skills(filter_unavailable=True) == []


def test_list_skills_filter_unavailable_includes_when_bin_requirement_met(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    skills_root = workspace / "skills"
    skills_root.mkdir(parents=True)
    skill_path = _write_skill(
        skills_root,
        "has_bin",
        metadata_json={"requires": {"bins": ["nanobot_test_fake_binary"]}},
    )
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    def fake_which(cmd: str) -> str | None:
        if cmd == "nanobot_test_fake_binary":
            return "/fake/nanobot_test_fake_binary"
        return None

    monkeypatch.setattr("nanobot.agent.skills.shutil.which", fake_which)

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    entries = loader.list_skills(filter_unavailable=True)
    assert entries == [
        {"name": "has_bin", "path": str(skill_path), "source": "workspace"},
    ]


def test_list_skills_filter_unavailable_false_keeps_unmet_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    skills_root = workspace / "skills"
    skills_root.mkdir(parents=True)
    skill_path = _write_skill(
        skills_root,
        "blocked",
        metadata_json={"requires": {"bins": ["nanobot_test_fake_binary"]}},
    )
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    monkeypatch.setattr("nanobot.agent.skills.shutil.which", lambda _cmd: None)

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    entries = loader.list_skills(filter_unavailable=False)
    assert entries == [
        {"name": "blocked", "path": str(skill_path), "source": "workspace"},
    ]


def test_list_skills_filter_unavailable_excludes_unmet_env_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    skills_root = workspace / "skills"
    skills_root.mkdir(parents=True)
    _write_skill(
        skills_root,
        "needs_env",
        metadata_json={"requires": {"env": ["NANOBOT_SKILLS_TEST_ENV_VAR"]}},
    )
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    monkeypatch.delenv("NANOBOT_SKILLS_TEST_ENV_VAR", raising=False)

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    assert loader.list_skills(filter_unavailable=True) == []


def test_list_skills_openclaw_metadata_parsed_for_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    skills_root = workspace / "skills"
    skills_root.mkdir(parents=True)
    skill_dir = skills_root / "openclaw_skill"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    oc_payload = json.dumps({"openclaw": {"requires": {"bins": ["nanobot_oc_bin"]}}}, separators=(",", ":"))
    skill_path.write_text(
        "\n".join(["---", f"metadata: {oc_payload}", "---", "", "# OC"]),
        encoding="utf-8",
    )
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    monkeypatch.setattr("nanobot.agent.skills.shutil.which", lambda _cmd: None)

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    assert loader.list_skills(filter_unavailable=True) == []

    monkeypatch.setattr(
        "nanobot.agent.skills.shutil.which",
        lambda cmd: "/x" if cmd == "nanobot_oc_bin" else None,
    )
    entries = loader.list_skills(filter_unavailable=True)
    assert entries == [
        {"name": "openclaw_skill", "path": str(skill_path), "source": "workspace"},
    ]


def test_disabled_skills_excluded_from_list(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    ws_skills = workspace / "skills"
    ws_skills.mkdir(parents=True)
    _write_skill(ws_skills, "alpha", body="# Alpha")
    beta_path = _write_skill(ws_skills, "beta", body="# Beta")
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin, disabled_skills={"alpha"})
    entries = loader.list_skills(filter_unavailable=False)
    assert len(entries) == 1
    assert entries[0]["name"] == "beta"
    assert entries[0]["path"] == str(beta_path)


def test_disabled_skills_empty_set_no_effect(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    ws_skills = workspace / "skills"
    ws_skills.mkdir(parents=True)
    _write_skill(ws_skills, "alpha", body="# Alpha")
    _write_skill(ws_skills, "beta", body="# Beta")
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin, disabled_skills=set())
    entries = loader.list_skills(filter_unavailable=False)
    assert len(entries) == 2


def test_disabled_skills_excluded_from_build_skills_summary(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    ws_skills = workspace / "skills"
    ws_skills.mkdir(parents=True)
    _write_skill(ws_skills, "alpha", body="# Alpha")
    _write_skill(ws_skills, "beta", body="# Beta")
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin, disabled_skills={"alpha"})
    summary = loader.build_skills_summary()
    assert "alpha" not in summary
    assert "beta" in summary


def test_disabled_skills_excluded_from_get_always_skills(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    ws_skills = workspace / "skills"
    ws_skills.mkdir(parents=True)
    _write_skill(ws_skills, "alpha", metadata_json={"always": True}, body="# Alpha")
    _write_skill(ws_skills, "beta", metadata_json={"always": True}, body="# Beta")
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin, disabled_skills={"alpha"})
    always = loader.get_always_skills()
    assert "alpha" not in always
    assert "beta" in always


# -- multiline description tests (YAML folded > and literal |) -----------------


def test_build_skills_summary_folded_description(tmp_path: Path) -> None:
    """description: > (YAML folded scalar) should be parsed correctly."""
    workspace = tmp_path / "ws"
    ws_skills = workspace / "skills"
    ws_skills.mkdir(parents=True)
    skill_dir = ws_skills / "pdf"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\n"
        "name: pdf\n"
        "description: >\n"
        "  Use this skill when visual quality and design identity matter for a PDF.\n"
        "  CREATE (generate from scratch): \"make a PDF\".\n"
        "---\n\n# PDF Skill\n",
        encoding="utf-8",
    )
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    summary = loader.build_skills_summary()
    assert "pdf" in summary
    assert "visual quality" in summary


def test_build_skills_summary_literal_description(tmp_path: Path) -> None:
    """description: | (YAML literal scalar) should be parsed correctly."""
    workspace = tmp_path / "ws"
    ws_skills = workspace / "skills"
    ws_skills.mkdir(parents=True)
    skill_dir = ws_skills / "multi"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\n"
        "name: multi\n"
        "description: |\n"
        "  Line one of description.\n"
        "  Line two of description.\n"
        "---\n\n# Multi\n",
        encoding="utf-8",
    )
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    meta = loader.get_skill_metadata("multi")
    assert meta is not None
    desc = meta.get("description")
    assert isinstance(desc, str)
    assert "Line one" in desc
    assert "Line two" in desc


def test_get_skill_metadata_handles_yaml_types(tmp_path: Path) -> None:
    """yaml.safe_load returns native types; always should be True, not 'true'."""
    workspace = tmp_path / "ws"
    ws_skills = workspace / "skills"
    ws_skills.mkdir(parents=True)
    skill_dir = ws_skills / "typed"
    skill_dir.mkdir(parents=True)
    payload = json.dumps({"nanobot": {"requires": {"bins": ["gh"]}, "always": True}}, separators=(",", ":"))
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\n"
        "name: typed\n"
        f"metadata: {payload}\n"
        "always: true\n"
        "---\n\n# Typed\n",
        encoding="utf-8",
    )
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    meta = loader.get_skill_metadata("typed")
    assert meta is not None
    # YAML parsed 'true' to Python True
    assert meta.get("always") is True
    # metadata is a parsed dict, not a JSON string
    assert isinstance(meta.get("metadata"), dict)


# -----------------------------------------------------------------------------
# Helpers for create/edit/patch/delete tests
# -----------------------------------------------------------------------------


def _skill_md(name: str, description: str, body: str = "# Skill body") -> str:
    """Return a valid SKILL.md string."""
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"


# -----------------------------------------------------------------------------
# create_skill
# -----------------------------------------------------------------------------

def test_create_skill_basic(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    result = loader.create_skill("my-skill", _skill_md("my-skill", "A test skill"))
    assert result["success"] is True
    assert result["message"] == "Skill 'my-skill' created."
    assert (workspace / "skills" / "my-skill" / "SKILL.md").exists()

    # reload and verify
    content = loader.load_skill("my-skill")
    assert content is not None
    assert "A test skill" in content


def test_create_skill_with_category(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    result = loader.create_skill("my-skill", _skill_md("my-skill", "In a category"), category="ml")
    assert result["success"] is True
    assert result.get("category") == "ml"
    assert (workspace / "skills" / "ml" / "my-skill" / "SKILL.md").exists()


def test_create_skill_name_collision(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    loader.create_skill("dup", _skill_md("dup", "First"))
    result = loader.create_skill("dup", _skill_md("dup", "Second"))
    assert result["success"] is False
    assert "already exists" in result["error"]


def test_create_skill_invalid_name_empty(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    result = loader.create_skill("", _skill_md("x", "desc"))
    assert result["success"] is False
    assert "required" in result["error"]


def test_create_skill_invalid_name_uppercase(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    result = loader.create_skill("Invalid-Name", _skill_md("Invalid-Name", "desc"))
    assert result["success"] is False
    assert "Invalid skill name" in result["error"]


def test_create_skill_invalid_name_too_long(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    long_name = "a" * 65
    result = loader.create_skill(long_name, _skill_md(long_name, "desc"))
    assert result["success"] is False
    assert "exceeds" in result["error"]


def test_create_skill_missing_frontmatter(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    result = loader.create_skill("no-fm", "just plain text no frontmatter")
    assert result["success"] is False
    assert "frontmatter" in result["error"]


def test_create_skill_missing_name_in_frontmatter(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    result = loader.create_skill("x", "---\ndescription: has no name field\n---\n\nBody")
    assert result["success"] is False
    assert "'name' field" in result["error"]


def test_create_skill_missing_description_in_frontmatter(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    result = loader.create_skill("x", "---\nname: x\n---\n\nBody")
    assert result["success"] is False
    assert "'description' field" in result["error"]


def test_create_skill_empty_body(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    result = loader.create_skill("x", "---\nname: x\ndescription: desc\n---\n")
    assert result["success"] is False
    assert "content after the frontmatter" in result["error"]


def test_create_skill_builtin_collision(tmp_path: Path) -> None:
    """Creating a skill that shadows a builtin is allowed (workspace wins)."""
    workspace = tmp_path / "ws"
    ws_skills = workspace / "skills"
    ws_skills.mkdir(parents=True)
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    _write_skill(builtin, "shadow-test", body="# Builtin")

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    result = loader.create_skill("shadow-test", _skill_md("shadow-test", "Workspace version"))
    assert result["success"] is True
    # workspace version loaded
    content = loader.load_skill("shadow-test")
    assert "Workspace version" in content


# -----------------------------------------------------------------------------
# edit_skill
# -----------------------------------------------------------------------------

def test_edit_skill_basic(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    loader.create_skill("editable", _skill_md("editable", "Original"))
    result = loader.edit_skill("editable", _skill_md("editable", "Updated description", "# Updated body"))
    assert result["success"] is True
    assert result["message"] == "Skill 'editable' updated."

    content = loader.load_skill("editable")
    assert "Updated description" in content
    assert "Updated body" in content


def test_edit_skill_not_found(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    result = loader.edit_skill("does-not-exist", _skill_md("x", "desc"))
    assert result["success"] is False
    assert "not found" in result["error"]


def test_edit_skill_builtin_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    _write_skill(builtin, "builtin-only", body="# Builtin")
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    result = loader.edit_skill("builtin-only", _skill_md("builtin-only", "Try to edit"))
    assert result["success"] is False
    assert "builtin" in result["error"].lower()


def test_edit_skill_invalid_frontmatter(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    loader.create_skill("bad-edit", _skill_md("bad-edit", "Original"))
    result = loader.edit_skill("bad-edit", "no frontmatter at all")
    assert result["success"] is False


# -----------------------------------------------------------------------------
# patch_skill
# -----------------------------------------------------------------------------

def test_patch_skill_basic(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    loader.create_skill("patchable", _skill_md("patchable", "Original description", "# Old step\n\nDo the thing."))
    result = loader.patch_skill("patchable", old_string="Do the thing.", new_string="Do the better thing.")
    assert result["success"] is True
    assert result["match_count"] == 1

    content = loader.load_skill("patchable")
    assert "Do the better thing." in content
    assert "Do the thing." not in content


def test_patch_skill_delete_with_empty_string(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    loader.create_skill("remover", _skill_md("remover", "Desc", "# Old\n\nRemove this line.\n\nEnd."))
    result = loader.patch_skill("remover", old_string="\nRemove this line.\n", new_string="")
    assert result["success"] is True

    content = loader.load_skill("remover")
    assert "Remove this line" not in content


def test_patch_skill_replace_all(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    loader.create_skill("replace-all", _skill_md("replace-all", "Desc", "# Step 1\n\nfoo\n# Step 2\n\nfoo\n# End"))
    result = loader.patch_skill("replace-all", old_string="foo", new_string="bar", replace_all=True)
    assert result["success"] is True
    assert result["match_count"] == 2

    content = loader.load_skill("replace-all")
    assert content.count("bar") == 2
    assert content.count("foo") == 0


def test_patch_skill_not_unique_without_replace_all(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    loader.create_skill("dup-match", _skill_md("dup-match", "Desc", "repeat repeat"))
    result = loader.patch_skill("dup-match", old_string="repeat", new_string="unique")
    assert result["success"] is False
    assert "not unique" in result["error"]


def test_patch_skill_not_found(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    result = loader.patch_skill("ghost", old_string="x", new_string="y")
    assert result["success"] is False
    assert "not found" in result["error"]


def test_patch_skill_builtin_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    _write_skill(builtin, "bi-patch", body="# BI")
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    result = loader.patch_skill("bi-patch", old_string="BI", new_string="modified")
    assert result["success"] is False
    assert "builtin" in result["error"].lower()


def test_patch_skill_old_string_not_found(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    loader.create_skill("no-match", _skill_md("no-match", "Desc", "# Body"))
    result = loader.patch_skill("no-match", old_string="this does not exist", new_string="replacement")
    assert result["success"] is False
    assert "not found" in result["error"]
    assert "file_preview" in result


def test_patch_skill_file_path(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    loader.create_skill("with-refs", _skill_md("with-refs", "Desc"))
    ref_dir = workspace / "skills" / "with-refs" / "references"
    ref_dir.mkdir()
    ref_file = ref_dir / "api.md"
    ref_file.write_text("# API Reference\n\nEndpoint: /v1/foo", encoding="utf-8")

    result = loader.patch_skill(
        "with-refs",
        old_string="Endpoint: /v1/foo",
        new_string="Endpoint: /v2/bar",
        file_path="references/api.md",
    )
    assert result["success"] is True
    assert ref_file.read_text(encoding="utf-8") == "# API Reference\n\nEndpoint: /v2/bar"


def test_patch_skill_file_path_not_found(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    loader.create_skill("missing-file", _skill_md("missing-file", "Desc"))
    result = loader.patch_skill("missing-file", old_string="x", new_string="y", file_path="references/nonexistent.md")
    assert result["success"] is False
    assert "not found" in result["error"]


def test_patch_skill_empty_old_string(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    result = loader.patch_skill("any", old_string="", new_string="x")
    assert result["success"] is False
    assert "old_string is required" in result["error"]


def test_patch_skill_none_new_string(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    result = loader.patch_skill("any", old_string="x", new_string=None)  # type: ignore
    assert result["success"] is False
    assert "new_string is required" in result["error"]


def test_patch_skill_breaks_frontmatter(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    loader.create_skill("breaks-fm", _skill_md("breaks-fm", "Desc"))
    # Try to remove the closing --- of frontmatter
    result = loader.patch_skill("breaks-fm", old_string="---\n\n", new_string="\n")
    assert result["success"] is False
    assert "break SKILL.md structure" in result["error"]


# -----------------------------------------------------------------------------
# delete_skill
# -----------------------------------------------------------------------------

def test_delete_skill_basic(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    loader.create_skill("to-delete", _skill_md("to-delete", "Desc"))
    assert loader.load_skill("to-delete") is not None

    result = loader.delete_skill("to-delete")
    assert result["success"] is True
    assert result["message"] == "Skill 'to-delete' deleted."
    assert loader.load_skill("to-delete") is None
    assert not (workspace / "skills" / "to-delete").exists()


def test_delete_skill_category_cleanup(tmp_path: Path) -> None:
    """Empty category directory is removed after deleting its only skill."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    loader.create_skill("lone", _skill_md("lone", "Desc"), category="solo")
    skill_path = workspace / "skills" / "solo" / "lone"
    assert skill_path.exists()

    loader.delete_skill("lone")
    assert not skill_path.exists()


def test_delete_skill_not_found(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    result = loader.delete_skill("nonexistent")
    assert result["success"] is False
    assert "not found" in result["error"]


def test_delete_skill_builtin_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    _write_skill(builtin, "bi-delete", body="# BI")
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    result = loader.delete_skill("bi-delete")
    assert result["success"] is False
    assert "builtin" in result["error"].lower()


# -----------------------------------------------------------------------------
# _find_skill / _is_local_skill / _validate_* helpers
# -----------------------------------------------------------------------------

def test_find_skill_finds_workspace_first(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    ws_skills = workspace / "skills"
    ws_skills.mkdir(parents=True)
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    ws_path = _write_skill(ws_skills, "both", body="# WS")

    _write_skill(builtin, "both", body="# BI")

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    found = loader._find_skill("both")
    assert found is not None
    assert found["path"] == ws_path.parent


def test_find_skill_returns_none_for_missing(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    assert loader._find_skill("does-not-exist") is None


def test_is_local_skill_true_for_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    skill_dir = workspace / "skills" / "my-skill"
    skill_dir.mkdir(parents=True)
    assert loader._is_local_skill(skill_dir) is True


def test_is_local_skill_false_for_builtin(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    skill_dir = builtin / "builtin-skill"
    skill_dir.mkdir(parents=True)
    assert loader._is_local_skill(skill_dir) is False


def test_validate_name_ok(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    assert loader._validate_name("valid-skill") is None
    assert loader._validate_name("valid.skill") is None
    assert loader._validate_name("valid_skill") is None
    assert loader._validate_name("a") is None


def test_validate_name_errors(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    assert "required" in loader._validate_name("")
    assert "exceeds" in loader._validate_name("a" * 65)
    assert "Invalid" in loader._validate_name("Invalid")
    assert "Invalid" in loader._validate_name("has space")
    assert "Invalid" in loader._validate_name("-starts-with-dash")
    assert "Invalid" in loader._validate_name("_starts-with-underscore")


def test_validate_frontmatter_ok(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    assert loader._validate_frontmatter(_skill_md("x", "desc", "# body")) is None


def test_validate_frontmatter_errors(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    assert "cannot be empty" in loader._validate_frontmatter("")
    assert "must start with YAML" in loader._validate_frontmatter("no frontmatter")
    assert "not closed" in loader._validate_frontmatter("---\nname: x\ndescription: y\n")
    assert "parse error" in loader._validate_frontmatter("---\ninvalid: yaml: [\n---\n\nbody")
    assert "'name' field" in loader._validate_frontmatter("---\ndescription: y\n---\n\nbody")
    assert "'description' field" in loader._validate_frontmatter("---\nname: x\n---\n\nbody")
    assert "exceeds" in loader._validate_frontmatter(f"---\nname: x\ndescription: {'x' * 1025}\n---\n\nbody")


def test_validate_content_size_ok(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    assert loader._validate_content_size("x") is None
    assert loader._validate_content_size("a" * 100_000) is None


def test_validate_content_size_over_limit(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    result = loader._validate_content_size("a" * 100_001)
    assert result is not None
    assert "100,001" in result and "100,000" in result
