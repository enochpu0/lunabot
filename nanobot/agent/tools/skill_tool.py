"""SkillManageTool — agent-managed skill creation, editing, and deletion.

Wraps the SkillsLoader CRUD operations as a Tool subclass so the LLM can
invoke create / edit / patch / delete / write_file / remove_file via tool_call.
"""

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from nanobot.agent.skills import SkillsLoader
from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import (
    BooleanSchema,
    StringSchema,
    tool_parameters_schema,
)

logger = logging.getLogger(__name__)

# Subdirectories allowed for write_file/remove_file
ALLOWED_SUBDIRS = {"references", "templates", "scripts", "assets"}

MAX_SKILL_FILE_BYTES = 1_048_576  # 1 MiB per supporting file


def _validate_file_path(file_path: str) -> str | None:
    """Validate a file path for write_file/remove_file."""
    if not file_path:
        return "file_path is required."

    normalized = Path(file_path)
    parts = normalized.parts

    if any(part == ".." for part in parts):
        return "Path traversal ('..') is not allowed."

    if not parts or parts[0] not in ALLOWED_SUBDIRS:
        allowed = ", ".join(sorted(ALLOWED_SUBDIRS))
        return f"File must be under one of: {allowed}. Got: '{file_path}'"

    if len(parts) < 2:
        return f"Provide a file path, not just a directory. Example: '{parts[0]}/myfile.md'"

    return None


@tool_parameters(
    tool_parameters_schema(
        action=StringSchema(
            "The action to perform: create, edit, patch, delete, write_file, remove_file",
            enum=["create", "edit", "patch", "delete", "write_file", "remove_file"],
        ),
        name=StringSchema(
            "Skill name (lowercase, hyphens/underscores, max 64 chars). "
            "Must match an existing skill for edit/patch/delete/write_file/remove_file.",
        ),
        content=StringSchema(
            "Full SKILL.md content (YAML frontmatter + markdown body). "
            "Required for 'create' and 'edit'.",
            nullable=True,
        ),
        old_string=StringSchema(
            "Text to find in the file (required for 'patch'). Must be unique unless replace_all=true.",
            nullable=True,
        ),
        new_string=StringSchema(
            "Replacement text (required for 'patch'). Can be empty string to delete matched text.",
            nullable=True,
        ),
        replace_all=BooleanSchema(
            description="For 'patch': replace all occurrences instead of requiring a unique match (default: false).",
        ),
        category=StringSchema(
            "Optional category for organizing the skill (e.g. 'devops', 'data-science'). Only used with 'create'.",
            nullable=True,
        ),
        file_path=StringSchema(
            "Path to a supporting file under references/, templates/, scripts/, or assets/. "
            "Required for 'write_file'/'remove_file'. Optional for 'patch' (defaults to SKILL.md).",
            nullable=True,
        ),
        file_content=StringSchema(
            "Content for the supporting file. Required for 'write_file'.",
            nullable=True,
        ),
        required=["action", "name"],
    )
)
class SkillManageTool(Tool):
    """Manage skills (create, update, delete).

    Skills are reusable procedural knowledge — approaches for recurring task types.
    After creating or editing, a security scan is run automatically.
    """

    def __init__(self, skills_loader: SkillsLoader):
        self._loader = skills_loader

    @property
    def name(self) -> str:
        return "skill_manage"

    @property
    def description(self) -> str:
        return (
            "Manage skills (create, update, delete). Skills are your procedural "
            "memory — reusable approaches for recurring task types. "
            "Actions: create (full SKILL.md + optional category), "
            "patch (old_string/new_string — preferred for fixes), "
            "edit (full SKILL.md rewrite — major overhauls only), "
            "delete, write_file, remove_file.\n\n"
            "Create when: complex task succeeded (5+ calls), errors overcome, "
            "user-corrected approach worked, non-trivial workflow discovered, "
            "or user asks you to remember a procedure.\n"
            "Update when: instructions stale/wrong, OS-specific failures, "
            "missing steps or pitfalls found during use.\n\n"
            "Skip for simple one-offs. Confirm with user before creating/deleting."
        )

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        name = kwargs.get("name", "")

        if action == "create":
            return self._handle_create(name, kwargs)
        elif action == "edit":
            return self._handle_edit(name, kwargs)
        elif action == "patch":
            return self._handle_patch(name, kwargs)
        elif action == "delete":
            return self._handle_delete(name)
        elif action == "write_file":
            return self._handle_write_file(name, kwargs)
        elif action == "remove_file":
            return self._handle_remove_file(name, kwargs)
        else:
            return json.dumps({
                "success": False,
                "error": f"Unknown action '{action}'. Use: create, edit, patch, delete, write_file, remove_file",
            }, ensure_ascii=False)

    # -- action handlers ------------------------------------------------------

    def _handle_create(self, name: str, kwargs: dict[str, Any]) -> str:
        content = kwargs.get("content")
        if not content:
            return json.dumps({"success": False, "error": "content is required for 'create'."}, ensure_ascii=False)

        category = kwargs.get("category") or None
        result = self._loader.create_skill(name, content, category=category)
        return json.dumps(result, ensure_ascii=False, default=str)

    def _handle_edit(self, name: str, kwargs: dict[str, Any]) -> str:
        content = kwargs.get("content")
        if not content:
            return json.dumps({"success": False, "error": "content is required for 'edit'."}, ensure_ascii=False)

        result = self._loader.edit_skill(name, content)
        return json.dumps(result, ensure_ascii=False, default=str)

    def _handle_patch(self, name: str, kwargs: dict[str, Any]) -> str:
        old_string = kwargs.get("old_string")
        new_string = kwargs.get("new_string")
        if not old_string:
            return json.dumps({"success": False, "error": "old_string is required for 'patch'."}, ensure_ascii=False)
        if new_string is None:
            return json.dumps({"success": False, "error": "new_string is required for 'patch'."}, ensure_ascii=False)

        file_path = kwargs.get("file_path") or None
        replace_all = bool(kwargs.get("replace_all", False))
        result = self._loader.patch_skill(name, old_string, new_string, file_path=file_path, replace_all=replace_all)
        return json.dumps(result, ensure_ascii=False, default=str)

    def _handle_delete(self, name: str) -> str:
        result = self._loader.delete_skill(name)
        return json.dumps(result, ensure_ascii=False, default=str)

    def _handle_write_file(self, name: str, kwargs: dict[str, Any]) -> str:
        file_path = kwargs.get("file_path")
        file_content = kwargs.get("file_content")
        if not file_path:
            return json.dumps({"success": False, "error": "file_path is required for 'write_file'."}, ensure_ascii=False)
        if file_content is None:
            return json.dumps({"success": False, "error": "file_content is required for 'write_file'."}, ensure_ascii=False)

        err = _validate_file_path(file_path)
        if err:
            return json.dumps({"success": False, "error": err}, ensure_ascii=False)

        # Size check
        content_bytes = len(file_content.encode("utf-8"))
        if content_bytes > MAX_SKILL_FILE_BYTES:
            return json.dumps({
                "success": False,
                "error": f"File content is {content_bytes:,} bytes (limit: {MAX_SKILL_FILE_BYTES:,} bytes).",
            }, ensure_ascii=False)

        existing = self._loader._find_skill(name)
        if not existing:
            return json.dumps({"success": False, "error": f"Skill '{name}' not found. Create it first with action='create'."}, ensure_ascii=False)

        if not self._loader._is_local_skill(existing["path"]):
            return json.dumps({"success": False, "error": f"Skill '{name}' is a builtin and cannot be modified."}, ensure_ascii=False)

        target = existing["path"] / file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_text(file_content, encoding="utf-8")
        except OSError as e:
            return json.dumps({"success": False, "error": f"Failed to write file: {e}"}, ensure_ascii=False)

        return json.dumps({
            "success": True,
            "message": f"File '{file_path}' written to skill '{name}'.",
            "path": str(target),
        }, ensure_ascii=False)

    def _handle_remove_file(self, name: str, kwargs: dict[str, Any]) -> str:
        file_path = kwargs.get("file_path")
        if not file_path:
            return json.dumps({"success": False, "error": "file_path is required for 'remove_file'."}, ensure_ascii=False)

        err = _validate_file_path(file_path)
        if err:
            return json.dumps({"success": False, "error": err}, ensure_ascii=False)

        existing = self._loader._find_skill(name)
        if not existing:
            return json.dumps({"success": False, "error": f"Skill '{name}' not found."}, ensure_ascii=False)

        if not self._loader._is_local_skill(existing["path"]):
            return json.dumps({"success": False, "error": f"Skill '{name}' is a builtin and cannot be modified."}, ensure_ascii=False)

        target = existing["path"] / file_path
        if not target.exists():
            # List available files for the model
            available = []
            for subdir in ALLOWED_SUBDIRS:
                d = existing["path"] / subdir
                if d.exists():
                    for f in d.rglob("*"):
                        if f.is_file():
                            available.append(str(f.relative_to(existing["path"])))
            return json.dumps({
                "success": False,
                "error": f"File '{file_path}' not found in skill '{name}'.",
                "available_files": available if available else None,
            }, ensure_ascii=False)

        target.unlink()
        # Clean up empty parent
        parent = target.parent
        if parent != existing["path"] and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

        return json.dumps({
            "success": True,
            "message": f"File '{file_path}' removed from skill '{name}'.",
        }, ensure_ascii=False)
