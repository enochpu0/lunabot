"""Skills loader for agent capabilities."""

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import yaml

from nanobot.agent.tools.skills_guard import scan_skill, format_scan_report, should_allow_install
from nanobot.runtime_context import RuntimeContextBlock

logger = logging.getLogger(__name__)

# Default builtin skills directory (relative to this file)
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"

# Opening ---, YAML body (group 1), closing --- on its own line; supports CRLF.
_STRIP_SKILL_FRONTMATTER = re.compile(
    r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?",
    re.DOTALL,
)
_SKILL_NAME = re.compile(r"^(?!.*--)[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_SKILL_REFERENCE = re.compile(r"(?<![\w$])\$([A-Za-z0-9_-]+)")


def parse_skill_metadata(content: str) -> dict[str, object] | None:
    """Parse a skill document's YAML frontmatter."""
    if not (match := _STRIP_SKILL_FRONTMATTER.match(content)):
        return None
    try:
        parsed = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None
    if not isinstance(parsed, dict):
        return None
    return {str(key): value for key, value in cast(dict[object, object], parsed).items()}


def valid_skill_metadata(metadata: dict[str, object], name: str) -> bool:
    """Return whether metadata satisfies the Agent Skills identity contract."""
    description = metadata.get("description")
    return (
        metadata.get("name") == name
        and len(name) <= 64
        and _SKILL_NAME.fullmatch(name) is not None
        and isinstance(description, str)
        and 1 <= len(description.strip()) <= 1024
    )

# Characters allowed in skill names (filesystem-safe, URL-friendly)
VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_SKILL_CONTENT_CHARS = 100_000


class SkillsLoader:
    """
    Loader for agent skills.

    Skills are markdown files (SKILL.md) that teach the agent how to use
    specific tools or perform certain tasks.
    """

    def __init__(self, workspace: Path, builtin_skills_dir: Path | None = None, disabled_skills: set[str] | None = None):
        self.workspace = workspace
        self.workspace_skills = workspace / "skills"
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR
        self.disabled_skills = disabled_skills or set()

    def _skill_aliases(self) -> dict[str, str]:
        """Return compatibility aliases owned by installed CLI Apps."""
        from nanobot.apps.cli import CliAppManager

        try:
            return CliAppManager(workspace=self.workspace).installed_skill_aliases()
        except OSError:
            return {}

    def _skill_entries_from_dir(self, base: Path, source: str, *, skip_names: set[str] | None = None) -> list[dict[str, str]]:
        if not base.exists():
            return []
        entries: list[dict[str, str]] = []
        for skill_dir in base.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            name = skill_dir.name
            if skip_names is not None and name in skip_names:
                continue
            entries.append({"name": name, "path": str(skill_file), "source": source})
        return entries

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """
        List all available skills.

        Args:
            filter_unavailable: If True, filter out skills with unmet requirements.

        Returns:
            List of skill info dicts with 'name', 'path', 'source'.
        """
        from nanobot.agent.plugins import enabled_agent_plugin_skills

        plugin_skills = enabled_agent_plugin_skills(self.workspace)
        skills = self._skill_entries_from_dir(self.workspace_skills, "workspace")
        seen_names = {entry["name"] for entry in skills}
        for name, path in plugin_skills:
            if name in seen_names:
                continue
            skills.append(
                {
                    "name": name,
                    "path": str(path),
                    "source": "plugin",
                }
            )
            seen_names.add(name)
        if self.builtin_skills and self.builtin_skills.exists():
            skills.extend(
                self._skill_entries_from_dir(self.builtin_skills, "builtin", skip_names=seen_names)
            )

        if self.disabled_skills:
            disabled = set(self.disabled_skills)
            for legacy, canonical in self._skill_aliases().items():
                if legacy in disabled or canonical in disabled:
                    disabled.update((legacy, canonical))
            skills = [s for s in skills if s["name"] not in disabled]

        if filter_unavailable:
            return [skill for skill in skills if self._check_requirements(self._get_skill_meta(skill["name"]))]
        return skills

    def load_skill(self, name: str) -> str | None:
        """
        Load a skill by name.

        Args:
            name: Skill name (directory name).

        Returns:
            Skill content or None if not found.
        """
        skills = self.list_skills(filter_unavailable=False)
        available = {skill["name"] for skill in skills}
        resolved = name if name in available else self._skill_aliases().get(name, name)
        entry = next((skill for skill in skills if skill["name"] == resolved), None)
        return Path(entry["path"]).read_text(encoding="utf-8") if entry else None

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """
        Load specific skills for inclusion in agent context.

        Args:
            skill_names: List of skill names to load.

        Returns:
            Formatted skills content.
        """
        parts = [
            f"### Skill: {name}\n\n{self._strip_frontmatter(markdown)}"
            for name in skill_names
            if (markdown := self.load_skill(name))
        ]
        return "\n\n---\n\n".join(parts)

    def get_explicitly_invoked_skills(self, text: str) -> list[str]:
        """Resolve ``$skill-name`` references to enabled, available skills."""
        if not text:
            return []
        available = {
            entry["name"]
            for entry in self.list_skills(filter_unavailable=True)
        }
        aliases = self._skill_aliases()
        invoked: list[str] = []
        for match in _SKILL_REFERENCE.finditer(text):
            requested = match.group(1)
            name = requested if requested in available else aliases.get(requested, requested)
            if name in available and name not in invoked:
                invoked.append(name)
        return invoked

    def build_explicit_skill_runtime_context(
        self,
        text: str,
    ) -> RuntimeContextBlock | None:
        """Load non-always skills explicitly invoked by the current message."""
        skill_names = self.get_explicitly_invoked_skills(text)
        if not skill_names:
            return None
        always_active = set(self.get_always_skills())
        skill_names = [name for name in skill_names if name not in always_active]
        content = self.load_skills_for_context(skill_names)
        if not content:
            return None
        return RuntimeContextBlock(
            source="explicit_skills",
            content=(
                "[Active Skills — instructions for this user turn]\n"
                f"{content}\n"
                "[/Active Skills]"
            ),
        )

    def build_skills_summary(
        self,
        exclude: set[str] | None = None,
        *,
        workspace: Path | None = None,
    ) -> str:
        """
        Build a summary of all skills (name, description, path, availability).

        This is used for progressive loading - the agent can read the full
        skill content using read_file when needed.

        Args:
            exclude: Set of skill names to omit from the summary.
            workspace: Effective project workspace used to choose safe display paths.

        Returns:
            Markdown-formatted skills summary.
        """
        all_skills = self.list_skills(filter_unavailable=False)
        if not all_skills:
            return ""

        agent_workspace = self.workspace.expanduser().resolve()
        project_workspace = (workspace or self.workspace).expanduser().resolve()
        use_relative_roots = project_workspace == agent_workspace
        sections: list[str] = []
        groups = (
            ("Workspace skills", "workspace", self.workspace_skills),
            ("Agent Plugin skills", "plugin", self.workspace / "plugins"),
            ("Built-in skills", "builtin", self.builtin_skills),
        )
        for label, source, root in groups:
            entries = [
                entry
                for entry in all_skills
                if entry["source"] == source and (not exclude or entry["name"] not in exclude)
            ]
            if not entries:
                continue

            resolved_root = root.expanduser().resolve()
            if use_relative_roots:
                display_root = Path("plugins" if source == "plugin" else "skills")
            else:
                display_root = resolved_root
            lines = [f"### {label} (`{display_root}`)"]
            for entry in entries:
                skill_name = entry["name"]
                meta = self._get_skill_meta(skill_name)
                available = self._check_requirements(meta)
                desc = self.get_skill_description(skill_name)
                suffix = ""
                if not available:
                    missing = self._get_missing_requirements(meta)
                    suffix = f" (unavailable: {missing})" if missing else " (unavailable)"
                relative_path = Path(entry["path"]).relative_to(root).as_posix()
                lines.append(f"- **{skill_name}** — {desc}{suffix}  `{relative_path}`")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    @staticmethod
    def _requirement_lists(skill_meta: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Return (bins, env) lists from skill metadata, tolerating null/wrong shapes."""
        requires = cast(dict[str, Any], skill_meta.get("requires") or {})
        if not isinstance(skill_meta.get("requires") or {}, dict):
            return [], []
        bins_raw: object = requires.get("bins") or []
        env_raw: object = requires.get("env") or []
        bins = [value for value in cast(list[object], bins_raw) if isinstance(value, str) and value.strip()] if isinstance(bins_raw, list) else []
        env = [value for value in cast(list[object], env_raw) if isinstance(value, str) and value.strip()] if isinstance(env_raw, list) else []
        return bins, env

    def _get_missing_requirements(self, skill_meta: dict[str, Any]) -> str:
        """Get a description of missing requirements."""
        required_bins, required_env_vars = self._requirement_lists(skill_meta)
        return ", ".join(
            [f"CLI: {command_name}" for command_name in required_bins if not shutil.which(command_name)]
            + [f"ENV: {env_name}" for env_name in required_env_vars if not os.environ.get(env_name)]
        )

    def get_skill_availability(self, name: str) -> tuple[bool, str]:
        """Return whether a skill can run and why not when it cannot."""
        meta = self._get_skill_meta(name)
        available = self._check_requirements(meta)
        return available, "" if available else self._get_missing_requirements(meta)

    def get_skill_requirements(self, name: str) -> dict[str, list[str]]:
        """Return explicit command/env requirements and currently missing entries."""
        bins, env = self._requirement_lists(self._get_skill_meta(name))
        return {
            "bins": bins,
            "env": env,
            "missing_bins": [value for value in bins if not shutil.which(value)],
            "missing_env": [value for value in env if not os.environ.get(value)],
        }

    def get_skill_description(self, name: str) -> str:
        """Get the description of a skill from its frontmatter."""
        meta = self.get_skill_metadata(name)
        description = meta.get("description") if meta else None
        if isinstance(description, str) and description:
            return description
        return name  # Fallback to skill name

    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from markdown content."""
        if not content.startswith("---"):
            return content
        match = _STRIP_SKILL_FRONTMATTER.match(content)
        if match:
            return content[match.end():].strip()
        return content

    def _parse_nanobot_metadata(self, raw: object) -> dict[str, Any]:
        """Extract nanobot/openclaw metadata from a frontmatter field.

        ``raw`` may be a dict (already parsed by yaml.safe_load) or a JSON str.
        """
        if isinstance(raw, dict):
            data = cast(dict[str, Any], raw)
        elif isinstance(raw, str):
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {}
        else:
            return {}
        if not isinstance(data, dict):
            return {}
        data_object = cast(dict[str, Any], data)
        payload = data_object.get("nanobot", data_object.get("openclaw", {}))
        return cast(dict[str, Any], payload) if isinstance(payload, dict) else {}

    def _check_requirements(self, skill_meta: dict[str, Any]) -> bool:
        """Check if skill requirements are met (bins, env vars)."""
        required_bins, required_env_vars = self._requirement_lists(skill_meta)
        return all(shutil.which(cmd) for cmd in required_bins) and all(
            os.environ.get(var) for var in required_env_vars
        )

    def _get_skill_meta(self, name: str) -> dict[str, Any]:
        """Get nanobot metadata for a skill (cached in frontmatter)."""
        raw_meta = self.get_skill_metadata(name) or {}
        return self._parse_nanobot_metadata(raw_meta.get("metadata"))

    def get_always_skills(self) -> list[str]:
        """Get skills marked as always=true that meet requirements."""
        return [
            entry["name"]
            for entry in self.list_skills(filter_unavailable=True)
            if (meta := self.get_skill_metadata(entry["name"]) or {})
            and (
                self._parse_nanobot_metadata(meta.get("metadata")).get("always")
                or meta.get("always")
            )
        ]

    def _validate_name(self, name: str) -> str | None:
        """Validate a skill name. Returns error message or None if valid."""
        if not name:
            return "Skill name is required."
        if len(name) > MAX_NAME_LENGTH:
            return f"Skill name exceeds {MAX_NAME_LENGTH} characters."
        if not VALID_NAME_RE.match(name):
            return (
                f"Invalid skill name '{name}'. Use lowercase letters, numbers, "
                "hyphens, dots, and underscores. Must start with a letter or digit."
            )
        return None

    def _validate_frontmatter(self, content: str) -> str | None:
        """
        Validate that SKILL.md content has proper frontmatter with required fields.
        Returns error message or None if valid.
        """
        if not content.strip():
            return "Content cannot be empty."

        if not content.startswith("---"):
            return "SKILL.md must start with YAML frontmatter (---)."

        end_match = re.search(r"\n---\s*\n", content[3:])
        if not end_match:
            return "SKILL.md frontmatter is not closed. Ensure you have a closing '---' line."

        yaml_content = content[3 : end_match.start() + 3]

        try:
            parsed = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            return f"YAML frontmatter parse error: {e}"

        if not isinstance(parsed, dict):
            return "Frontmatter must be a YAML mapping (key: value pairs)."

        if "name" not in parsed:
            return "Frontmatter must include 'name' field."
        if "description" not in parsed:
            return "Frontmatter must include 'description' field."
        if len(str(parsed["description"])) > MAX_DESCRIPTION_LENGTH:
            return f"Description exceeds {MAX_DESCRIPTION_LENGTH} characters."

        body = content[end_match.end() + 3 :].strip()
        if not body:
            return "SKILL.md must have content after the frontmatter (instructions, procedures, etc.)."

        return None

    def _validate_content_size(self, content: str, label: str = "SKILL.md") -> str | None:
        """Check that content doesn't exceed the character limit."""
        if len(content) > MAX_SKILL_CONTENT_CHARS:
            return (
                f"{label} content is {len(content):,} characters "
                f"(limit: {MAX_SKILL_CONTENT_CHARS:,})."
            )
        return None

    def _atomic_write_text(self, file_path: Path, content: str) -> None:
        """Atomically write text content to a file using a temp file + os.replace."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            dir=str(file_path.parent),
            prefix=f".{file_path.name}.tmp.",
            suffix="",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(temp_path, file_path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                logger.error("Failed to remove temporary file %s during atomic write", temp_path)
            raise

    def _is_local_skill(self, skill_path: Path) -> bool:
        """Check if a skill path is within the workspace skills dir (user-editable)."""
        try:
            skill_path.resolve().relative_to(self.workspace_skills.resolve())
            return True
        except ValueError:
            return False

    def _find_skill(self, name: str) -> dict | None:
        """Find a skill by name across workspace and builtin dirs (including category subdirs). Returns {path} or None."""
        for base in [self.workspace_skills, self.builtin_skills]:
            if base and base.exists():
                # Flat lookup first
                skill_dir = base / name
                if skill_dir.exists() and (skill_dir / "SKILL.md").exists():
                    return {"path": skill_dir}
                # Recursive lookup for skills in category subdirectories
                for skill_md in base.rglob("SKILL.md"):
                    if skill_md.parent.name == name:
                        return {"path": skill_md.parent}
        return None

    def create_skill(self, name: str, content: str, category: str | None = None) -> dict[str, object]:
        """
        Create a new skill in the workspace skills directory.

        Args:
            name: Skill name (directory name).
            content: Full SKILL.md content (YAML frontmatter + markdown body).
            category: Optional category subdirectory.

        Returns:
            Result dict with success, message, and path.
        """
        err = self._validate_name(name)
        if err:
            return {"success": False, "error": err}

        err = self._validate_frontmatter(content)
        if err:
            return {"success": False, "error": err}

        err = self._validate_content_size(content)
        if err:
            return {"success": False, "error": err}

        # Only block workspace-level collision (shadowing builtins is allowed)
        ws_skill_dir = self.workspace_skills / name
        if ws_skill_dir.exists() and (ws_skill_dir / "SKILL.md").exists():
            return {
                "success": False,
                "error": f"A skill named '{name}' already exists at {ws_skill_dir}.",
            }


        if category:
            skill_dir = self.workspace_skills / category / name
        else:
            skill_dir = self.workspace_skills / name

        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        try:
            self._atomic_write_text(skill_md, content)
        except Exception as e:
            shutil.rmtree(skill_dir, ignore_errors=True)
            return {"success": False, "error": f"Failed to write skill: {e}"}

        result: dict[str, object] = {
            "success": True,
            "message": f"Skill '{name}' created.",
            "path": str(skill_dir),
            "skill_md": str(skill_md),
        }
        if category:
            result["category"] = category

        # Security scan
        try:
            scan_result = scan_skill(skill_dir, source="agent-created")
            result["scan_verdict"] = scan_result.verdict
            result["scan_findings"] = len(scan_result.findings)
            if scan_result.findings:
                allowed, reason = should_allow_install(scan_result)
                if allowed is not True:
                    logger.warning(
                        "Security scan for skill '%s': %s\n%s",
                        name, reason, format_scan_report(scan_result),
                    )
                result["scan_report"] = format_scan_report(scan_result)
        except Exception as exc:
            logger.debug("Security scan failed for skill '%s': %s", name, exc)

        return result

    def edit_skill(self, name: str, content: str) -> dict[str, object]:
        """
        Replace the SKILL.md content of an existing skill (full rewrite).

        Args:
            name: Skill name to edit.
            content: Full new SKILL.md content.

        Returns:
            Result dict with success, message, and path.
        """
        err = self._validate_frontmatter(content)
        if err:
            return {"success": False, "error": err}

        err = self._validate_content_size(content)
        if err:
            return {"success": False, "error": err}

        existing = self._find_skill(name)
        if not existing:
            return {"success": False, "error": f"Skill '{name}' not found."}

        if not self._is_local_skill(existing["path"]):
            return {
                "success": False,
                "error": f"Skill '{name}' is in the builtin directory and cannot be modified. "
                "Copy it to your workspace skills directory first.",
            }

        skill_md = existing["path"] / "SKILL.md"
        original_content = skill_md.read_text(encoding="utf-8") if skill_md.exists() else None
        try:
            self._atomic_write_text(skill_md, content)
        except Exception as e:
            if original_content is not None:
                self._atomic_write_text(skill_md, original_content)
            return {"success": False, "error": f"Failed to write skill: {e}"}

        result: dict[str, object] = {
            "success": True,
            "message": f"Skill '{name}' updated.",
            "path": str(existing["path"]),
        }

        # Security scan
        try:
            scan_result = scan_skill(existing["path"], source="agent-created")
            result["scan_verdict"] = scan_result.verdict
            result["scan_findings"] = len(scan_result.findings)
            if scan_result.findings:
                allowed, reason = should_allow_install(scan_result)
                if allowed is not True:
                    logger.warning(
                        "Security scan for skill '%s': %s\n%s",
                        name, reason, format_scan_report(scan_result),
                    )
                result["scan_report"] = format_scan_report(scan_result)
        except Exception as exc:
            logger.debug("Security scan failed for skill '%s': %s", name, exc)

        return result

    def patch_skill(
        self,
        name: str,
        old_string: str,
        new_string: str,
        file_path: str | None = None,
        replace_all: bool = False,
    ) -> dict[str, object]:
        """
        Targeted find-and-replace within a skill file.

        Uses fuzzy matching to handle whitespace/indentation differences.
        Defaults to SKILL.md; use file_path to patch a supporting file.

        Args:
            name: Skill name.
            old_string: Text to find. Must be unique unless replace_all=True.
            new_string: Replacement text (can be empty to delete).
            file_path: Optional supporting file path (under references/, templates/, scripts/, assets/).
            replace_all: If True, replace all occurrences.

        Returns:
            Result dict with success, message, and match_count.
        """
        if not old_string:
            return {"success": False, "error": "old_string is required for 'patch'."}
        if new_string is None:
            return {"success": False, "error": "new_string is required for 'patch'."}

        existing = self._find_skill(name)
        if not existing:
            return {"success": False, "error": f"Skill '{name}' not found."}

        if not self._is_local_skill(existing["path"]):
            return {
                "success": False,
                "error": f"Skill '{name}' is in the builtin directory and cannot be modified.",
            }

        skill_dir = existing["path"]

        if file_path:
            target = skill_dir / file_path
        else:
            target = skill_dir / "SKILL.md"

        if not target.exists():
            return {"success": False, "error": f"File not found: {target.relative_to(skill_dir)}"}

        content = target.read_text(encoding="utf-8")

        if replace_all:
            match_count = content.count(old_string)
            if match_count == 0:
                return {
                    "success": False,
                    "error": f"old_string not found in {target.name}.",
                    "file_preview": content[:500],
                }
            new_content = content.replace(old_string, new_string)
        else:
            idx = content.find(old_string)
            if idx == -1:
                return {
                    "success": False,
                    "error": f"old_string not found in {target.name}.",
                    "file_preview": content[:500],
                }
            if content.count(old_string) > 1:
                return {
                    "success": False,
                    "error": "old_string is not unique. Use replace_all=True to replace all occurrences.",
                }
            new_content = content[:idx] + new_string + content[idx + len(old_string) :]
            match_count = 1

        err = self._validate_content_size(new_content, label=target.name if file_path else "SKILL.md")
        if err:
            return {"success": False, "error": err}

        if not file_path:
            err = self._validate_frontmatter(new_content)
            if err:
                return {"success": False, "error": f"Patch would break SKILL.md structure: {err}"}

        try:
            self._atomic_write_text(target, new_content)
        except Exception as e:
            return {"success": False, "error": f"Failed to write patch: {e}"}

        return {
            "success": True,
            "message": f"Patched {target.name if not file_path else file_path} in skill '{name}' ({match_count} replacement{'s' if match_count > 1 else ''}).",
            "match_count": match_count,
        }

    def delete_skill(self, name: str) -> dict[str, object]:
        """
        Delete a workspace skill entirely.

        Args:
            name: Skill name to delete.

        Returns:
            Result dict with success and message.
        """
        existing = self._find_skill(name)
        if not existing:
            return {"success": False, "error": f"Skill '{name}' not found."}

        if not self._is_local_skill(existing["path"]):
            return {
                "success": False,
                "error": f"Skill '{name}' is in the builtin directory and cannot be deleted.",
            }

        skill_dir = existing["path"]
        shutil.rmtree(skill_dir)

        parent = skill_dir.parent
        if parent != self.workspace_skills and parent.exists() and not any(parent.iterdir()):
            try:
                parent.rmdir()
            except OSError:
                pass

        return {
            "success": True,
            "message": f"Skill '{name}' deleted.",
        }

    def get_skill_metadata(self, name: str) -> dict[str, object] | None:
        """
        Get metadata from a skill's frontmatter.

        Args:
            name: Skill name.

        Returns:
            Metadata dict or None.
        """
        return parse_skill_metadata(self.load_skill(name) or "")
