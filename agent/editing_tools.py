import os
import stat
import tempfile
from pathlib import Path

from .tools import ToolDefinition

MAX_WRITE_FILE_BYTES = 200_000
MAX_PATCH_FILE_BYTES = 1_000_000

WRITE_FILE_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "File path relative to the workspace root",
        },
        "content": {
            "type": "string",
            "description": "Complete UTF-8 text content for the file",
        },
        "create_parent": {
            "type": "boolean",
            "description": "Create missing parent directories",
        },
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}

APPLY_PATCH_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Existing UTF-8 file path relative to the workspace root",
        },
        "old_text": {
            "type": "string",
            "description": "Exact text to replace",
        },
        "new_text": {
            "type": "string",
            "description": "Replacement text",
        },
        "replace_all": {
            "type": "boolean",
            "description": "Replace every occurrence instead of requiring one match",
        },
    },
    "required": ["path", "old_text", "new_text"],
    "additionalProperties": False,
}


def _resolve_write_target(root: Path, path_value: str) -> Path:
    relative_path = Path(path_value)
    if relative_path.is_absolute():
        raise ValueError("path must be relative to the workspace")

    current = root
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symbolic links are not allowed in write paths: {current}")

    target = (root / relative_path).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError("Path is outside the workspace") from error
    return target


def _atomic_write(target: Path, data: bytes, original_mode: int | None) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(data)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        if original_mode is not None:
            temporary_path.chmod(original_mode)

        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def create_write_file_tool(workspace_root: Path) -> ToolDefinition:
    if not workspace_root.is_dir():
        raise ValueError(f"Workspace root {workspace_root} is not a directory")

    root = workspace_root.resolve()

    def write_file_handler(arguments: dict[str, object]) -> str:
        path_value = arguments.get("path")
        content = arguments.get("content")
        create_parent = arguments.get("create_parent", False)

        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError("path must be a non-empty string")
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        if not isinstance(create_parent, bool):
            raise ValueError("create_parent must be a boolean")

        encoded_content = content.encode("utf-8")
        if len(encoded_content) > MAX_WRITE_FILE_BYTES:
            raise ValueError(
                f"content exceeds the {MAX_WRITE_FILE_BYTES}-byte limit"
            )

        target = _resolve_write_target(root, path_value)
        parent = target.parent

        if not parent.exists():
            if not create_parent:
                raise ValueError(f"Parent directory does not exist: {parent}")
            parent.mkdir(parents=True)

        target = _resolve_write_target(root, path_value)
        if not target.parent.is_dir():
            raise ValueError(f"Parent path is not a directory: {target.parent}")
        if target.exists() and not target.is_file():
            raise ValueError(f"Target is not a regular file: {target}")

        existed = target.exists()
        previous_size = target.stat().st_size if existed else 0
        original_mode = (
            stat.S_IMODE(target.stat().st_mode) if existed else None
        )

        _atomic_write(target, encoded_content, original_mode)

        action = "Updated" if existed else "Created"
        relative_target = target.relative_to(root)
        return (
            f"{action}: {relative_target}\n"
            f"Previous size: {previous_size} bytes\n"
            f"New size: {len(encoded_content)} bytes"
        )

    return ToolDefinition(
        name="write_file",
        description="Create or completely replace a UTF-8 text file in the workspace",
        parameters=WRITE_FILE_PARAMETERS,
        handler=write_file_handler,
    )


def create_apply_patch_tool(workspace_root: Path) -> ToolDefinition:
    if not workspace_root.is_dir():
        raise ValueError(f"Workspace root {workspace_root} is not a directory")

    root = workspace_root.resolve()

    def apply_patch_handler(arguments: dict[str, object]) -> str:
        path_value = arguments.get("path")
        old_text = arguments.get("old_text")
        new_text = arguments.get("new_text")
        replace_all = arguments.get("replace_all", False)

        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError("path must be a non-empty string")
        if not isinstance(old_text, str) or not old_text:
            raise ValueError("old_text must be a non-empty string")
        if not isinstance(new_text, str):
            raise ValueError("new_text must be a string")
        if not isinstance(replace_all, bool):
            raise ValueError("replace_all must be a boolean")

        target = _resolve_write_target(root, path_value)
        if not target.is_file():
            raise ValueError(f"File not found: {target}")
        if target.stat().st_size > MAX_PATCH_FILE_BYTES:
            raise ValueError(
                f"file exceeds the {MAX_PATCH_FILE_BYTES}-byte patch limit"
            )

        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("File is not valid UTF-8 text") from error

        occurrences = content.count(old_text)
        if occurrences == 0:
            raise ValueError("old_text was not found in the file")
        if occurrences > 1 and not replace_all:
            raise ValueError(
                "old_text occurs more than once; set replace_all to true or "
                "provide a more specific match"
            )

        replacement_count = occurrences if replace_all else 1
        updated_content = content.replace(
            old_text,
            new_text,
            -1 if replace_all else 1,
        )
        encoded_content = updated_content.encode("utf-8")
        if len(encoded_content) > MAX_PATCH_FILE_BYTES:
            raise ValueError(
                f"patched file exceeds the {MAX_PATCH_FILE_BYTES}-byte limit"
            )

        original_mode = stat.S_IMODE(target.stat().st_mode)
        previous_size = target.stat().st_size
        _atomic_write(target, encoded_content, original_mode)

        relative_target = target.relative_to(root)
        return (
            f"Patched: {relative_target}\n"
            f"Replacements: {replacement_count}\n"
            f"Previous size: {previous_size} bytes\n"
            f"New size: {len(encoded_content)} bytes"
        )

    return ToolDefinition(
        name="apply_patch",
        description="Replace exact text in an existing UTF-8 workspace file",
        parameters=APPLY_PATCH_PARAMETERS,
        handler=apply_patch_handler,
    )
