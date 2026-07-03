import json
import os
import selectors
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from itertools import islice

from .tools import ToolDefinition

MAX_FILE_ROWS = 200
MAX_FILE_CHARS = 20_000

READ_FILE_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "File path relative to the workspace root"
        },
        "line_index": {"type": "integer", "minimum": 0},
        "way": {
            "type": "string",
            "enum": ["head", "tail", "middle"],
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}

MAX_LIST_ENTRIES = 200

LIST_FILES_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Directory path relative to the workspace root"
        },
        "depth": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "Maximum directory traversal depth",
        },
    },
    "required": [],
    "additionalProperties": False,
}

MAX_SEARCH_RESULTS = 100
MAX_SEARCH_FILE_BYTES = 1_000_000
MAX_MATCH_LINE_CHARS = 500
SEARCH_TIMEOUT_SECONDS = 10

SEARCH_TEXT_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Literal text to search for",
        },
        "path": {
            "type": "string",
            "description": "File or directory relative to the workspace root",
        },
        "case_sensitive": {
            "type": "boolean",
            "description": "Whether matching is case-sensitive",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

def create_read_file_tool(workspace_root: Path) -> ToolDefinition:
    if not workspace_root.is_dir():
        raise ValueError(f"Workspace root {workspace_root} is not a directory")
    
    def read_file_handler(arguments: dict[str, object]) -> str:

        path_value = arguments.get("path")
        line_index = arguments.get("line_index", 0)
        way = arguments.get("way", "head")

        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError("path must be a non-empty string")

        if not isinstance(line_index, int) or isinstance(line_index, bool):
            raise ValueError("line_index must be an integer")

        if line_index < 0:
            raise ValueError("line_index must not be negative")

        if not isinstance(way, str):
            raise ValueError("way must be a string")

        root = workspace_root.resolve()
        target = (root / path_value).resolve()

        try:
            target.relative_to(root)
        except ValueError as error:
            raise ValueError("Path is outside the workspace") from error

        if not target.is_file():
            raise ValueError(f"File not found: {target}")

        if way == "head":
            start = line_index
            end = line_index + MAX_FILE_ROWS

        elif way == "tail":
            start = max(0, line_index - MAX_FILE_ROWS + 1)
            end = line_index + 1

        elif way == "middle":
            before = MAX_FILE_ROWS // 2
            start = max(0, line_index - before)
            end = start + MAX_FILE_ROWS

        else:
            raise ValueError("way must be 'head', 'tail', or 'middle'")

        with target.open(encoding="utf-8") as file:
            selected_lines = islice(file, start, end)

            output: list[str] = []
            current_chars = 0
            truncated = False

            for actual_index, line in enumerate(selected_lines, start=start):
                numbered_line = f"{actual_index + 1}: {line}"
                remaining = MAX_FILE_CHARS - current_chars

                if len(numbered_line) > remaining:
                    output.append(numbered_line[:remaining])
                    truncated = True
                    break

                output.append(numbered_line)
                current_chars += len(numbered_line)

            if truncated:
                output.append("\n[Output truncated]")

            return "".join(output)

    return ToolDefinition(
        name="read_file",
        description="Read a text file inside the workspace",
        parameters=READ_FILE_PARAMETERS,
        handler=read_file_handler,
    )



def create_list_files_tool(workspace_root: Path) -> ToolDefinition:

    if not workspace_root.is_dir():
        raise ValueError(f"Workspace root {workspace_root} is not a directory") 

    def list_files_handler(arguments: dict[str, object]) -> str:

        path_value = arguments.get("path", ".")
        depth = arguments.get("depth", 2)

        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError("path must be a non-empty string")
        
        if not isinstance(depth, int) or isinstance(depth, bool):
            raise ValueError("depth must be an integer")

        if not 1 <= depth <= 5:
            raise ValueError("depth must be between 1 and 5")

        root = workspace_root.resolve()
        target = (root / path_value).resolve()

        try:
            target.relative_to(root)
        except ValueError as error:
            raise ValueError("Path is outside the workspace") from error

        if not target.is_dir():
            raise ValueError(f"Directory not found: {target}")

        entries: list[str] = []
        scan_limit = MAX_LIST_ENTRIES + 1

        def walk(directory: Path, current_depth: int) -> None:
            if current_depth > depth or len(entries) >= scan_limit:
                return

            children = sorted(directory.iterdir(), key=lambda item: item.name.casefold())

            for child in children:
                if len(entries) >= scan_limit:
                    return

                relative_path = child.relative_to(root)

                if child.is_symlink():
                    entries.append(f"{relative_path} -> [symlink]")
                    continue

                if child.is_dir():
                    entries.append(f"{relative_path}/")
                    walk(child, current_depth + 1)
                else:
                    entries.append(str(relative_path))

        walk(target, 1)        

        if not entries:
            return "[Directory is empty]"

        truncated = len(entries) > MAX_LIST_ENTRIES
        entries = entries[:MAX_LIST_ENTRIES]

        if truncated:
            entries.append("[Output truncated]")

        return "\n".join(entries)

    return ToolDefinition(
        name="list_files",
        description="List files and directories inside the workspace",
        parameters=LIST_FILES_PARAMETERS,
        handler=list_files_handler,
    )



def create_search_text_tool(workspace_root: Path) -> ToolDefinition:

    if not workspace_root.is_dir():
        raise ValueError(f"Workspace root {workspace_root} is not a directory")
    
    rg_path = shutil.which("rg")
    if rg_path is None:
        raise RuntimeError("rg command is not available")

    def search_text_handle(arguments: dict[str, object]) -> str:

        query = arguments.get("query")
        path_value = arguments.get("path", ".")
        case_sensitive = arguments.get("case_sensitive", False)

        if not isinstance(query, str) or not query:
            raise ValueError("query must be a non-empty string")

        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError("path must be a non-empty string")

        if not isinstance(case_sensitive, bool):
            raise ValueError("case_sensitive must be a boolean")

        root = workspace_root.resolve()
        target = (root / path_value).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise ValueError("Path is outside the workspace") from error

        results: list[str] = []

        command = [
            rg_path,
            "--json",
            "--fixed-strings",
            "--line-number",
            "--color=never",
            f"--max-filesize={MAX_SEARCH_FILE_BYTES}",
        ]

        if not case_sensitive:
            command.append("--ignore-case")

        relative_target = target.relative_to(root)
        search_path = str(relative_target) if relative_target.parts else "."
        command.extend(["--", query, search_path])

        if not target.exists():
            raise ValueError(f"Path not found: {target}")

        if not target.is_file() and not target.is_dir():
            raise ValueError(f"Path is not a file or directory: {target}")

        if target.is_file() and target.stat().st_size > MAX_SEARCH_FILE_BYTES:
            return "[No matches found]"

        stopped_early = False

        with tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                command,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                bufsize=0,
            )

            if process.stdout is None:
                process.kill()
                process.wait()
                raise RuntimeError("Failed to capture rg stdout")

            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + SEARCH_TIMEOUT_SECONDS
            pending = b""

            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"rg search timed out after {SEARCH_TIMEOUT_SECONDS} seconds"
                        )

                    events = selector.select(timeout=remaining)
                    if not events:
                        raise TimeoutError(
                            f"rg search timed out after {SEARCH_TIMEOUT_SECONDS} seconds"
                        )

                    chunk = os.read(process.stdout.fileno(), 65_536)
                    if chunk:
                        pending += chunk
                        json_lines = pending.split(b"\n")
                        pending = json_lines.pop()
                    else:
                        json_lines = [pending] if pending else []
                        pending = b""

                    for json_line in json_lines:
                        if not json_line:
                            continue

                        obj = json.loads(json_line)
                        if obj["type"] != "match":
                            continue

                        data = obj["data"]
                        path_text = data["path"].get("text")
                        line_text = data["lines"].get("text")
                        if not isinstance(path_text, str) or not isinstance(
                            line_text, str
                        ):
                            continue

                        text = line_text.rstrip("\r\n")
                        if len(text) > MAX_MATCH_LINE_CHARS:
                            text = text[:MAX_MATCH_LINE_CHARS] + "..."

                        results.append(
                            f'{path_text}:{data["line_number"]}: {text}'
                        )

                        if len(results) > MAX_SEARCH_RESULTS:
                            stopped_early = True
                            break

                    if stopped_early or not chunk:
                        break
            except BaseException:
                if process.poll() is None:
                    process.kill()
                process.wait()
                raise
            finally:
                selector.close()
                process.stdout.close()

            if stopped_early and process.poll() is None:
                process.terminate()

            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

            if not stopped_early and process.returncode not in (0, 1):
                stderr_file.seek(0)
                stderr = stderr_file.read(4_000).decode(
                    "utf-8", errors="replace"
                )
                raise RuntimeError(f"rg failed: {stderr.strip()}")

        if not results:
            return "[No matches found]"

        truncated = len(results) > MAX_SEARCH_RESULTS
        results = results[:MAX_SEARCH_RESULTS]

        if truncated:
            results.append("[Output truncated]")

        return "\n".join(results)

    return ToolDefinition(
        name="search_text",
        description="Search text",
        parameters=SEARCH_TEXT_PARAMETERS,
        handler=search_text_handle,
    )
