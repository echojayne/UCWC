import os
import selectors
import signal
import subprocess
import time
from pathlib import Path

from .tools import ToolDefinition

DEFAULT_COMMAND_TIMEOUT_SECONDS = 30
MAX_COMMAND_TIMEOUT_SECONDS = 120
MAX_COMMAND_ARGUMENTS = 100
MAX_COMMAND_ARGUMENT_CHARS = 20_000
MAX_COMMAND_OUTPUT_BYTES = 50_000

EXEC_COMMAND_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "Command and arguments as an array, without a shell",
        },
        "cwd": {
            "type": "string",
            "description": "Working directory relative to the workspace root",
        },
        "timeout_seconds": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_COMMAND_TIMEOUT_SECONDS,
            "description": "Maximum command runtime",
        },
    },
    "required": ["command"],
    "additionalProperties": False,
}


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=0.5)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _append_output(
    destination: bytearray,
    chunk: bytes,
) -> bool:
    remaining = MAX_COMMAND_OUTPUT_BYTES - len(destination)
    if remaining <= 0:
        return True
    destination.extend(chunk[:remaining])
    return len(chunk) > remaining


def create_exec_command_tool(workspace_root: Path) -> ToolDefinition:
    if not workspace_root.is_dir():
        raise ValueError(f"Workspace root {workspace_root} is not a directory")

    root = workspace_root.resolve()

    def exec_command_handler(arguments: dict[str, object]) -> str:
        command_value = arguments.get("command")
        cwd_value = arguments.get("cwd", ".")
        timeout_value = arguments.get(
            "timeout_seconds",
            DEFAULT_COMMAND_TIMEOUT_SECONDS,
        )

        if not isinstance(command_value, list) or not command_value:
            raise ValueError("command must be a non-empty array of strings")
        if len(command_value) > MAX_COMMAND_ARGUMENTS:
            raise ValueError(
                f"command cannot contain more than {MAX_COMMAND_ARGUMENTS} arguments"
            )
        if not all(isinstance(item, str) for item in command_value):
            raise ValueError("command must contain only strings")

        command = list(command_value)
        if not command[0]:
            raise ValueError("command executable must not be empty")
        if any("\x00" in item for item in command):
            raise ValueError("command arguments must not contain null bytes")
        if sum(len(item) for item in command) > MAX_COMMAND_ARGUMENT_CHARS:
            raise ValueError("command arguments are too large")

        if not isinstance(cwd_value, str) or not cwd_value.strip():
            raise ValueError("cwd must be a non-empty string")
        if not isinstance(timeout_value, int) or isinstance(timeout_value, bool):
            raise ValueError("timeout_seconds must be an integer")
        if not 1 <= timeout_value <= MAX_COMMAND_TIMEOUT_SECONDS:
            raise ValueError(
                f"timeout_seconds must be between 1 and "
                f"{MAX_COMMAND_TIMEOUT_SECONDS}"
            )

        cwd_path = Path(cwd_value)
        if cwd_path.is_absolute():
            raise ValueError("cwd must be relative to the workspace")
        cwd = (root / cwd_path).resolve()
        try:
            cwd.relative_to(root)
        except ValueError as error:
            raise ValueError("cwd is outside the workspace") from error
        if not cwd.is_dir():
            raise ValueError(f"Working directory not found: {cwd}")

        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            bufsize=0,
        )

        if process.stdout is None or process.stderr is None:
            _stop_process_group(process)
            raise RuntimeError("Failed to capture command output")

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")

        stdout = bytearray()
        stderr = bytearray()
        stdout_truncated = False
        stderr_truncated = False
        timed_out = False
        deadline = time.monotonic() + timeout_value

        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    _stop_process_group(process)
                    break

                events = selector.select(timeout=remaining)
                if not events:
                    timed_out = True
                    _stop_process_group(process)
                    break

                for key, _ in events:
                    chunk = os.read(key.fd, 65_536)
                    if not chunk:
                        stream = (
                            process.stdout
                            if key.data == "stdout"
                            else process.stderr
                        )
                        selector.unregister(stream)
                        stream.close()
                        continue

                    if key.data == "stdout":
                        stdout_truncated |= _append_output(stdout, chunk)
                    else:
                        stderr_truncated |= _append_output(stderr, chunk)
        except BaseException:
            _stop_process_group(process)
            raise
        finally:
            for key in list(selector.get_map().values()):
                stream = (
                    process.stdout
                    if key.data == "stdout"
                    else process.stderr
                )
                selector.unregister(stream)
                stream.close()
            selector.close()

        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                _stop_process_group(process)

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        if stdout_truncated:
            stdout_text += "\n[stdout truncated]"
        if stderr_truncated:
            stderr_text += "\n[stderr truncated]"

        status = "yes" if timed_out else "no"
        return (
            f"Exit code: {process.returncode}\n"
            f"Timed out: {status}\n"
            f"stdout:\n{stdout_text or '[empty]'}\n"
            f"stderr:\n{stderr_text or '[empty]'}"
        )

    return ToolDefinition(
        name="exec_command",
        description="Run a non-interactive command inside the workspace without a shell",
        parameters=EXEC_COMMAND_PARAMETERS,
        handler=exec_command_handler,
    )
