"""Portable elapsed-time reporting for the reproducibility runner."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import tempfile
import time
from collections.abc import Mapping, MutableMapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

TIMING_FILE_LOCK_TIMEOUT_SECONDS = 15 * 60
_TIMING_FILE_LOCK_POLL_SECONDS = 0.25


def timings_filename(num_agents: int | None = None) -> str:
    _ = num_agents
    return "timings.json"


def _timings_root(path: str | Path) -> Path:
    root = Path(path)
    if root.suffix:
        return root.parent
    return root


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, Path):
        return str(value)

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_ready(item())
        except (TypeError, ValueError):
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_ready(tolist())

    if isinstance(value, dict):
        return {str(key): _json_ready(nested) for key, nested in value.items()}

    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]

    if isinstance(value, set):
        return [_json_ready(item) for item in sorted(value, key=repr)]

    raise TypeError(
        f"Unsupported JSON value for timings export: {type(value).__name__}"
    )


def load_experiment_timings(
    save_dir: str | Path,
    *,
    num_agents: int | None = None,
) -> dict[str, dict[str, Any]]:
    root = _timings_root(save_dir)
    path = root / timings_filename(num_agents)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _load_timing_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


@contextmanager
def _timing_file_lock(target: Path, *, timeout_seconds: float | None = None):
    lock_path = target.parent / f"{target.name}.lock"
    timeout_seconds = (
        TIMING_FILE_LOCK_TIMEOUT_SECONDS
        if timeout_seconds is None
        else float(timeout_seconds)
    )
    if timeout_seconds < 0:
        raise ValueError("timing lock timeout must be non-negative")

    with lock_path.open("a", encoding="utf-8") as lock_handle:
        started_at = time.monotonic()
        while True:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                elapsed_seconds = time.monotonic() - started_at
                if elapsed_seconds >= timeout_seconds:
                    raise TimeoutError(
                        f"Timed out after {timeout_seconds:.0f}s waiting for "
                        f"exclusive lock on {lock_path} while saving {target}. "
                        "The lock is released automatically when the writer "
                        "process exits, so this usually means a live writer is stuck."
                    ) from exc
                time.sleep(
                    min(
                        _TIMING_FILE_LOCK_POLL_SECONDS,
                        timeout_seconds - elapsed_seconds,
                    )
                )
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


def _atomic_write_timing_payload(target: Path, payload: Mapping[str, Any]) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        temp_path = None
        _fsync_directory(target.parent)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def record_timing(
    timings: MutableMapping[str, dict[str, Any]],
    stage: str,
    seconds: float,
    run_idx: int | None = None,
    label: str | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> MutableMapping[str, dict[str, Any]]:
    reset_stage = run_idx is None or int(run_idx) == 0
    if reset_stage:
        timings[stage] = {"total_seconds": 0.0, "runs": []}

    entry = timings.setdefault(stage, {"total_seconds": 0.0, "runs": []})
    run_entry: dict[str, Any] = {"seconds": float(seconds)}
    if run_idx is not None:
        run_entry["run_idx"] = int(run_idx)
    if label is not None:
        run_entry["label"] = label
    if metadata is not None:
        run_entry.update(
            {str(key): _json_ready(value) for key, value in metadata.items()}
        )

    if run_idx is not None:
        entry["runs"] = [
            run
            for run in entry["runs"]
            if not (
                isinstance(run, Mapping)
                and run.get("run_idx") is not None
                and int(run["run_idx"]) == int(run_idx)
            )
        ]
    entry["runs"].append(run_entry)
    entry["runs"].sort(
        key=lambda run: (
            int(run.get("run_idx", 10**9)) if isinstance(run, Mapping) else 10**9
        )
    )
    entry["total_seconds"] = float(
        sum(
            float(run.get("seconds", 0.0))
            for run in entry["runs"]
            if isinstance(run, Mapping)
        )
    )
    return timings


def save_experiment_timings(
    timings, save_dir, *, num_agents=None, lock_timeout_seconds=None
):
    target = _timings_root(save_dir) / timings_filename(num_agents)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _timing_file_lock(target, timeout_seconds=lock_timeout_seconds):
        payload = {**_load_timing_payload(target), **_json_ready(dict(timings))}
        _atomic_write_timing_payload(target, payload)
    return timings


def summarize_timings(timings):
    if not timings:
        return "No timings recorded."
    return "\n".join(
        f"{stage}: {entry.get('total_seconds', 0.0):.2f}s total ({len(entry.get('runs', []))} entries)"
        for stage, entry in sorted(timings.items())
    )
