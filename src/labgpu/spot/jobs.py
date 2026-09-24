"""SQLite-backed spot job queue and run history (SPEC 2.7)."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .jobspec import JobSpec
from .model import QueuedJob, RunningSpot


class JobState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PREEMPTING = "PREEMPTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


ACTIVE_STATES = (JobState.RUNNING, JobState.PREEMPTING)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    uid INTEGER NOT NULL,
    gid INTEGER NOT NULL,
    spec TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    preemptions INTEGER NOT NULL DEFAULT 0,
    gpu_uuid TEXT,
    container TEXT,
    submitted_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    exit_code INTEGER,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    attempt INTEGER NOT NULL,
    gpu_uuid TEXT NOT NULL,
    container TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    outcome TEXT
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class JobRecord:
    id: int
    name: str
    uid: int
    gid: int
    spec: JobSpec
    priority: int
    state: JobState
    attempts: int
    preemptions: int
    gpu_uuid: str | None
    container: str | None
    submitted_at: float
    started_at: float | None
    finished_at: float | None
    exit_code: int | None
    reason: str | None


class JobStore:
    def __init__(self, path: Path | str, *, clock=time.time) -> None:
        self._conn = sqlite3.connect(str(path), isolation_level=None, timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._clock = clock

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    # ---- submission & queries ----

    def submit(self, spec: JobSpec, uid: int, gid: int) -> int:
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO jobs (name, uid, gid, spec, priority, state, submitted_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (spec.name, uid, gid, spec.to_json(), spec.priority, JobState.QUEUED, self._clock()),
            )
            return int(cur.lastrowid or 0)

    def get(self, job_id: int) -> JobRecord | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _record(row) if row else None

    def list(self, *, include_finished: bool = False) -> list[JobRecord]:
        if include_finished:
            rows = self._conn.execute("SELECT * FROM jobs ORDER BY id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE state IN (?, ?, ?) ORDER BY id",
                (JobState.QUEUED, JobState.RUNNING, JobState.PREEMPTING),
            ).fetchall()
        return [_record(r) for r in rows]

    def queued(self, default_ram: int) -> list[QueuedJob]:
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE state = ? ORDER BY id", (JobState.QUEUED,)
        ).fetchall()
        result = []
        for r in rows:
            spec = JobSpec.from_json(r["spec"])
            result.append(
                QueuedJob(
                    r["id"], r["priority"], r["submitted_at"], spec.gpu_mem,
                    spec.ram or default_ram, spec.gpu_models,
                )
            )
        return result

    def active(self) -> list[RunningSpot]:
        """Jobs whose container may still hold a GPU, including cancelled ones still stopping."""
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE state IN (?, ?)"
            " OR (state = ? AND container IS NOT NULL AND finished_at IS NULL) ORDER BY id",
            (*ACTIVE_STATES, JobState.CANCELLED),
        ).fetchall()
        return [
            RunningSpot(r["id"], r["gpu_uuid"], r["container"], r["state"] != JobState.RUNNING)
            for r in rows
        ]

    # ---- transitions ----

    def mark_running(self, job_id: int, gpu_uuid: str, container: str) -> int:
        """QUEUED → RUNNING. Returns the attempt number."""
        now = self._clock()
        with self._tx() as c:
            row = c.execute("SELECT state, attempts FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None or row["state"] != JobState.QUEUED:
                raise ValueError(f"job {job_id} is not queued")
            attempt = row["attempts"] + 1
            c.execute(
                "UPDATE jobs SET state = ?, attempts = ?, gpu_uuid = ?, container = ?,"
                " started_at = ?, reason = NULL WHERE id = ?",
                (JobState.RUNNING, attempt, gpu_uuid, container, now, job_id),
            )
            c.execute(
                "INSERT INTO runs (job_id, attempt, gpu_uuid, container, started_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (job_id, attempt, gpu_uuid, container, now),
            )
            return attempt

    def next_attempt(self, job_id: int) -> int:
        row = self._conn.execute("SELECT attempts FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return int(row["attempts"]) + 1

    def mark_preempting(self, job_id: int, reason: str) -> None:
        with self._tx() as c:
            c.execute(
                "UPDATE jobs SET state = ?, reason = ? WHERE id = ? AND state = ?",
                (JobState.PREEMPTING, reason, job_id, JobState.RUNNING),
            )

    def fail(self, job_id: int, reason: str) -> None:
        """Mark a job failed without it having run (e.g. `docker run` refused it)."""
        with self._tx() as c:
            c.execute(
                "UPDATE jobs SET state = ?, reason = ?, finished_at = ? WHERE id = ?",
                (JobState.FAILED, reason, self._clock(), job_id),
            )

    def finish(self, job_id: int, exit_code: int | None, *, max_attempts: int, vanished: bool = False) -> JobState:
        """
        Record the end of the current run. A run that ended because we reclaimed it (state
        PREEMPTING) or whose container disappeared goes back to the queue.
        """
        now = self._clock()
        with self._tx() as c:
            row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise ValueError(f"no job {job_id}")
            state = JobState(row["state"])
            preempted = state is JobState.PREEMPTING or vanished
            if state is JobState.CANCELLED:
                new_state, outcome = JobState.CANCELLED, "cancelled"
            elif preempted:
                if row["attempts"] >= max_attempts:
                    new_state, outcome = JobState.FAILED, "preempted"
                else:
                    new_state, outcome = JobState.QUEUED, "preempted"
            elif exit_code == 0:
                new_state, outcome = JobState.SUCCEEDED, "succeeded"
            else:
                new_state, outcome = JobState.FAILED, "failed"
            reason = row["reason"]
            if new_state is JobState.FAILED and preempted:
                reason = f"gave up after {row['attempts']} attempts (last: {reason})"
            elif outcome == "failed":
                reason = f"exited with code {exit_code}"
            c.execute(
                "UPDATE jobs SET state = ?, exit_code = ?, preemptions = preemptions + ?,"
                " finished_at = ?, reason = ?,"
                " gpu_uuid = CASE WHEN ? THEN NULL ELSE gpu_uuid END,"
                " container = CASE WHEN ? THEN NULL ELSE container END"
                " WHERE id = ?",
                (
                    new_state,
                    exit_code,
                    1 if preempted else 0,
                    None if new_state is JobState.QUEUED else now,
                    reason,
                    new_state is JobState.QUEUED,
                    new_state is JobState.QUEUED,
                    job_id,
                ),
            )
            c.execute(
                "UPDATE runs SET finished_at = ?, outcome = ?"
                " WHERE job_id = ? AND attempt = ? AND finished_at IS NULL",
                (now, outcome, job_id, row["attempts"]),
            )
            return new_state

    def cancel(self, job_id: int) -> str | None:
        """Cancel a job. Returns the container to stop if it was running."""
        with self._tx() as c:
            row = c.execute("SELECT state, container FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise ValueError(f"no job {job_id}")
            state = JobState(row["state"])
            if state not in (JobState.QUEUED, *ACTIVE_STATES):
                raise ValueError(f"job {job_id} is already {state}")
            c.execute(
                "UPDATE jobs SET state = ?, reason = 'cancelled by user',"
                " finished_at = CASE WHEN ? THEN ? ELSE finished_at END WHERE id = ?",
                (JobState.CANCELLED, state is JobState.QUEUED, self._clock(), job_id),
            )
            return row["container"] if state in ACTIVE_STATES else None

    # ---- settings (pause/resume) ----

    def paused(self) -> tuple[bool, frozenset[str]]:
        rows = dict(self._conn.execute("SELECT key, value FROM settings").fetchall())
        node = rows.get("paused_node") == "1"
        gpus = frozenset(json.loads(rows.get("paused_gpus", "[]")))
        return node, gpus

    def set_paused(self, paused: bool, gpu: str | None = None) -> None:
        with self._tx() as c:
            if gpu is None:
                c.execute(
                    "INSERT OR REPLACE INTO settings (key, value) VALUES ('paused_node', ?)",
                    ("1" if paused else "0",),
                )
                return
            row = c.execute("SELECT value FROM settings WHERE key = 'paused_gpus'").fetchone()
            gpus = set(json.loads(row["value"])) if row else set()
            (gpus.add if paused else gpus.discard)(gpu)
            c.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES ('paused_gpus', ?)",
                (json.dumps(sorted(gpus)),),
            )


def _record(r: sqlite3.Row) -> JobRecord:
    return JobRecord(
        id=r["id"],
        name=r["name"],
        uid=r["uid"],
        gid=r["gid"],
        spec=JobSpec.from_json(r["spec"]),
        priority=r["priority"],
        state=JobState(r["state"]),
        attempts=r["attempts"],
        preemptions=r["preemptions"],
        gpu_uuid=r["gpu_uuid"],
        container=r["container"],
        submitted_at=r["submitted_at"],
        started_at=r["started_at"],
        finished_at=r["finished_at"],
        exit_code=r["exit_code"],
        reason=r["reason"],
    )
