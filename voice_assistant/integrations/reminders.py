from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Reminder:
    id: int
    text: str
    due_at_utc: datetime
    timezone_name: str
    status: str
    source: str | None = None
    external_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        local_due = self.due_at_utc.astimezone(ZoneInfo(self.timezone_name))
        return {
            "id": self.id,
            "text": self.text,
            "due_at": local_due.isoformat(),
            "timezone": self.timezone_name,
            "status": self.status,
        }


class ReminderStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create(self, text: str, due_at: datetime, timezone_name: str) -> Reminder:
        if not text.strip():
            raise ValueError("Reminder text is required.")
        if due_at.tzinfo is None:
            raise ValueError("Reminder due time must be timezone-aware.")
        due_utc = due_at.astimezone(timezone.utc)
        created_utc = datetime.now(timezone.utc)
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                INSERT INTO reminders(text, due_at_utc, timezone_name, status, created_at_utc)
                VALUES (?, ?, ?, 'pending', ?)
                """,
                (text.strip(), due_utc.isoformat(), timezone_name, created_utc.isoformat()),
            )
            reminder_id = int(cursor.lastrowid)
        reminder = Reminder(reminder_id, text.strip(), due_utc, timezone_name, "pending")
        logger.info("Reminder created: id=%d due_at=%s", reminder.id, due_at.isoformat())
        return reminder

    def upsert_external(
        self,
        source: str,
        external_id: str,
        text: str,
        due_at: datetime,
        timezone_name: str,
    ) -> Reminder:
        source = source.strip()
        external_id = external_id.strip()
        text = text.strip()
        if not source or not external_id:
            raise ValueError("External reminder source and ID are required.")
        if not text:
            raise ValueError("Reminder text is required.")
        if due_at.tzinfo is None:
            raise ValueError("Reminder due time must be timezone-aware.")

        due_utc = due_at.astimezone(timezone.utc)
        created_utc = datetime.now(timezone.utc)
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT id, text, due_at_utc, timezone_name, status,
                       source, external_id, cancel_reason
                FROM reminders
                WHERE source = ? AND external_id = ?
                """,
                (source, external_id),
            ).fetchone()
            if row is None:
                cursor = connection.execute(
                    """
                    INSERT INTO reminders(
                        text, due_at_utc, timezone_name, status, created_at_utc,
                        source, external_id
                    ) VALUES (?, ?, ?, 'pending', ?, ?, ?)
                    """,
                    (
                        text,
                        due_utc.isoformat(),
                        timezone_name,
                        created_utc.isoformat(),
                        source,
                        external_id,
                    ),
                )
                reminder_id = int(cursor.lastrowid)
                status = "pending"
                logger.info(
                    "External reminder created: id=%d source=%s due_at=%s",
                    reminder_id,
                    source,
                    due_at.isoformat(),
                )
            else:
                reminder_id = int(row[0])
                old_due = _parse_utc(row[2])
                status = str(row[4])
                cancel_reason = str(row[7] or "")
                due_changed = old_due != due_utc
                if (status == "fired" and due_changed) or (
                    status == "cancelled" and cancel_reason == "source_missing"
                ):
                    status = "pending"
                connection.execute(
                    """
                    UPDATE reminders
                    SET text = ?, due_at_utc = ?, timezone_name = ?, status = ?,
                        fired_at_utc = CASE WHEN ? = 'pending' THEN NULL ELSE fired_at_utc END,
                        cancel_reason = CASE WHEN ? = 'pending' THEN NULL ELSE cancel_reason END
                    WHERE id = ?
                    """,
                    (
                        text,
                        due_utc.isoformat(),
                        timezone_name,
                        status,
                        status,
                        status,
                        reminder_id,
                    ),
                )
                logger.debug(
                    "External reminder synchronized: id=%d source=%s status=%s due_changed=%s",
                    reminder_id,
                    source,
                    status,
                    due_changed,
                )
            connection.commit()
        return Reminder(
            reminder_id,
            text,
            due_utc,
            timezone_name,
            status,
            source,
            external_id,
        )

    def cancel_missing_external(
        self,
        source: str,
        active_external_ids: set[str],
    ) -> int:
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT id, external_id FROM reminders
                WHERE source = ? AND status = 'pending'
                """,
                (source,),
            ).fetchall()
            missing_ids = [
                int(row[0])
                for row in rows
                if str(row[1]) not in active_external_ids
            ]
            if missing_ids:
                connection.executemany(
                    """
                    UPDATE reminders
                    SET status = 'cancelled', cancel_reason = 'source_missing'
                    WHERE id = ? AND status = 'pending'
                    """,
                    ((reminder_id,) for reminder_id in missing_ids),
                )
            connection.commit()
        if missing_ids:
            logger.info(
                "Cancelled %d stale external reminder(s) for source=%s.",
                len(missing_ids),
                source,
            )
        return len(missing_ids)

    def list_pending(self, limit: int = 20) -> list[Reminder]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """
                SELECT id, text, due_at_utc, timezone_name, status, source, external_id
                FROM reminders WHERE status = 'pending'
                ORDER BY due_at_utc LIMIT ?
                """,
                (max(1, min(100, limit)),),
            ).fetchall()
        return [_from_row(row) for row in rows]

    def claim_due(self, now: datetime, limit: int = 10) -> list[Reminder]:
        if now.tzinfo is None:
            raise ValueError("Reminder polling time must be timezone-aware.")
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT id, text, due_at_utc, timezone_name, status, source, external_id
                FROM reminders
                WHERE status = 'pending' AND due_at_utc <= ?
                ORDER BY due_at_utc LIMIT ?
                """,
                (now.astimezone(timezone.utc).isoformat(), max(1, min(100, limit))),
            ).fetchall()
            ids = [int(row[0]) for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                connection.execute(
                    # Only the number of integer IDs controls this placeholder string.
                    f"UPDATE reminders SET status = 'firing' WHERE id IN ({placeholders})",  # nosec
                    ids,
                )
            connection.commit()
        claimed = [
            Reminder(item.id, item.text, item.due_at_utc, item.timezone_name, "firing")
            for item in (_from_row(row) for row in rows)
        ]
        for reminder in claimed:
            logger.info("Reminder due and claimed: id=%d", reminder.id)
        return claimed

    def mark_fired(self, reminder_id: int) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE reminders
                SET status = 'fired', fired_at_utc = ?, cancel_reason = NULL
                WHERE id = ?
                """,
                (datetime.now(timezone.utc).isoformat(), reminder_id),
            )
        logger.info("Reminder fired: id=%d", reminder_id)

    def release(self, reminder_id: int) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE reminders SET status = 'pending' WHERE id = ? AND status = 'firing'",
                (reminder_id,),
            )
        logger.warning("Reminder playback did not complete; released id=%d for retry.", reminder_id)

    def cancel(self, reminder_id: int) -> bool:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE reminders SET status = 'cancelled', cancel_reason = 'user'
                WHERE id = ? AND status IN ('pending', 'firing')
                """,
                (reminder_id,),
            )
        cancelled = cursor.rowcount > 0
        logger.info("Reminder cancellation: id=%d cancelled=%s", reminder_id, cancelled)
        return cancelled

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    due_at_utc TEXT NOT NULL,
                    timezone_name TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','firing','fired','cancelled')),
                    created_at_utc TEXT NOT NULL,
                    fired_at_utc TEXT,
                    source TEXT,
                    external_id TEXT,
                    cancel_reason TEXT
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(reminders)").fetchall()
            }
            for name, declaration in (
                ("source", "TEXT"),
                ("external_id", "TEXT"),
                ("cancel_reason", "TEXT"),
            ):
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE reminders ADD COLUMN {name} {declaration}"
                    )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS reminders_external_identity
                ON reminders(source, external_id)
                WHERE source IS NOT NULL AND external_id IS NOT NULL
                """
            )
            recovered = connection.execute(
                "UPDATE reminders SET status = 'pending' WHERE status = 'firing'"
            ).rowcount
        if recovered:
            logger.warning("Recovered %d interrupted reminder(s) after restart.", recovered)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path, timeout=10.0)


class ReminderScheduler:
    def __init__(
        self,
        store: ReminderStore,
        poll_interval_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        now_provider: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.store = store
        self._poll_interval = max(0.25, poll_interval_seconds)
        self._clock = clock
        self._now_provider = now_provider
        self._next_poll = 0.0

    def poll(self, force: bool = False) -> list[Reminder]:
        now_tick = self._clock()
        if not force and now_tick < self._next_poll:
            return []
        self._next_poll = now_tick + self._poll_interval
        return self.store.claim_due(self._now_provider())


def _from_row(row: tuple[object, ...]) -> Reminder:
    due = _parse_utc(row[2])
    return Reminder(
        id=int(row[0]),
        text=str(row[1]),
        due_at_utc=due.astimezone(timezone.utc),
        timezone_name=str(row[3]),
        status=str(row[4]),
        source=None if len(row) < 6 or row[5] is None else str(row[5]),
        external_id=None if len(row) < 7 or row[6] is None else str(row[6]),
    )


def _parse_utc(value: object) -> datetime:
    due = datetime.fromisoformat(str(value))
    if due.tzinfo is None:
        due = due.replace(tzinfo=timezone.utc)
    return due.astimezone(timezone.utc)
