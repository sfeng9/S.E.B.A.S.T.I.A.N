from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import unicodedata
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
TASK_STATUSES = frozenset({"pending", "completed"})
STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "did",
        "do",
        "for",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "note",
        "of",
        "on",
        "the",
        "to",
        "what",
        "where",
        "you",
    }
)


class PersonalDataError(RuntimeError):
    pass


class DuplicateRecordError(PersonalDataError):
    pass


@dataclass(frozen=True)
class Note:
    id: int
    title: str
    content: str
    tags: tuple[str, ...]
    created_at_utc: str
    updated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "tags": list(self.tags),
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
        }


@dataclass(frozen=True)
class Task:
    id: int
    title: str
    status: str
    priority: str | None
    due_at_utc: str | None
    timezone_name: str | None
    reminder_id: int | None
    project: str | None
    category: str | None
    recurrence: str | None
    created_at_utc: str
    updated_at_utc: str
    completed_at_utc: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "priority": self.priority,
            "due_at_utc": self.due_at_utc,
            "timezone_name": self.timezone_name,
            "reminder_id": self.reminder_id,
            "project": self.project,
            "category": self.category,
            "recurrence": self.recurrence,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "completed_at_utc": self.completed_at_utc,
        }


@dataclass(frozen=True)
class PersonalList:
    id: int
    name: str
    aliases: tuple[str, ...]
    created_at_utc: str
    updated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "aliases": list(self.aliases),
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
        }


@dataclass(frozen=True)
class ListItem:
    id: int
    list_id: int
    text: str
    completed: bool
    created_at_utc: str
    updated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "list_id": self.list_id,
            "text": self.text,
            "completed": self.completed,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
        }


class PersonalDatabase:
    """Lazy, versioned SQLite connection factory for private personal data."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._initialized = False
        self._initialization_lock = threading.Lock()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            self._initialize(connection)
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self, connection: sqlite3.Connection) -> None:
        if self._initialized:
            return
        with self._initialization_lock:
            if self._initialized:
                return
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise PersonalDataError(
                    f"Database schema {version} is newer than supported version {SCHEMA_VERSION}."
                )
            if version < 1:
                _migration_1(connection)
                connection.execute("PRAGMA user_version = 1")
                connection.commit()
                logger.info("Personal data migration applied: version=1")
            self._initialized = True


class NotesRepository:
    def __init__(
        self,
        database: PersonalDatabase,
        now_provider: Callable[[], datetime],
    ) -> None:
        self._database = database
        self._now_provider = now_provider

    def create(self, title: str, content: str, tags: Sequence[str] = ()) -> Note:
        clean_title = _required_text(title, "A note title is required.", 160)
        clean_content = _required_text(content, "Note content is required.", 20000)
        clean_tags = _clean_tags(tags)
        now = _utc_iso(self._now_provider())
        with self._database.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO notes(title, content, tags_json, created_at_utc, updated_at_utc)
                VALUES (?, ?, ?, ?, ?)
                """,
                (clean_title, clean_content, json.dumps(clean_tags), now, now),
            )
            note_id = int(cursor.lastrowid)
            row = connection.execute(
                "SELECT * FROM notes WHERE id = ?", (note_id,)
            ).fetchone()
        note = _note_from_row(row)
        logger.info("Note created: id=%d title=%r tags=%d", note.id, note.title, len(note.tags))
        return note

    def get(self, note_id: int) -> Note | None:
        with self._database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM notes WHERE id = ?", (int(note_id),)
            ).fetchone()
        return _note_from_row(row) if row is not None else None

    def list(self, limit: int = 10) -> list[Note]:
        with self._database.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM notes ORDER BY updated_at_utc DESC, id DESC LIMIT ?",
                (_limit(limit, 100),),
            ).fetchall()
        return [_note_from_row(row) for row in rows]

    def search(self, query: str, limit: int = 8) -> list[Note]:
        terms = _search_terms(query)
        if not terms:
            return self.list(limit)
        clauses = " OR ".join("search_text LIKE ?" for _ in terms)
        params = [f"%{term}%" for term in terms]
        with self._database.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM notes WHERE {clauses} ORDER BY updated_at_utc DESC LIMIT 100",  # nosec B608
                params,
            ).fetchall()
        query_text = _normalize(query)
        ranked = sorted(
            (_note_from_row(row) for row in rows),
            key=lambda note: _search_score(
                query_text,
                terms,
                _normalize(f"{note.title} {note.content} {' '.join(note.tags)}"),
            ),
            reverse=True,
        )[: _limit(limit, 20)]
        logger.info("Note search completed: terms=%d matches=%d", len(terms), len(ranked))
        return ranked

    def update(
        self,
        note_id: int,
        *,
        title: str | None = None,
        content: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> Note | None:
        current = self.get(note_id)
        if current is None:
            return None
        new_title = current.title if title is None else _required_text(title, "A note title is required.", 160)
        new_content = current.content if content is None else _required_text(content, "Note content is required.", 20000)
        new_tags = current.tags if tags is None else _clean_tags(tags)
        now = _utc_iso(self._now_provider())
        with self._database.connection() as connection:
            connection.execute(
                """
                UPDATE notes SET title = ?, content = ?, tags_json = ?,
                    search_text = ?, updated_at_utc = ? WHERE id = ?
                """,
                (
                    new_title,
                    new_content,
                    json.dumps(new_tags),
                    _normalize(f"{new_title} {new_content} {' '.join(new_tags)}"),
                    now,
                    int(note_id),
                ),
            )
        logger.info("Note updated: id=%d title=%r", note_id, new_title)
        return self.get(note_id)

    def delete(self, note_id: int) -> Note | None:
        current = self.get(note_id)
        if current is None:
            return None
        with self._database.connection() as connection:
            connection.execute("DELETE FROM notes WHERE id = ?", (int(note_id),))
        logger.info("Note deleted: id=%d title=%r", current.id, current.title)
        return current


class TasksRepository:
    def __init__(
        self,
        database: PersonalDatabase,
        now_provider: Callable[[], datetime],
    ) -> None:
        self._database = database
        self._now_provider = now_provider

    def create(
        self,
        title: str,
        *,
        due_at_utc: str | None = None,
        timezone_name: str | None = None,
        priority: str | None = None,
        reminder_id: int | None = None,
        project: str | None = None,
        category: str | None = None,
        recurrence: str | None = None,
    ) -> Task:
        clean_title = _required_text(title, "A task title is required.", 500)
        now = _utc_iso(self._now_provider())
        values = (
            clean_title,
            _normalize(clean_title),
            _optional_text(priority, 40),
            due_at_utc,
            _optional_text(timezone_name, 100),
            reminder_id,
            _optional_text(project, 160),
            _optional_text(category, 160),
            _optional_text(recurrence, 160),
            now,
            now,
        )
        with self._database.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO tasks(
                    title, normalized_title, status, priority, due_at_utc,
                    timezone_name, reminder_id, project, category, recurrence,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            task_id = int(cursor.lastrowid)
        task = self.get(task_id)
        if task is None:
            raise PersonalDataError("The created task could not be loaded.")
        logger.info("Task created: id=%d due=%s priority=%s", task.id, bool(due_at_utc), task.priority)
        return task

    def get(self, task_id: int) -> Task | None:
        with self._database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (int(task_id),)
            ).fetchone()
        return _task_from_row(row) if row is not None else None

    def list(
        self,
        *,
        status: str | None = None,
        due_after_utc: str | None = None,
        due_before_utc: str | None = None,
        limit: int = 20,
    ) -> list[Task]:
        clauses: list[str] = []
        params: list[Any] = []
        if status and status != "all":
            _validate_status(status)
            clauses.append("status = ?")
            params.append(status)
        if due_after_utc:
            clauses.append("due_at_utc >= ?")
            params.append(due_after_utc)
        if due_before_utc:
            clauses.append("due_at_utc < ?")
            params.append(due_before_utc)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(_limit(limit, 100))
        with self._database.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM tasks {where}
                ORDER BY CASE WHEN due_at_utc IS NULL THEN 1 ELSE 0 END,
                    due_at_utc, created_at_utc, id LIMIT ?
                """,  # nosec B608
                params,
            ).fetchall()
        return [_task_from_row(row) for row in rows]

    def find(
        self,
        query: str,
        *,
        status: str | None = None,
        limit: int = 10,
    ) -> list[Task]:
        terms = _search_terms(query)
        clauses: list[str] = []
        params: list[Any] = []
        if terms:
            clauses.append("(" + " OR ".join("normalized_title LIKE ?" for _ in terms) + ")")
            params.extend(f"%{term}%" for term in terms)
        if status and status != "all":
            _validate_status(status)
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(_limit(limit, 20))
        with self._database.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM tasks {where} ORDER BY updated_at_utc DESC, id DESC LIMIT ?",  # nosec B608
                params,
            ).fetchall()
        return [_task_from_row(row) for row in rows]

    def set_status(self, task_id: int, status: str) -> Task | None:
        _validate_status(status)
        now = _utc_iso(self._now_provider())
        completed_at = now if status == "completed" else None
        with self._database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE tasks SET status = ?, completed_at_utc = ?, updated_at_utc = ?
                WHERE id = ?
                """,
                (status, completed_at, now, int(task_id)),
            )
        if cursor.rowcount == 0:
            return None
        logger.info("Task status changed: id=%d status=%s", task_id, status)
        return self.get(task_id)

    def update(
        self,
        task_id: int,
        *,
        title: str | None = None,
        due_at_utc: str | None | object = ...,
        timezone_name: str | None | object = ...,
        priority: str | None | object = ...,
        project: str | None | object = ...,
        category: str | None | object = ...,
    ) -> Task | None:
        current = self.get(task_id)
        if current is None:
            return None
        new_title = current.title if title is None else _required_text(title, "A task title is required.", 500)
        values = {
            "due_at_utc": current.due_at_utc if due_at_utc is ... else due_at_utc,
            "timezone_name": current.timezone_name if timezone_name is ... else timezone_name,
            "priority": current.priority if priority is ... else priority,
            "project": current.project if project is ... else project,
            "category": current.category if category is ... else category,
        }
        now = _utc_iso(self._now_provider())
        with self._database.connection() as connection:
            connection.execute(
                """
                UPDATE tasks SET title = ?, normalized_title = ?, due_at_utc = ?,
                    timezone_name = ?, priority = ?, project = ?, category = ?,
                    updated_at_utc = ? WHERE id = ?
                """,
                (
                    new_title,
                    _normalize(new_title),
                    values["due_at_utc"],
                    _optional_text(values["timezone_name"], 100),
                    _optional_text(values["priority"], 40),
                    _optional_text(values["project"], 160),
                    _optional_text(values["category"], 160),
                    now,
                    int(task_id),
                ),
            )
        logger.info("Task updated: id=%d", task_id)
        return self.get(task_id)

    def set_reminder(self, task_id: int, reminder_id: int) -> Task | None:
        with self._database.connection() as connection:
            connection.execute(
                "UPDATE tasks SET reminder_id = ?, updated_at_utc = ? WHERE id = ?",
                (int(reminder_id), _utc_iso(self._now_provider()), int(task_id)),
            )
        return self.get(task_id)

    def delete(self, task_id: int) -> Task | None:
        current = self.get(task_id)
        if current is None:
            return None
        with self._database.connection() as connection:
            connection.execute("DELETE FROM tasks WHERE id = ?", (int(task_id),))
        logger.info("Task deleted: id=%d", current.id)
        return current

    def delete_completed(self) -> int:
        with self._database.connection() as connection:
            cursor = connection.execute("DELETE FROM tasks WHERE status = 'completed'")
        count = max(0, cursor.rowcount)
        logger.info("Completed tasks deleted: count=%d", count)
        return count


class ListsRepository:
    def __init__(
        self,
        database: PersonalDatabase,
        now_provider: Callable[[], datetime],
    ) -> None:
        self._database = database
        self._now_provider = now_provider

    def create(self, name: str, aliases: Sequence[str] = ()) -> PersonalList:
        clean_name = _required_text(name, "A list name is required.", 160)
        clean_aliases = _clean_aliases(aliases, clean_name)
        if self.find_by_name(clean_name, exact=True):
            raise DuplicateRecordError(f"A list named {clean_name} already exists.")
        now = _utc_iso(self._now_provider())
        try:
            with self._database.connection() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO lists(name, normalized_name, aliases_json, created_at_utc, updated_at_utc)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (clean_name, _normalize(clean_name), json.dumps(clean_aliases), now, now),
                )
                list_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise DuplicateRecordError(f"A list named {clean_name} already exists.") from exc
        created = self.get(list_id)
        if created is None:
            raise PersonalDataError("The created list could not be loaded.")
        logger.info("List created: id=%d name=%r aliases=%d", created.id, created.name, len(created.aliases))
        return created

    def get(self, list_id: int) -> PersonalList | None:
        with self._database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM lists WHERE id = ?", (int(list_id),)
            ).fetchone()
        return _list_from_row(row) if row is not None else None

    def list(self, limit: int = 20) -> list[PersonalList]:
        with self._database.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM lists ORDER BY updated_at_utc DESC, id DESC LIMIT ?",
                (_limit(limit, 100),),
            ).fetchall()
        return [_list_from_row(row) for row in rows]

    def find_by_name(self, name: str, *, exact: bool = False) -> list[PersonalList]:
        target = _normalize(name)
        if not target:
            return []
        matches: list[PersonalList] = []
        for item in self.list(100):
            names = {_normalize(item.name), *(_normalize(alias) for alias in item.aliases)}
            if target in names or (not exact and any(target in candidate for candidate in names)):
                matches.append(item)
        return matches

    def rename(self, list_id: int, new_name: str) -> PersonalList | None:
        current = self.get(list_id)
        if current is None:
            return None
        clean_name = _required_text(new_name, "A list name is required.", 160)
        conflicts = [item for item in self.find_by_name(clean_name, exact=True) if item.id != current.id]
        if conflicts:
            raise DuplicateRecordError(f"A list named {clean_name} already exists.")
        aliases = _clean_aliases((*current.aliases, current.name), clean_name)
        with self._database.connection() as connection:
            connection.execute(
                """
                UPDATE lists SET name = ?, normalized_name = ?, aliases_json = ?,
                    updated_at_utc = ? WHERE id = ?
                """,
                (clean_name, _normalize(clean_name), json.dumps(aliases), _utc_iso(self._now_provider()), int(list_id)),
            )
        logger.info("List renamed: id=%d old_name=%r new_name=%r", list_id, current.name, clean_name)
        return self.get(list_id)

    def delete(self, list_id: int) -> tuple[PersonalList, list[ListItem]] | None:
        current = self.get(list_id)
        if current is None:
            return None
        items = self.get_items(list_id, include_completed=True, limit=500)
        with self._database.connection() as connection:
            connection.execute("DELETE FROM lists WHERE id = ?", (int(list_id),))
        logger.info("List deleted: id=%d item_count=%d", list_id, len(items))
        return current, items

    def get_items(
        self,
        list_id: int,
        *,
        include_completed: bool = True,
        limit: int = 100,
    ) -> list[ListItem]:
        completed_clause = "" if include_completed else "AND completed = 0"
        with self._database.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM list_items WHERE list_id = ? {completed_clause}
                ORDER BY completed, created_at_utc, id LIMIT ?
                """,  # nosec B608
                (int(list_id), _limit(limit, 500)),
            ).fetchall()
        return [_item_from_row(row) for row in rows]

    def add_items(self, list_id: int, items: Sequence[str]) -> dict[str, list[ListItem]]:
        if self.get(list_id) is None:
            raise PersonalDataError("That list does not exist.")
        cleaned = _deduplicate_input(items)
        if not cleaned:
            raise ValueError("At least one list item is required.")
        result: dict[str, list[ListItem]] = {"added": [], "reopened": [], "duplicates": []}
        now = _utc_iso(self._now_provider())
        with self._database.connection() as connection:
            for text in cleaned:
                normalized = _normalize(text)
                row = connection.execute(
                    "SELECT * FROM list_items WHERE list_id = ? AND normalized_text = ?",
                    (int(list_id), normalized),
                ).fetchone()
                if row is not None:
                    item = _item_from_row(row)
                    if item.completed:
                        connection.execute(
                            "UPDATE list_items SET completed = 0, updated_at_utc = ? WHERE id = ?",
                            (now, item.id),
                        )
                        refreshed = connection.execute(
                            "SELECT * FROM list_items WHERE id = ?", (item.id,)
                        ).fetchone()
                        result["reopened"].append(_item_from_row(refreshed))
                    else:
                        result["duplicates"].append(item)
                    continue
                cursor = connection.execute(
                    """
                    INSERT INTO list_items(
                        list_id, text, normalized_text, completed, created_at_utc, updated_at_utc
                    ) VALUES (?, ?, ?, 0, ?, ?)
                    """,
                    (int(list_id), text, normalized, now, now),
                )
                inserted = connection.execute(
                    "SELECT * FROM list_items WHERE id = ?", (int(cursor.lastrowid),)
                ).fetchone()
                result["added"].append(_item_from_row(inserted))
            connection.execute(
                "UPDATE lists SET updated_at_utc = ? WHERE id = ?", (now, int(list_id))
            )
        logger.info(
            "List items processed: list_id=%d added=%d reopened=%d duplicates=%d",
            list_id,
            len(result["added"]),
            len(result["reopened"]),
            len(result["duplicates"]),
        )
        return result

    def find_items(self, list_id: int, query: str, limit: int = 10) -> list[ListItem]:
        target = _normalize(query)
        if not target:
            return []
        with self._database.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM list_items
                WHERE list_id = ? AND normalized_text LIKE ?
                ORDER BY completed, created_at_utc, id LIMIT ?
                """,
                (int(list_id), f"%{target}%", _limit(limit, 20)),
            ).fetchall()
        return [_item_from_row(row) for row in rows]

    def set_item_completed(self, item_id: int, completed: bool) -> ListItem | None:
        now = _utc_iso(self._now_provider())
        with self._database.connection() as connection:
            cursor = connection.execute(
                "UPDATE list_items SET completed = ?, updated_at_utc = ? WHERE id = ?",
                (1 if completed else 0, now, int(item_id)),
            )
            row = connection.execute(
                "SELECT * FROM list_items WHERE id = ?", (int(item_id),)
            ).fetchone()
        if cursor.rowcount == 0 or row is None:
            return None
        logger.info("List item status changed: id=%d completed=%s", item_id, completed)
        return _item_from_row(row)

    def remove_item(self, item_id: int) -> ListItem | None:
        with self._database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM list_items WHERE id = ?", (int(item_id),)
            ).fetchone()
            if row is None:
                return None
            connection.execute("DELETE FROM list_items WHERE id = ?", (int(item_id),))
        item = _item_from_row(row)
        logger.info("List item removed: id=%d list_id=%d", item.id, item.list_id)
        return item

    def clear(self, list_id: int) -> list[ListItem]:
        items = self.get_items(list_id, include_completed=True, limit=500)
        with self._database.connection() as connection:
            connection.execute("DELETE FROM list_items WHERE list_id = ?", (int(list_id),))
            connection.execute(
                "UPDATE lists SET updated_at_utc = ? WHERE id = ?",
                (_utc_iso(self._now_provider()), int(list_id)),
            )
        logger.info("List cleared: id=%d item_count=%d", list_id, len(items))
        return items


class PersonalDataStore:
    def __init__(
        self,
        path: Path,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        clock = now_provider or (lambda: datetime.now(timezone.utc))
        self.database = PersonalDatabase(path)
        self.notes = NotesRepository(self.database, clock)
        self.tasks = TasksRepository(self.database, clock)
        self.lists = ListsRepository(self.database, clock)


def _migration_1(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            tags_json TEXT NOT NULL DEFAULT '[]',
            search_text TEXT NOT NULL DEFAULT '',
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS notes_updated_idx ON notes(updated_at_utc DESC);

        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            normalized_title TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'completed')),
            priority TEXT,
            due_at_utc TEXT,
            timezone_name TEXT,
            reminder_id INTEGER,
            project TEXT,
            category TEXT,
            recurrence TEXT,
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL,
            completed_at_utc TEXT
        );
        CREATE INDEX IF NOT EXISTS tasks_status_due_idx ON tasks(status, due_at_utc);
        CREATE INDEX IF NOT EXISTS tasks_title_idx ON tasks(normalized_title);

        CREATE TABLE IF NOT EXISTS lists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            normalized_name TEXT NOT NULL UNIQUE,
            aliases_json TEXT NOT NULL DEFAULT '[]',
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS list_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            list_id INTEGER NOT NULL REFERENCES lists(id) ON DELETE CASCADE,
            text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            completed INTEGER NOT NULL DEFAULT 0 CHECK(completed IN (0, 1)),
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL,
            UNIQUE(list_id, normalized_text)
        );
        CREATE INDEX IF NOT EXISTS list_items_list_idx
            ON list_items(list_id, completed, created_at_utc);

        CREATE TRIGGER IF NOT EXISTS notes_search_insert
        AFTER INSERT ON notes BEGIN
            UPDATE notes SET search_text = lower(
                NEW.title || ' ' || NEW.content || ' ' || NEW.tags_json
            ) WHERE id = NEW.id;
        END;
        """
    )


def _note_from_row(row: sqlite3.Row) -> Note:
    try:
        tags = tuple(str(item) for item in json.loads(str(row["tags_json"])))
    except (json.JSONDecodeError, TypeError):
        tags = ()
    return Note(
        id=int(row["id"]),
        title=str(row["title"]),
        content=str(row["content"]),
        tags=tags,
        created_at_utc=str(row["created_at_utc"]),
        updated_at_utc=str(row["updated_at_utc"]),
    )


def _task_from_row(row: sqlite3.Row) -> Task:
    return Task(
        id=int(row["id"]),
        title=str(row["title"]),
        status=str(row["status"]),
        priority=_row_optional(row, "priority"),
        due_at_utc=_row_optional(row, "due_at_utc"),
        timezone_name=_row_optional(row, "timezone_name"),
        reminder_id=None if row["reminder_id"] is None else int(row["reminder_id"]),
        project=_row_optional(row, "project"),
        category=_row_optional(row, "category"),
        recurrence=_row_optional(row, "recurrence"),
        created_at_utc=str(row["created_at_utc"]),
        updated_at_utc=str(row["updated_at_utc"]),
        completed_at_utc=_row_optional(row, "completed_at_utc"),
    )


def _list_from_row(row: sqlite3.Row) -> PersonalList:
    try:
        aliases = tuple(str(item) for item in json.loads(str(row["aliases_json"])))
    except (json.JSONDecodeError, TypeError):
        aliases = ()
    return PersonalList(
        id=int(row["id"]),
        name=str(row["name"]),
        aliases=aliases,
        created_at_utc=str(row["created_at_utc"]),
        updated_at_utc=str(row["updated_at_utc"]),
    )


def _item_from_row(row: sqlite3.Row) -> ListItem:
    return ListItem(
        id=int(row["id"]),
        list_id=int(row["list_id"]),
        text=str(row["text"]),
        completed=bool(row["completed"]),
        created_at_utc=str(row["created_at_utc"]),
        updated_at_utc=str(row["updated_at_utc"]),
    )


def _row_optional(row: sqlite3.Row, key: str) -> str | None:
    return None if row[key] is None else str(row[key])


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _normalize(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text.casefold())
    return " ".join(text.split())


def _search_terms(value: str) -> tuple[str, ...]:
    terms: list[str] = []
    for word in _normalize(value).split():
        if word in STOP_WORDS or len(word) < 2:
            continue
        terms.append("air" if word == "ac" else word)
    return tuple(dict.fromkeys(terms))


def _search_score(query: str, terms: Sequence[str], haystack: str) -> tuple[int, int]:
    return (10 if query and query in haystack else 0) + sum(
        2 if re.search(rf"\b{re.escape(term)}\b", haystack) else 1 if term in haystack else 0
        for term in terms
    ), -len(haystack)


def _required_text(value: Any, message: str, maximum: int) -> str:
    text = " ".join(str(value).split())
    if not text:
        raise ValueError(message)
    if len(text) > maximum:
        raise ValueError(f"Text is too long; maximum length is {maximum} characters.")
    return text


def _optional_text(value: Any, maximum: int) -> str | None:
    if value is None or value is ...:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    if len(text) > maximum:
        raise ValueError(f"Text is too long; maximum length is {maximum} characters.")
    return text


def _clean_tags(tags: Sequence[str]) -> tuple[str, ...]:
    result = []
    for value in tags[:20]:
        text = _optional_text(value, 80)
        if text and _normalize(text) not in {_normalize(item) for item in result}:
            result.append(text)
    return tuple(result)


def _clean_aliases(aliases: Sequence[str], list_name: str) -> tuple[str, ...]:
    result = []
    list_key = _normalize(list_name)
    for value in aliases[:20]:
        text = _optional_text(value, 160)
        if text and _normalize(text) != list_key and _normalize(text) not in {_normalize(item) for item in result}:
            result.append(text)
    return tuple(result)


def _deduplicate_input(items: Sequence[str]) -> list[str]:
    result: list[str] = []
    keys: set[str] = set()
    for value in items[:50]:
        text = _required_text(value, "List items cannot be empty.", 500)
        key = _normalize(text)
        if key not in keys:
            result.append(text)
            keys.add(key)
    return result


def _validate_status(status: str) -> None:
    if status not in TASK_STATUSES:
        raise ValueError("Task status must be pending, completed, or all.")


def _limit(value: int, maximum: int) -> int:
    return max(1, min(maximum, int(value)))
