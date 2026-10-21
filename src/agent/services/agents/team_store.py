"""Persistent store for Multi-Agent Teams and Workflow Executions."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any

from agent.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

__all__ = ["TeamRegistryStore"]


class TeamRegistryStore:
    """SQLite-backed storage for Multi-Agent Teams and audit-trailed Workflow Executions."""

    def __init__(self, db_path: str | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self.db_path = db_path or self._settings.agent_registry_db_path

        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables for teams and execution runs."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS multi_agent_teams (
                    team_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    strategy TEXT NOT NULL,
                    members_json TEXT NOT NULL,
                    config_json TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_executions (
                    run_id TEXT PRIMARY KEY,
                    team_id TEXT,
                    strategy TEXT NOT NULL,
                    status TEXT NOT NULL,
                    query TEXT NOT NULL,
                    final_answer TEXT,
                    steps_json TEXT DEFAULT '[]',
                    requires_approval INTEGER DEFAULT 0,
                    approval_status TEXT DEFAULT 'NONE',
                    approval_comment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()

    # Team Management
    def save_team(
        self,
        team_id: str,
        name: str,
        strategy: str,
        members: list[str] | list[dict[str, Any]],
        description: str = "",
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert or update a multi-agent team definition."""
        members_json = json.dumps(members)
        config_json = json.dumps(config or {})

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO multi_agent_teams (
                    team_id, name, description, strategy, members_json, config_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(team_id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    strategy = excluded.strategy,
                    members_json = excluded.members_json,
                    config_json = excluded.config_json,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                (team_id, name, description, strategy, members_json, config_json),
            )
            conn.commit()

        logger.info("Persisted team '%s' (%s) with strategy '%s'", name, team_id, strategy)
        return self.get_team(team_id) or {}

    def get_team(self, team_id: str) -> dict[str, Any] | None:
        """Fetch a team by its unique identifier."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM multi_agent_teams WHERE team_id = ?", (team_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._team_row_to_dict(row)

    def list_teams(self) -> list[dict[str, Any]]:
        """List all persisted multi-agent teams."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM multi_agent_teams ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [self._team_row_to_dict(r) for r in rows]

    def delete_team(self, team_id: str) -> bool:
        """Delete a multi-agent team."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM multi_agent_teams WHERE team_id = ?", (team_id,))
            conn.commit()
            return cursor.rowcount > 0

    # Workflow Execution History & Auditing
    def create_execution(
        self,
        run_id: str,
        team_id: str | None,
        strategy: str,
        query: str,
        status: str = "RUNNING",
    ) -> dict[str, Any]:
        """Record the start of a workflow execution run."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO workflow_executions (
                    run_id, team_id, strategy, status, query, steps_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, '[]', CURRENT_TIMESTAMP)
                ON CONFLICT(run_id) DO UPDATE SET
                    status = excluded.status,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                (run_id, team_id, strategy, status, query),
            )
            conn.commit()
        return self.get_execution(run_id) or {}

    def update_execution(
        self,
        run_id: str,
        status: str,
        final_answer: str | None = None,
        steps: list[dict[str, Any]] | None = None,
        requires_approval: bool | None = None,
        approval_status: str | None = None,
        approval_comment: str | None = None,
    ) -> dict[str, Any] | None:
        """Update state, outputs, steps, and approval status for an execution run."""
        steps_json = json.dumps(steps) if steps is not None else None
        with self._get_connection() as conn:
            query = """
                UPDATE workflow_executions
                SET status = ?,
                    final_answer = COALESCE(?, final_answer),
                    steps_json = COALESCE(?, steps_json),
                    requires_approval = COALESCE(?, requires_approval),
                    approval_status = COALESCE(?, approval_status),
                    approval_comment = COALESCE(?, approval_comment),
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id = ?;
            """
            conn.execute(
                query,
                (
                    status,
                    final_answer,
                    steps_json,
                    (1 if requires_approval else 0) if requires_approval is not None else None,
                    approval_status,
                    approval_comment,
                    run_id,
                ),
            )
            conn.commit()
        return self.get_execution(run_id)

    def get_execution(self, run_id: str) -> dict[str, Any] | None:
        """Fetch details of a workflow run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM workflow_executions WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._exec_row_to_dict(row)

    def list_executions(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent workflow execution runs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM workflow_executions ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            rows = cursor.fetchall()
            return [self._exec_row_to_dict(r) for r in rows]

    # Helper Serialization
    def _team_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        try:
            data["members"] = json.loads(data.get("members_json") or "[]")
        except Exception:
            data["members"] = []
        try:
            data["config"] = json.loads(data.get("config_json") or "{}")
        except Exception:
            data["config"] = {}
        data.pop("members_json", None)
        data.pop("config_json", None)
        return data

    def _exec_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        try:
            data["steps"] = json.loads(data.get("steps_json") or "[]")
        except Exception:
            data["steps"] = []
        data.pop("steps_json", None)
        data["requires_approval"] = bool(data.get("requires_approval", 0))
        return data
