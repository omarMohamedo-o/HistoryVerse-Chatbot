"""SQLite-backed persistent registry store for dynamic custom domain agents."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any

from agent.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

__all__ = ["AgentRegistryStore"]


class AgentRegistryStore:
    """Thread-safe, lightweight persistent storage for dynamically created custom agents."""

    def __init__(self, db_path: str | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self.db_path = db_path or self._settings.agent_registry_db_path

        # In-memory database doesn't need parent directory creation
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create custom_agents table if not already present."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS custom_agents (
                    agent_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    provider_name TEXT,
                    model_name TEXT,
                    temperature REAL,
                    metadata_json TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()

    def save_agent(
        self,
        agent_id: str,
        name: str,
        role: str,
        system_prompt: str,
        domain: str = "general",
        provider_name: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert or replace a custom agent record."""
        # Sanitize metadata to mask any accidental secrets
        clean_meta = self._sanitize_metadata(metadata or {})
        meta_json = json.dumps(clean_meta)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO custom_agents (
                    agent_id, name, role, domain, system_prompt,
                    provider_name, model_name, temperature, metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(agent_id) DO UPDATE SET
                    name = excluded.name,
                    role = excluded.role,
                    domain = excluded.domain,
                    system_prompt = excluded.system_prompt,
                    provider_name = excluded.provider_name,
                    model_name = excluded.model_name,
                    temperature = excluded.temperature,
                    metadata_json = excluded.metadata_json,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                (
                    agent_id,
                    name,
                    role,
                    domain,
                    system_prompt,
                    provider_name,
                    model_name,
                    temperature,
                    meta_json,
                ),
            )
            conn.commit()

        logger.info("Persisted custom agent '%s' (%s) to registry store", name, agent_id)
        return self.get_agent(agent_id) or {}

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        """Fetch a single custom agent record by identifier."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM custom_agents WHERE agent_id = ?", (agent_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_dict(row)

    def list_agents(self) -> list[dict[str, Any]]:
        """List all persisted custom agents."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM custom_agents ORDER BY created_at ASC")
            rows = cursor.fetchall()
            return [self._row_to_dict(r) for r in rows]

    def delete_agent(self, agent_id: str) -> bool:
        """Delete a custom agent record by identifier."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM custom_agents WHERE agent_id = ?", (agent_id,))
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        try:
            data["metadata"] = json.loads(data.get("metadata_json") or "{}")
        except Exception:
            data["metadata"] = {}
        data.pop("metadata_json", None)
        return data

    def _sanitize_metadata(self, meta: dict[str, Any]) -> dict[str, Any]:
        """Mask potential credentials or keys from metadata."""
        sanitized = {}
        sensitive_keys = {"api_key", "secret", "password", "token", "auth"}
        for k, v in meta.items():
            if any(s in k.lower() for s in sensitive_keys):
                sanitized[k] = "******"
            elif isinstance(v, dict):
                sanitized[k] = self._sanitize_metadata(v)
            else:
                sanitized[k] = v
        return sanitized
