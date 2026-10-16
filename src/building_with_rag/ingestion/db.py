"""Shared connection helper for the Story 2.2 commands. Never prints a credential."""

from __future__ import annotations

from pymongo import MongoClient
from pymongo.database import Database

from building_with_rag.settings import get_settings


class PrerequisiteError(Exception):
    """A missing setting or unreachable service; the message names it without secret values."""


def require_mongodb_uri() -> str:
    uri = get_settings().mongodb_uri
    if uri is None or not uri.get_secret_value().strip():
        raise PrerequisiteError(
            "MONGODB_URI is empty. Set it in your untracked .env (see .env.example)."
        )
    return uri.get_secret_value().strip()


def require_voyage_key() -> str:
    key = get_settings().voyage_api_key
    if key is None or not key.get_secret_value().strip():
        raise PrerequisiteError(
            "VOYAGE_API_KEY is empty. Set it in your untracked .env (see .env.example)."
        )
    return key.get_secret_value().strip()


def connect() -> tuple[MongoClient, Database]:
    """Connect, ping and return (client, database named by MONGODB_DB_NAME)."""
    uri = require_mongodb_uri()
    client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    try:
        client.admin.command("ping")
    except Exception as exc:
        client.close()
        raise PrerequisiteError(
            f"MongoDB is unreachable ({type(exc).__name__}). Check MONGODB_URI, your network "
            "and that your IP is in the Atlas access list."
        ) from None
    return client, client[get_settings().mongodb_db_name]
