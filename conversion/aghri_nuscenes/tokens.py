"""Deterministic privacy-safe tokens."""

from __future__ import annotations

import uuid


def deterministic_token(namespace: str | uuid.UUID, canonical_value: str) -> str:
    ns = namespace if isinstance(namespace, uuid.UUID) else uuid.UUID(namespace)
    return uuid.uuid5(ns, canonical_value).hex
