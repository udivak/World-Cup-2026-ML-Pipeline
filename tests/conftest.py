"""Shared test setup.

Load ``.env`` (if present) before tests are collected so the DB-gated tests — which skip on a
missing ``DATABASE_URL`` — actually run against the live Supabase project during local
development. In CI (no ``.env``), ``load_dotenv`` is a no-op and those tests skip gracefully.
"""

from dotenv import load_dotenv

load_dotenv()
