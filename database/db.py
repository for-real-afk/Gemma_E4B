import json
import logging
import os
import uuid
from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from .models import Base, UserProfile, ChatSession, QueryAnalytic

logger = logging.getLogger(__name__)

# ── connection setup ──────────────────────────────────────────────────────────

_raw_url = os.getenv("DATABASE_URL", "sqlite:///./gemma_chat.db")

# Supabase (and some other providers) return "postgres://" which SQLAlchemy 2.0
# no longer accepts — it requires "postgresql://"
DATABASE_URL = _raw_url.replace("postgres://", "postgresql://", 1) if _raw_url.startswith("postgres://") else _raw_url

_is_sqlite = DATABASE_URL.startswith("sqlite")

if _is_sqlite:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},  # SQLite + multi-thread safety
        pool_pre_ping=True,
    )
else:
    # PostgreSQL / Supabase
    # pool_size + max_overflow kept modest for Supabase free tier (max 60 direct connections)
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,   # recycles stale connections transparently
        pool_size=5,
        max_overflow=10,
    )
    logger.info("Using PostgreSQL backend: %s", DATABASE_URL.split("@")[-1])  # log host only, not password

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Create all tables if they don't exist yet. Safe to call on every startup."""
    Base.metadata.create_all(engine)
    _run_migrations()
    backend = "SQLite" if _is_sqlite else "PostgreSQL"
    logger.info("Database tables ready (%s).", backend)


def _run_migrations() -> None:
    """
    Apply additive schema changes that create_all() won't handle on existing tables.
    Each migration is idempotent — safe to run on every startup.
    """
    try:
        with engine.connect() as conn:
            if _is_sqlite:
                # SQLite: check PRAGMA, add only if missing
                existing = {r[1] for r in conn.execute(text("PRAGMA table_info(chat_sessions)"))}
                if "pdf_chunks" not in existing:
                    conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN pdf_chunks TEXT"))
                    conn.commit()
                    logger.info("Migration: added pdf_chunks column (SQLite).")
            else:
                # PostgreSQL: IF NOT EXISTS handles idempotency natively
                conn.execute(text(
                    "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS pdf_chunks TEXT"
                ))
                conn.commit()
    except Exception as exc:
        logger.warning("Migration skipped (non-fatal): %s", exc)


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def upsert_user(db: Session, user_id: str) -> UserProfile:
    """Create user on first visit; update last_seen on every subsequent request."""
    now  = datetime.utcnow()
    user = db.get(UserProfile, user_id)
    if user is None:
        short = user_id[-6:].upper()
        user  = UserProfile(
            user_id      = user_id,
            display_name = f"User-{short}",
            created_at   = now,
            last_seen    = now,
        )
        db.add(user)
        logger.info("New user created: %s", user_id)
    else:
        user.last_seen = now
    db.commit()
    return user


def upsert_session(db: Session, session_id: str, user_id: str) -> ChatSession:
    """Create session row on first message; bump last_active on every request."""
    now  = datetime.utcnow()
    sess = db.get(ChatSession, session_id)
    if sess is None:
        sess = ChatSession(
            session_id  = session_id,
            user_id     = user_id,
            created_at  = now,
            last_active = now,
        )
        db.add(sess)
        logger.info("New session created: %s (user=%s)", session_id, user_id)
    else:
        sess.last_active = now
    db.commit()
    return sess


def save_pdf_chunks(db: Session, session_id: str, chunks: list[str]) -> None:
    """Persist PDF text chunks for a session so they survive server restarts."""
    sess = db.get(ChatSession, session_id)
    if sess is None:
        logger.warning("save_pdf_chunks: session %s not found", session_id)
        return
    sess.pdf_chunks = json.dumps(chunks)
    db.commit()
    logger.info("Saved %d PDF chunks to DB for session %s", len(chunks), session_id)


def load_pdf_chunks(db: Session, session_id: str) -> list[str] | None:
    """Load PDF text chunks from DB. Returns None if none saved."""
    sess = db.get(ChatSession, session_id)
    if sess is None or not sess.pdf_chunks:
        return None
    try:
        return json.loads(sess.pdf_chunks)
    except Exception as e:
        logger.error("Failed to deserialize pdf_chunks for %s: %s", session_id, e)
        return None


def log_query(
    db:              Session,
    *,
    session_id:      str,
    user_id:         str,
    model_used:      str,
    tokens_in:       int,
    tokens_out:      int,
    tokens_attach:   int,
    latency_ms:      float,
    cost_usd:        float,
    cost_inr:        float,
    query_text:      str | None = None,
    has_attachment:  bool       = False,
    attachment_type: str | None = None,
) -> QueryAnalytic:
    record = QueryAnalytic(
        query_id        = str(uuid.uuid4()),
        session_id      = session_id,
        user_id         = user_id,
        model_used      = model_used,
        tokens_in       = tokens_in,
        tokens_out      = tokens_out,
        tokens_attach   = tokens_attach,
        latency_ms      = round(latency_ms, 2),
        cost_usd        = cost_usd,
        cost_inr        = cost_inr,
        timestamp       = datetime.utcnow(),
        query_text      = (query_text or "")[:500] or None,
        has_attachment  = has_attachment,
        attachment_type = attachment_type,
    )
    db.add(record)
    db.commit()
    return record
