from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class UserProfile(Base):
    """
    Anonymous user identity anchored to a browser localStorage UUID.
    No login required — created on first request, recognised on every subsequent one.
    """
    __tablename__ = "user_profiles"

    user_id      = Column(String(64), primary_key=True)
    display_name = Column(String(64), nullable=True)          # e.g. "User-a3f9b2"
    created_at   = Column(DateTime,   nullable=False, default=datetime.utcnow)
    last_seen    = Column(DateTime,   nullable=False, default=datetime.utcnow)


class ChatSession(Base):
    """
    One browser tab / conversation = one session.
    Links a session_id (from localStorage) to a user_id.
    """
    __tablename__ = "chat_sessions"

    session_id  = Column(String(64), primary_key=True)
    user_id     = Column(
        String(64),
        ForeignKey("user_profiles.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at  = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_active = Column(DateTime, nullable=False, default=datetime.utcnow)
    # JSON array of text chunks from the last uploaded PDF.
    # Persisted so follow-up questions survive server restarts.
    pdf_chunks  = Column(Text, nullable=True)


class QueryAnalytic(Base):
    """
    One row per LLM call.  Captures full cost/perf breakdown for every query.
    """
    __tablename__ = "query_analytics"

    query_id        = Column(String(64),  primary_key=True)
    session_id      = Column(
        String(64),
        ForeignKey("chat_sessions.session_id", ondelete="SET NULL"),
        nullable=True,
    )
    user_id         = Column(
        String(64),
        ForeignKey("user_profiles.user_id", ondelete="SET NULL"),
        nullable=True,
    )
    model_used      = Column(String(32),  nullable=False)
    tokens_in       = Column(Integer,     nullable=False)           # full prompt context
    tokens_out      = Column(Integer,     nullable=False)           # completion only
    tokens_attach   = Column(Integer,     nullable=False, default=0)  # file content tokens
    latency_ms      = Column(Float,       nullable=False)
    cost_usd        = Column(Float,       nullable=False)
    cost_inr        = Column(Float,       nullable=False)
    timestamp       = Column(DateTime,    nullable=False, default=datetime.utcnow)
    query_text      = Column(String(500), nullable=True)            # first 500 chars of user message
    has_attachment  = Column(Boolean,     nullable=False, default=False)
    attachment_type = Column(String(16),  nullable=True)            # 'image' | 'pdf' | None
