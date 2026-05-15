"""HTTP request and response models for the public API."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class AccessRequest(BaseModel):
    email: EmailStr


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ClarificationResponsePayload(BaseModel):
    kind: str = "clarification_required"
    reason: str
    question: str
    hints: list[str] = []
    original_question: str = ""


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    kind: str = "answer"
    clarification: ClarificationResponsePayload | None = None
    citation_urls: list[str] = Field(default_factory=list)
    remaining: int
    quota: int


class QuotaResponse(BaseModel):
    quota: int
    used: int
    remaining: int
    status: str


class SubscriptionRequest(BaseModel):
    email: EmailStr
    name: str | None = None
    sources: list[str] | None = None
    frequency: str = "daily"
    timezone: str = "Asia/Shanghai"


class UnsubscribeRequest(BaseModel):
    email: EmailStr


class SubscriptionResponse(BaseModel):
    email: EmailStr
    name: str | None = None
    is_active: bool
    sources: list[str]
    frequency: str
    timezone: str
