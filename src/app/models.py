from pydantic import BaseModel, model_validator
from typing import List, Optional


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    session_id: Optional[str] = "default"

    @model_validator(mode='after')
    def validate_user_message_exists(self):
        """Ensure at least one user message exists."""
        has_user_message = any(m.role == "user" for m in self.messages)
        if not has_user_message:
            raise ValueError("No user message provided")
        return self

    @property
    def user_message(self) -> str:
        """Extract the last user message."""
        return next(
            (m.content for m in reversed(self.messages) if m.role == "user"),
            ""
        )