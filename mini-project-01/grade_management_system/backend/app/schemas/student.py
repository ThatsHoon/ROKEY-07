from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from datetime import datetime
from app.models.student import StudentStatus


class StudentBase(BaseModel):
    student_number: str = Field(..., min_length=1, max_length=20)


class StudentCreate(StudentBase):
    user_id: Optional[UUID] = None
    # User creation fields (if creating user simultaneously)
    email: Optional[str] = None
    name: Optional[str] = None
    password: Optional[str] = None
    phone: Optional[str] = None


class StudentUpdate(BaseModel):
    student_number: Optional[str] = Field(None, min_length=1, max_length=20)
    status: Optional[StudentStatus] = None


class StudentResponse(StudentBase):
    id: UUID
    user_id: Optional[UUID] = None
    status: StudentStatus
    enrolled_at: datetime
    created_at: datetime
    updated_at: datetime

    # User info (if joined)
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    user_phone: Optional[str] = None

    class Config:
        from_attributes = True


class StudentListResponse(BaseModel):
    total: int
    items: list[StudentResponse]
