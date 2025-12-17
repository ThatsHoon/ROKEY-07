from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime, date


# Course Schemas
class CourseBase(BaseModel):
    code: str = Field(..., min_length=1, max_length=20)
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    total_sessions: int = Field(default=0, ge=0)


class CourseCreate(CourseBase):
    teacher_id: Optional[UUID] = None


class CourseUpdate(BaseModel):
    code: Optional[str] = Field(None, min_length=1, max_length=20)
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    teacher_id: Optional[UUID] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    total_sessions: Optional[int] = Field(None, ge=0)


class CourseResponse(CourseBase):
    id: UUID
    teacher_id: Optional[UUID] = None
    teacher_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Class Schemas
class ClassBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    capacity: int = Field(default=30, ge=1)
    schedule: Optional[Dict[str, Any]] = None


class ClassCreate(ClassBase):
    course_id: UUID


class ClassUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=50)
    capacity: Optional[int] = Field(None, ge=1)
    schedule: Optional[Dict[str, Any]] = None


class ClassResponse(ClassBase):
    id: UUID
    course_id: UUID
    course_name: Optional[str] = None
    student_count: int = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Enrollment Schemas
class EnrollmentCreate(BaseModel):
    student_id: UUID
    class_id: UUID


class EnrollmentResponse(BaseModel):
    id: UUID
    student_id: UUID
    class_id: UUID
    enrolled_at: datetime
    dropped_at: Optional[datetime] = None

    student_name: Optional[str] = None
    student_number: Optional[str] = None
    class_name: Optional[str] = None

    class Config:
        from_attributes = True
