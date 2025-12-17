from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID
from datetime import datetime, date
from decimal import Decimal
from app.models.grade import EvaluationType


# Evaluation Schemas
class EvaluationBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    type: EvaluationType
    description: Optional[str] = None
    weight: Decimal = Field(..., ge=0, le=100)  # 가중치 0-100%
    max_score: Decimal = Field(default=Decimal("100"), ge=0)
    due_date: Optional[date] = None


class EvaluationCreate(EvaluationBase):
    course_id: UUID


class EvaluationUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    type: Optional[EvaluationType] = None
    description: Optional[str] = None
    weight: Optional[Decimal] = Field(None, ge=0, le=100)
    max_score: Optional[Decimal] = Field(None, ge=0)
    due_date: Optional[date] = None


class EvaluationResponse(EvaluationBase):
    id: UUID
    course_id: UUID
    course_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Grade Schemas
class GradeBase(BaseModel):
    score: Optional[Decimal] = Field(None, ge=0)
    comments: Optional[str] = None


class GradeCreate(GradeBase):
    evaluation_id: UUID
    student_id: UUID


class GradeBulkCreate(BaseModel):
    """성적 일괄 입력"""
    evaluation_id: UUID
    grades: List[dict]  # [{"student_id": UUID, "score": Decimal, "comments": str}]


class GradeUpdate(BaseModel):
    score: Optional[Decimal] = Field(None, ge=0)
    comments: Optional[str] = None


class GradeResponse(GradeBase):
    id: UUID
    evaluation_id: UUID
    student_id: UUID
    graded_by: Optional[UUID] = None
    graded_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    evaluation_name: Optional[str] = None
    evaluation_type: Optional[EvaluationType] = None
    max_score: Optional[Decimal] = None
    weight: Optional[Decimal] = None
    student_name: Optional[str] = None
    student_number: Optional[str] = None

    class Config:
        from_attributes = True


class StudentGradeSummary(BaseModel):
    """학생별 성적 요약"""
    student_id: UUID
    student_name: str
    student_number: str
    course_id: UUID
    course_name: str

    grades: List[GradeResponse]
    total_weighted_score: Decimal  # 가중치 적용 총점
    letter_grade: str  # A, B, C, D, F
    rank: Optional[int] = None  # 석차


class CourseGradeSummary(BaseModel):
    """과정별 성적 통계"""
    course_id: UUID
    course_name: str
    evaluation_id: UUID
    evaluation_name: str

    total_students: int
    graded_count: int
    average_score: Decimal
    max_score_achieved: Decimal
    min_score_achieved: Decimal
    standard_deviation: Decimal


def calculate_letter_grade(score: Decimal) -> str:
    """점수를 등급으로 변환"""
    if score >= 95:
        return "A+"
    elif score >= 90:
        return "A"
    elif score >= 85:
        return "B+"
    elif score >= 80:
        return "B"
    elif score >= 75:
        return "C+"
    elif score >= 70:
        return "C"
    elif score >= 65:
        return "D+"
    elif score >= 60:
        return "D"
    else:
        return "F"
