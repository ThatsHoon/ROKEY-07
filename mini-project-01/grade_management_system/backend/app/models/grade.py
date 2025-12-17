from sqlalchemy import Column, String, Integer, ForeignKey, DateTime, Date, Text, Numeric, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class EvaluationType(enum.Enum):
    """평가 유형"""
    MIDTERM = "중간고사"
    FINAL = "기말고사"
    QUIZ = "퀴즈"
    ASSIGNMENT = "과제"
    PROJECT = "프로젝트"
    ATTENDANCE = "출결"
    PRACTICAL = "실기"


class Evaluation(Base):
    """평가항목 테이블"""
    __tablename__ = "evaluations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    course_id = Column(String(36), ForeignKey("courses.id"), nullable=False)

    name = Column(String(100), nullable=False)  # 평가명
    type = Column(SQLEnum(EvaluationType), nullable=False)
    description = Column(Text)

    weight = Column(Numeric(5, 2), nullable=False)  # 가중치 (%)
    max_score = Column(Numeric(5, 2), nullable=False, default=100)  # 만점

    due_date = Column(Date)  # 마감일

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    course = relationship("Course", back_populates="evaluations")
    grades = relationship("Grade", back_populates="evaluation")

    def __repr__(self):
        return f"<Evaluation {self.name} weight={self.weight}%>"


class Grade(Base):
    """성적 테이블"""
    __tablename__ = "grades"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    evaluation_id = Column(String(36), ForeignKey("evaluations.id"), nullable=False)
    student_id = Column(String(36), ForeignKey("students.id"), nullable=False)

    score = Column(Numeric(5, 2))  # 취득 점수
    comments = Column(Text)  # 코멘트/피드백

    graded_by = Column(String(36), ForeignKey("users.id"))  # 채점자
    graded_at = Column(DateTime)  # 채점일시

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    evaluation = relationship("Evaluation", back_populates="grades")
    student = relationship("Student", back_populates="grades")
    grader = relationship("User", back_populates="graded_items", foreign_keys=[graded_by])

    def __repr__(self):
        return f"<Grade student={self.student_id} score={self.score}>"
