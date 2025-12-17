from sqlalchemy import Column, String, ForeignKey, DateTime, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class StudentStatus(enum.Enum):
    """학생 상태"""
    ACTIVE = "재학"
    LEAVE = "휴학"
    GRADUATED = "수료"
    DROPPED = "중퇴"


class Student(Base):
    """학생 테이블"""
    __tablename__ = "students"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), unique=True)
    student_number = Column(String(20), unique=True, nullable=False, index=True)  # 학번

    status = Column(SQLEnum(StudentStatus), default=StudentStatus.ACTIVE)
    enrolled_at = Column(DateTime, default=datetime.utcnow)  # 입학일

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="student")
    enrollments = relationship("Enrollment", back_populates="student")
    attendance_records = relationship("Attendance", back_populates="student")
    grades = relationship("Grade", back_populates="student")

    def __repr__(self):
        return f"<Student {self.student_number}>"
