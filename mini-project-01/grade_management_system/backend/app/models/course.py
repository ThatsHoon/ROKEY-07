from sqlalchemy import Column, String, Integer, ForeignKey, DateTime, Date, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class Course(Base):
    """과정 테이블"""
    __tablename__ = "courses"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    code = Column(String(20), unique=True, nullable=False, index=True)  # 과정 코드
    name = Column(String(200), nullable=False)
    description = Column(Text)

    teacher_id = Column(String(36), ForeignKey("users.id"))  # 담당 강사

    start_date = Column(Date)
    end_date = Column(Date)
    total_sessions = Column(Integer, default=0)  # 총 회차

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    teacher = relationship("User", back_populates="taught_courses")
    classes = relationship("Class", back_populates="course")
    evaluations = relationship("Evaluation", back_populates="course")

    def __repr__(self):
        return f"<Course {self.code}: {self.name}>"


class Class(Base):
    """반 테이블"""
    __tablename__ = "classes"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    course_id = Column(String(36), ForeignKey("courses.id"), nullable=False)

    name = Column(String(50), nullable=False)  # 반 이름 (예: A반, B반)
    capacity = Column(Integer, default=30)  # 정원
    schedule = Column(Text)  # JSON as Text for SQLite

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    course = relationship("Course", back_populates="classes")
    enrollments = relationship("Enrollment", back_populates="class_")
    attendance_records = relationship("Attendance", back_populates="class_")

    def __repr__(self):
        return f"<Class {self.name}>"


class Enrollment(Base):
    """수강 신청 테이블"""
    __tablename__ = "enrollments"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    student_id = Column(String(36), ForeignKey("students.id"), nullable=False)
    class_id = Column(String(36), ForeignKey("classes.id"), nullable=False)

    enrolled_at = Column(DateTime, default=datetime.utcnow)
    dropped_at = Column(DateTime)  # 수강 취소일

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    student = relationship("Student", back_populates="enrollments")
    class_ = relationship("Class", back_populates="enrollments")

    def __repr__(self):
        return f"<Enrollment student={self.student_id} class={self.class_id}>"
