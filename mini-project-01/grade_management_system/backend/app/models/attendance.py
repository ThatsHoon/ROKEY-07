from sqlalchemy import Column, String, Integer, ForeignKey, DateTime, Date, Time, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class AttendanceStatus(enum.Enum):
    """출결 상태"""
    PRESENT = "출석"
    LATE = "지각"
    EARLY_LEAVE = "조퇴"
    ABSENT = "결석"
    EXCUSED = "공결"


class Attendance(Base):
    """출결 테이블"""
    __tablename__ = "attendance"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    student_id = Column(String(36), ForeignKey("students.id"), nullable=False)
    class_id = Column(String(36), ForeignKey("classes.id"), nullable=False)

    session_no = Column(Integer, nullable=False)  # 회차 번호
    date = Column(Date, nullable=False)
    status = Column(SQLEnum(AttendanceStatus), default=AttendanceStatus.PRESENT)

    check_in_time = Column(Time)  # 입실 시간
    check_out_time = Column(Time)  # 퇴실 시간
    notes = Column(Text)  # 비고

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(36), ForeignKey("users.id"))

    # Relationships
    student = relationship("Student", back_populates="attendance_records")
    class_ = relationship("Class", back_populates="attendance_records")
    logs = relationship("AttendanceLog", back_populates="attendance")

    def __repr__(self):
        return f"<Attendance {self.student_id} session={self.session_no} status={self.status}>"


class AttendanceLog(Base):
    """출결 변경 이력 테이블"""
    __tablename__ = "attendance_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    attendance_id = Column(String(36), ForeignKey("attendance.id"), nullable=False)

    changed_by = Column(String(36), ForeignKey("users.id"), nullable=False)
    old_status = Column(SQLEnum(AttendanceStatus))
    new_status = Column(SQLEnum(AttendanceStatus), nullable=False)
    reason = Column(Text)  # 변경 사유

    changed_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    attendance = relationship("Attendance", back_populates="logs")

    def __repr__(self):
        return f"<AttendanceLog {self.attendance_id} {self.old_status} -> {self.new_status}>"
