from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID
from datetime import datetime, date, time
from app.models.attendance import AttendanceStatus


class AttendanceBase(BaseModel):
    session_no: int = Field(..., ge=1)
    date: date
    status: AttendanceStatus = AttendanceStatus.PRESENT
    check_in_time: Optional[time] = None
    check_out_time: Optional[time] = None
    notes: Optional[str] = None


class AttendanceCreate(AttendanceBase):
    student_id: UUID
    class_id: UUID


class AttendanceBulkCreate(BaseModel):
    """반 전체 출결 일괄 입력"""
    class_id: UUID
    session_no: int = Field(..., ge=1)
    date: date
    records: List[dict]  # [{"student_id": UUID, "status": AttendanceStatus, "notes": str}]


class AttendanceUpdate(BaseModel):
    status: Optional[AttendanceStatus] = None
    check_in_time: Optional[time] = None
    check_out_time: Optional[time] = None
    notes: Optional[str] = None
    change_reason: Optional[str] = None  # 변경 사유 (선택)


class AttendanceResponse(AttendanceBase):
    id: UUID
    student_id: UUID
    class_id: UUID
    created_at: datetime
    updated_at: datetime

    student_name: Optional[str] = None
    student_number: Optional[str] = None

    class Config:
        from_attributes = True


class AttendanceLogResponse(BaseModel):
    id: UUID
    attendance_id: UUID
    changed_by: UUID
    old_status: Optional[AttendanceStatus]
    new_status: AttendanceStatus
    reason: Optional[str]
    changed_at: datetime
    changed_by_name: Optional[str] = None

    class Config:
        from_attributes = True


class AttendanceSummary(BaseModel):
    """출결 집계"""
    student_id: UUID
    student_name: str
    student_number: str
    total_sessions: int
    present_count: int
    late_count: int
    early_leave_count: int
    absent_count: int
    excused_count: int
    attendance_rate: float  # 출석률 (%)


class ClassAttendanceSummary(BaseModel):
    """반별 출결 현황"""
    class_id: UUID
    class_name: str
    total_students: int
    session_no: int
    date: date
    present_count: int
    late_count: int
    absent_count: int
    attendance_rate: float
