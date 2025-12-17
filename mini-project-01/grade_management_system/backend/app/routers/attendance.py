from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func
from typing import List, Optional
from uuid import UUID
from datetime import date

from app.database import get_db
from app.models.user import User
from app.models.student import Student
from app.models.course import Enrollment
from app.models.attendance import Attendance, AttendanceLog, AttendanceStatus
from app.schemas.attendance import (
    AttendanceCreate, AttendanceUpdate, AttendanceResponse,
    AttendanceBulkCreate, AttendanceSummary, AttendanceLogResponse,
    ClassAttendanceSummary
)
from app.utils.security import get_current_active_user, require_roles

router = APIRouter()


@router.get("", response_model=List[AttendanceResponse])
async def get_attendance(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    class_id: Optional[UUID] = None,
    student_id: Optional[UUID] = None,
    session_no: Optional[int] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    status: Optional[AttendanceStatus] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """출결 목록 조회"""
    query = db.query(Attendance).options(joinedload(Attendance.student))

    if class_id:
        query = query.filter(Attendance.class_id == str(class_id))
    if student_id:
        query = query.filter(Attendance.student_id == str(student_id))
    if session_no:
        query = query.filter(Attendance.session_no == session_no)
    if date_from:
        query = query.filter(Attendance.date >= date_from)
    if date_to:
        query = query.filter(Attendance.date <= date_to)
    if status:
        query = query.filter(Attendance.status == status)

    # Students can only see their own attendance
    if current_user.role.name == "student":
        student = db.query(Student).filter(Student.user_id == current_user.id).first()
        if student:
            query = query.filter(Attendance.student_id == student.id)

    records = query.order_by(Attendance.date.desc(), Attendance.session_no).offset(skip).limit(limit).all()

    result = []
    for record in records:
        resp = AttendanceResponse(
            id=record.id,
            student_id=record.student_id,
            class_id=record.class_id,
            session_no=record.session_no,
            date=record.date,
            status=record.status,
            check_in_time=record.check_in_time,
            check_out_time=record.check_out_time,
            notes=record.notes,
            created_at=record.created_at,
            updated_at=record.updated_at,
            student_name=record.student.user.name if record.student and record.student.user else None,
            student_number=record.student.student_number if record.student else None
        )
        result.append(resp)

    return result


@router.post("", response_model=AttendanceResponse, status_code=status.HTTP_201_CREATED)
async def create_attendance(
    attendance: AttendanceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher"]))
):
    """출결 입력"""
    # Check if already exists
    existing = db.query(Attendance).filter(
        Attendance.student_id == str(attendance.student_id),
        Attendance.class_id == str(attendance.class_id),
        Attendance.session_no == attendance.session_no
    ).first()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="해당 회차의 출결 기록이 이미 존재합니다"
        )

    attendance_data = attendance.model_dump()
    attendance_data['student_id'] = str(attendance.student_id)
    attendance_data['class_id'] = str(attendance.class_id)
    db_attendance = Attendance(
        **attendance_data,
        created_by=str(current_user.id)
    )
    db.add(db_attendance)
    db.commit()
    db.refresh(db_attendance)
    return db_attendance


@router.post("/bulk", response_model=List[AttendanceResponse], status_code=status.HTTP_201_CREATED)
async def create_bulk_attendance(
    bulk_data: AttendanceBulkCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher"]))
):
    """출결 일괄 입력"""
    created_records = []

    for record in bulk_data.records:
        # Check if already exists
        student_id_str = str(record["student_id"])
        existing = db.query(Attendance).filter(
            Attendance.student_id == student_id_str,
            Attendance.class_id == str(bulk_data.class_id),
            Attendance.session_no == bulk_data.session_no
        ).first()

        if existing:
            continue

        db_attendance = Attendance(
            student_id=student_id_str,
            class_id=str(bulk_data.class_id),
            session_no=bulk_data.session_no,
            date=bulk_data.date,
            status=AttendanceStatus(record.get("status", "출석")),
            notes=record.get("notes"),
            created_by=str(current_user.id)
        )
        db.add(db_attendance)
        created_records.append(db_attendance)

    db.commit()
    for record in created_records:
        db.refresh(record)

    return created_records


@router.get("/{attendance_id}", response_model=AttendanceResponse)
async def get_attendance_detail(
    attendance_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """출결 상세 조회"""
    record = db.query(Attendance).options(
        joinedload(Attendance.student)
    ).filter(Attendance.id == str(attendance_id)).first()

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="출결 기록을 찾을 수 없습니다"
        )

    return record


@router.put("/{attendance_id}", response_model=AttendanceResponse)
async def update_attendance(
    attendance_id: UUID,
    attendance_update: AttendanceUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher"]))
):
    """출결 수정 (변경 이력 기록)"""
    record = db.query(Attendance).filter(Attendance.id == str(attendance_id)).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="출결 기록을 찾을 수 없습니다"
        )

    # Create change log
    if attendance_update.status and attendance_update.status != record.status:
        log = AttendanceLog(
            attendance_id=str(attendance_id),
            changed_by=str(current_user.id),
            old_status=record.status,
            new_status=attendance_update.status,
            reason=attendance_update.change_reason
        )
        db.add(log)

    # Update record
    update_data = attendance_update.model_dump(exclude_unset=True, exclude={"change_reason"})
    for field, value in update_data.items():
        setattr(record, field, value)

    db.commit()
    db.refresh(record)
    return record


@router.get("/{attendance_id}/logs", response_model=List[AttendanceLogResponse])
async def get_attendance_logs(
    attendance_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher", "staff"]))
):
    """출결 변경 이력 조회"""
    logs = db.query(AttendanceLog).filter(
        AttendanceLog.attendance_id == str(attendance_id)
    ).order_by(AttendanceLog.changed_at.desc()).all()

    return logs


@router.get("/class/{class_id}/summary", response_model=ClassAttendanceSummary)
async def get_class_attendance_summary(
    class_id: UUID,
    session_no: int,
    target_date: date,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher", "staff"]))
):
    """반별 출결 현황"""
    from app.models.course import Class

    cls = db.query(Class).filter(Class.id == str(class_id)).first()
    if not cls:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="반을 찾을 수 없습니다"
        )

    total_students = db.query(Enrollment).filter(
        Enrollment.class_id == str(class_id),
        Enrollment.dropped_at.is_(None)
    ).count()

    records = db.query(Attendance).filter(
        Attendance.class_id == str(class_id),
        Attendance.session_no == session_no,
        Attendance.date == target_date
    ).all()

    present_count = sum(1 for r in records if r.status == AttendanceStatus.PRESENT)
    late_count = sum(1 for r in records if r.status == AttendanceStatus.LATE)
    absent_count = sum(1 for r in records if r.status == AttendanceStatus.ABSENT)

    attendance_rate = (present_count + late_count * 0.5) / total_students * 100 if total_students > 0 else 0

    return ClassAttendanceSummary(
        class_id=class_id,
        class_name=cls.name,
        total_students=total_students,
        session_no=session_no,
        date=target_date,
        present_count=present_count,
        late_count=late_count,
        absent_count=absent_count,
        attendance_rate=round(attendance_rate, 2)
    )


@router.get("/student/{student_id}/summary", response_model=AttendanceSummary)
async def get_student_attendance_summary(
    student_id: UUID,
    class_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """학생별 출결 집계"""
    student = db.query(Student).options(joinedload(Student.user)).filter(Student.id == str(student_id)).first()
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="학생을 찾을 수 없습니다"
        )

    # Students can only view their own summary
    if current_user.role.name == "student" and student.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="접근 권한이 없습니다"
        )

    records = db.query(Attendance).filter(
        Attendance.student_id == str(student_id),
        Attendance.class_id == str(class_id)
    ).all()

    total_sessions = len(records)
    present_count = sum(1 for r in records if r.status == AttendanceStatus.PRESENT)
    late_count = sum(1 for r in records if r.status == AttendanceStatus.LATE)
    early_leave_count = sum(1 for r in records if r.status == AttendanceStatus.EARLY_LEAVE)
    absent_count = sum(1 for r in records if r.status == AttendanceStatus.ABSENT)
    excused_count = sum(1 for r in records if r.status == AttendanceStatus.EXCUSED)

    # 출석률: (출석 + 지각*0.5 + 공결) / 전체
    attendance_rate = (present_count + late_count * 0.5 + excused_count) / total_sessions * 100 if total_sessions > 0 else 0

    return AttendanceSummary(
        student_id=student_id,
        student_name=student.user.name if student.user else "Unknown",
        student_number=student.student_number,
        total_sessions=total_sessions,
        present_count=present_count,
        late_count=late_count,
        early_leave_count=early_leave_count,
        absent_count=absent_count,
        excused_count=excused_count,
        attendance_rate=round(attendance_rate, 2)
    )
