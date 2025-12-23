from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional
from uuid import UUID

from app.database import get_db
from app.models.user import User, Role
from app.models.student import Student, StudentStatus
from app.models.course import Enrollment
from app.schemas.student import StudentCreate, StudentUpdate, StudentResponse
from app.schemas.course import EnrollmentCreate, EnrollmentResponse
from app.utils.security import get_password_hash, get_current_active_user, require_roles

router = APIRouter()


@router.get("", response_model=List[StudentResponse])
async def get_students(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[StudentStatus] = None,
    search: Optional[str] = None,
    class_id: Optional[UUID] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher", "staff"]))
):
    """학생 목록 조회"""
    query = db.query(Student).options(joinedload(Student.user))

    if status:
        query = query.filter(Student.status == status)
    if search:
        query = query.join(User).filter(
            (User.name.ilike(f"%{search}%")) |
            (Student.student_number.ilike(f"%{search}%"))
        )
    if class_id:
        query = query.join(Enrollment).filter(Enrollment.class_id == str(class_id))

    students = query.offset(skip).limit(limit).all()

    # Convert to response
    result = []
    for student in students:
        resp = StudentResponse(
            id=student.id,
            user_id=student.user_id,
            student_number=student.student_number,
            status=student.status,
            enrolled_at=student.enrolled_at,
            created_at=student.created_at,
            updated_at=student.updated_at,
            user_name=student.user.name if student.user else None,
            user_email=student.user.email if student.user else None,
            user_phone=student.user.phone if student.user else None
        )
        result.append(resp)

    return result


@router.post("", response_model=StudentResponse, status_code=status.HTTP_201_CREATED)
async def create_student(
    student_data: StudentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "staff"]))
):
    """학생 등록"""
    # Check student number exists
    existing = db.query(Student).filter(Student.student_number == student_data.student_number).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 등록된 학번입니다"
        )

    user_id = student_data.user_id

    # Create user if not provided
    if not user_id and student_data.email:
        # Check email exists
        existing_user = db.query(User).filter(User.email == student_data.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="이미 등록된 이메일입니다"
            )

        # Get student role
        student_role = db.query(Role).filter(Role.name == "student").first()
        if not student_role:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="학생 역할이 설정되지 않았습니다"
            )

        new_user = User(
            email=student_data.email,
            password_hash=get_password_hash(student_data.password or "default123"),
            name=student_data.name or "Unknown",
            phone=student_data.phone,
            role_id=student_role.id
        )
        db.add(new_user)
        db.flush()
        user_id = new_user.id

    db_student = Student(
        user_id=user_id,
        student_number=student_data.student_number
    )
    db.add(db_student)
    db.commit()
    db.refresh(db_student)

    # Load user info
    if db_student.user:
        return StudentResponse(
            id=db_student.id,
            user_id=db_student.user_id,
            student_number=db_student.student_number,
            status=db_student.status,
            enrolled_at=db_student.enrolled_at,
            created_at=db_student.created_at,
            updated_at=db_student.updated_at,
            user_name=db_student.user.name,
            user_email=db_student.user.email,
            user_phone=db_student.user.phone
        )

    return db_student


@router.get("/{student_id}", response_model=StudentResponse)
async def get_student(
    student_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """학생 상세 조회"""
    student = db.query(Student).options(joinedload(Student.user)).filter(Student.id == str(student_id)).first()

    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="학생을 찾을 수 없습니다"
        )

    # Students can only view their own info
    if current_user.role.name == "student" and student.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="접근 권한이 없습니다"
        )

    return StudentResponse(
        id=student.id,
        user_id=student.user_id,
        student_number=student.student_number,
        status=student.status,
        enrolled_at=student.enrolled_at,
        created_at=student.created_at,
        updated_at=student.updated_at,
        user_name=student.user.name if student.user else None,
        user_email=student.user.email if student.user else None,
        user_phone=student.user.phone if student.user else None
    )


@router.put("/{student_id}", response_model=StudentResponse)
async def update_student(
    student_id: UUID,
    student_update: StudentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "staff"]))
):
    """학생 정보 수정"""
    student = db.query(Student).filter(Student.id == str(student_id)).first()
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="학생을 찾을 수 없습니다"
        )

    update_data = student_update.model_dump(exclude_unset=True)

    # Check student number uniqueness
    if "student_number" in update_data:
        existing = db.query(Student).filter(
            Student.student_number == update_data["student_number"],
            Student.id != student_id
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="이미 등록된 학번입니다"
            )

    for field, value in update_data.items():
        setattr(student, field, value)

    db.commit()
    db.refresh(student)
    return student


@router.delete("/{student_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_student(
    student_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """학생 삭제 (실제 삭제)"""
    student = db.query(Student).filter(Student.id == str(student_id)).first()
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="학생을 찾을 수 없습니다"
        )

    # 관련 데이터 삭제 (수강, 출결, 성적)
    from app.models.course import Enrollment
    from app.models.attendance import Attendance
    from app.models.grade import Grade

    db.query(Grade).filter(Grade.student_id == str(student_id)).delete()
    db.query(Attendance).filter(Attendance.student_id == str(student_id)).delete()
    db.query(Enrollment).filter(Enrollment.student_id == str(student_id)).delete()

    # 연결된 사용자 정보 저장
    user = student.user

    # 학생 삭제
    db.delete(student)

    # 연결된 사용자도 삭제
    if user:
        db.delete(user)

    db.commit()


# ============ Enrollment Endpoints ============

@router.post("/{student_id}/enroll", response_model=EnrollmentResponse, status_code=status.HTTP_201_CREATED)
async def enroll_student(
    student_id: UUID,
    enrollment: EnrollmentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "staff"]))
):
    """수강 신청"""
    student = db.query(Student).filter(Student.id == str(student_id)).first()
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="학생을 찾을 수 없습니다"
        )

    # Check if already enrolled
    existing = db.query(Enrollment).filter(
        Enrollment.student_id == str(student_id),
        Enrollment.class_id == str(enrollment.class_id),
        Enrollment.dropped_at.is_(None)
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 등록된 수강 신청입니다"
        )

    db_enrollment = Enrollment(
        student_id=str(student_id),
        class_id=str(enrollment.class_id)
    )
    db.add(db_enrollment)
    db.commit()
    db.refresh(db_enrollment)
    return db_enrollment


@router.get("/{student_id}/enrollments", response_model=List[EnrollmentResponse])
async def get_student_enrollments(
    student_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """학생 수강 목록 조회"""
    enrollments = db.query(Enrollment).filter(
        Enrollment.student_id == str(student_id)
    ).all()
    return enrollments
