from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional
from uuid import UUID
import json

from app.database import get_db
from app.models.user import User
from app.models.course import Course, Class, Enrollment
from app.schemas.course import (
    CourseCreate, CourseUpdate, CourseResponse,
    ClassBase, ClassCreate, ClassUpdate, ClassResponse,
    EnrollmentResponse
)
from app.utils.security import get_current_active_user, require_roles

router = APIRouter()


# ============ Course Endpoints ============

@router.get("", response_model=List[CourseResponse])
async def get_courses(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    search: Optional[str] = None,
    teacher_id: Optional[UUID] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """과정 목록 조회"""
    query = db.query(Course).options(joinedload(Course.teacher))

    if search:
        query = query.filter(
            (Course.name.ilike(f"%{search}%")) |
            (Course.code.ilike(f"%{search}%"))
        )
    if teacher_id:
        query = query.filter(Course.teacher_id == teacher_id)

    courses = query.offset(skip).limit(limit).all()

    result = []
    for course in courses:
        resp = CourseResponse(
            id=course.id,
            code=course.code,
            name=course.name,
            description=course.description,
            teacher_id=course.teacher_id,
            teacher_name=course.teacher.name if course.teacher else None,
            start_date=course.start_date,
            end_date=course.end_date,
            total_sessions=course.total_sessions,
            created_at=course.created_at,
            updated_at=course.updated_at
        )
        result.append(resp)

    return result


@router.post("", response_model=CourseResponse, status_code=status.HTTP_201_CREATED)
async def create_course(
    course: CourseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """과정 생성"""
    existing = db.query(Course).filter(Course.code == course.code).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 존재하는 과정 코드입니다"
        )

    db_course = Course(**course.model_dump())
    db.add(db_course)
    db.commit()
    db.refresh(db_course)
    return db_course


@router.get("/{course_id}", response_model=CourseResponse)
async def get_course(
    course_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """과정 상세 조회"""
    course = db.query(Course).options(joinedload(Course.teacher)).filter(Course.id == str(course_id)).first()
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="과정을 찾을 수 없습니다"
        )

    return CourseResponse(
        id=course.id,
        code=course.code,
        name=course.name,
        description=course.description,
        teacher_id=course.teacher_id,
        teacher_name=course.teacher.name if course.teacher else None,
        start_date=course.start_date,
        end_date=course.end_date,
        total_sessions=course.total_sessions,
        created_at=course.created_at,
        updated_at=course.updated_at
    )


@router.put("/{course_id}", response_model=CourseResponse)
async def update_course(
    course_id: UUID,
    course_update: CourseUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """과정 수정"""
    course = db.query(Course).filter(Course.id == str(course_id)).first()
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="과정을 찾을 수 없습니다"
        )

    update_data = course_update.model_dump(exclude_unset=True)

    if "code" in update_data:
        existing = db.query(Course).filter(
            Course.code == update_data["code"],
            Course.id != str(course_id)
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="이미 존재하는 과정 코드입니다"
            )

    for field, value in update_data.items():
        setattr(course, field, value)

    db.commit()
    db.refresh(course)
    return course


@router.delete("/{course_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_course(
    course_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """과정 삭제"""
    course = db.query(Course).filter(Course.id == str(course_id)).first()
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="과정을 찾을 수 없습니다"
        )

    db.delete(course)
    db.commit()


# ============ Class Endpoints ============

@router.get("/{course_id}/classes", response_model=List[ClassResponse])
async def get_classes(
    course_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """과정의 반 목록 조회"""
    classes = db.query(Class).filter(Class.course_id == str(course_id)).all()

    result = []
    for cls in classes:
        student_count = db.query(Enrollment).filter(
            Enrollment.class_id == cls.id,
            Enrollment.dropped_at.is_(None)
        ).count()

        resp = ClassResponse(
            id=cls.id,
            course_id=cls.course_id,
            name=cls.name,
            capacity=cls.capacity,
            schedule=json.loads(cls.schedule) if cls.schedule else None,
            student_count=student_count,
            created_at=cls.created_at,
            updated_at=cls.updated_at
        )
        result.append(resp)

    return result


@router.post("/{course_id}/classes", response_model=ClassResponse, status_code=status.HTTP_201_CREATED)
async def create_class(
    course_id: UUID,
    class_data: ClassBase,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """반 생성"""
    course = db.query(Course).filter(Course.id == str(course_id)).first()
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="과정을 찾을 수 없습니다"
        )

    db_class = Class(
        course_id=str(course_id),
        name=class_data.name,
        capacity=class_data.capacity,
        schedule=json.dumps(class_data.schedule) if class_data.schedule else None
    )
    db.add(db_class)
    db.commit()
    db.refresh(db_class)

    return ClassResponse(
        id=db_class.id,
        course_id=db_class.course_id,
        name=db_class.name,
        capacity=db_class.capacity,
        schedule=json.loads(db_class.schedule) if db_class.schedule else None,
        student_count=0,
        created_at=db_class.created_at,
        updated_at=db_class.updated_at
    )


@router.get("/classes/{class_id}", response_model=ClassResponse)
async def get_class(
    class_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """반 상세 조회"""
    cls = db.query(Class).options(joinedload(Class.course)).filter(Class.id == str(class_id)).first()
    if not cls:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="반을 찾을 수 없습니다"
        )

    student_count = db.query(Enrollment).filter(
        Enrollment.class_id == str(class_id),
        Enrollment.dropped_at.is_(None)
    ).count()

    return ClassResponse(
        id=cls.id,
        course_id=cls.course_id,
        course_name=cls.course.name if cls.course else None,
        name=cls.name,
        capacity=cls.capacity,
        schedule=json.loads(cls.schedule) if cls.schedule else None,
        student_count=student_count,
        created_at=cls.created_at,
        updated_at=cls.updated_at
    )


@router.put("/classes/{class_id}", response_model=ClassResponse)
async def update_class(
    class_id: UUID,
    class_update: ClassUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """반 수정"""
    cls = db.query(Class).filter(Class.id == str(class_id)).first()
    if not cls:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="반을 찾을 수 없습니다"
        )

    update_data = class_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if field == 'schedule' and value is not None:
            value = json.dumps(value)
        setattr(cls, field, value)

    db.commit()
    db.refresh(cls)

    student_count = db.query(Enrollment).filter(
        Enrollment.class_id == str(class_id),
        Enrollment.dropped_at.is_(None)
    ).count()

    return ClassResponse(
        id=cls.id,
        course_id=cls.course_id,
        name=cls.name,
        capacity=cls.capacity,
        schedule=json.loads(cls.schedule) if cls.schedule else None,
        student_count=student_count,
        created_at=cls.created_at,
        updated_at=cls.updated_at
    )


@router.delete("/classes/{class_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_class(
    class_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """반 삭제"""
    cls = db.query(Class).filter(Class.id == str(class_id)).first()
    if not cls:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="반을 찾을 수 없습니다"
        )

    db.delete(cls)
    db.commit()


@router.get("/classes/{class_id}/students", response_model=List[EnrollmentResponse])
async def get_class_students(
    class_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """반 수강생 목록 조회"""
    enrollments = db.query(Enrollment).options(
        joinedload(Enrollment.student)
    ).filter(
        Enrollment.class_id == str(class_id),
        Enrollment.dropped_at.is_(None)
    ).all()

    result = []
    for enrollment in enrollments:
        resp = EnrollmentResponse(
            id=enrollment.id,
            student_id=enrollment.student_id,
            class_id=enrollment.class_id,
            enrolled_at=enrollment.enrolled_at,
            dropped_at=enrollment.dropped_at,
            student_name=enrollment.student.user.name if enrollment.student and enrollment.student.user else None,
            student_number=enrollment.student.student_number if enrollment.student else None
        )
        result.append(resp)

    return result
