from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from decimal import Decimal

from app.database import get_db
from app.models.user import User
from app.models.student import Student
from app.models.course import Course, Class, Enrollment
from app.models.grade import Evaluation, Grade, EvaluationType
from app.schemas.grade import (
    EvaluationCreate, EvaluationUpdate, EvaluationResponse,
    GradeCreate, GradeUpdate, GradeResponse,
    GradeBulkCreate, StudentGradeSummary, CourseGradeSummary,
    calculate_letter_grade
)
from app.utils.security import get_current_active_user, require_roles

router = APIRouter()


# ============ Evaluation Endpoints ============

@router.get("/evaluations", response_model=List[EvaluationResponse])
async def get_evaluations(
    course_id: Optional[UUID] = None,
    type: Optional[EvaluationType] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """평가항목 목록 조회"""
    query = db.query(Evaluation).options(joinedload(Evaluation.course))

    if course_id:
        query = query.filter(Evaluation.course_id == str(course_id))
    if type:
        query = query.filter(Evaluation.type == type)

    evaluations = query.all()

    result = []
    for eval in evaluations:
        resp = EvaluationResponse(
            id=eval.id,
            course_id=eval.course_id,
            course_name=eval.course.name if eval.course else None,
            name=eval.name,
            type=eval.type,
            description=eval.description,
            weight=eval.weight,
            max_score=eval.max_score,
            due_date=eval.due_date,
            created_at=eval.created_at,
            updated_at=eval.updated_at
        )
        result.append(resp)

    return result


@router.post("/evaluations", response_model=EvaluationResponse, status_code=status.HTTP_201_CREATED)
async def create_evaluation(
    evaluation: EvaluationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher"]))
):
    """평가항목 생성"""
    # Check course exists
    course = db.query(Course).filter(Course.id == str(evaluation.course_id)).first()
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="과정을 찾을 수 없습니다"
        )

    # Check total weight doesn't exceed 100%
    existing_weight = db.query(func.sum(Evaluation.weight)).filter(
        Evaluation.course_id == str(evaluation.course_id)
    ).scalar() or 0

    if existing_weight + float(evaluation.weight) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"가중치 합계가 100%를 초과합니다 (현재: {existing_weight}%)"
        )

    eval_data = evaluation.model_dump()
    eval_data['course_id'] = str(evaluation.course_id)
    db_evaluation = Evaluation(**eval_data)
    db.add(db_evaluation)
    db.commit()
    db.refresh(db_evaluation)
    return db_evaluation


@router.get("/evaluations/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(
    evaluation_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """평가항목 상세 조회"""
    evaluation = db.query(Evaluation).options(
        joinedload(Evaluation.course)
    ).filter(Evaluation.id == str(evaluation_id)).first()

    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="평가항목을 찾을 수 없습니다"
        )

    return evaluation


@router.put("/evaluations/{evaluation_id}", response_model=EvaluationResponse)
async def update_evaluation(
    evaluation_id: UUID,
    evaluation_update: EvaluationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher"]))
):
    """평가항목 수정"""
    evaluation = db.query(Evaluation).filter(Evaluation.id == str(evaluation_id)).first()
    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="평가항목을 찾을 수 없습니다"
        )

    update_data = evaluation_update.model_dump(exclude_unset=True)

    # Check weight constraint if updating weight
    if "weight" in update_data:
        other_weight = db.query(func.sum(Evaluation.weight)).filter(
            Evaluation.course_id == evaluation.course_id,
            Evaluation.id != str(evaluation_id)
        ).scalar() or 0

        if other_weight + float(update_data["weight"]) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"가중치 합계가 100%를 초과합니다 (현재 다른 항목: {other_weight}%)"
            )

    for field, value in update_data.items():
        setattr(evaluation, field, value)

    db.commit()
    db.refresh(evaluation)
    return evaluation


@router.delete("/evaluations/{evaluation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_evaluation(
    evaluation_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """평가항목 삭제"""
    evaluation = db.query(Evaluation).filter(Evaluation.id == str(evaluation_id)).first()
    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="평가항목을 찾을 수 없습니다"
        )

    db.delete(evaluation)
    db.commit()


# ============ Grade Endpoints ============

@router.get("", response_model=List[GradeResponse])
async def get_grades(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    evaluation_id: Optional[UUID] = None,
    student_id: Optional[UUID] = None,
    course_id: Optional[UUID] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """성적 목록 조회"""
    query = db.query(Grade).options(
        joinedload(Grade.evaluation),
        joinedload(Grade.student)
    )

    if evaluation_id:
        query = query.filter(Grade.evaluation_id == str(evaluation_id))
    if student_id:
        query = query.filter(Grade.student_id == str(student_id))
    if course_id:
        query = query.join(Evaluation).filter(Evaluation.course_id == str(course_id))

    # Students can only see their own grades
    if current_user.role.name == "student":
        student = db.query(Student).filter(Student.user_id == current_user.id).first()
        if student:
            query = query.filter(Grade.student_id == student.id)

    grades = query.offset(skip).limit(limit).all()

    result = []
    for grade in grades:
        resp = GradeResponse(
            id=grade.id,
            evaluation_id=grade.evaluation_id,
            student_id=grade.student_id,
            score=grade.score,
            comments=grade.comments,
            graded_by=grade.graded_by,
            graded_at=grade.graded_at,
            created_at=grade.created_at,
            updated_at=grade.updated_at,
            evaluation_name=grade.evaluation.name if grade.evaluation else None,
            evaluation_type=grade.evaluation.type if grade.evaluation else None,
            max_score=grade.evaluation.max_score if grade.evaluation else None,
            weight=grade.evaluation.weight if grade.evaluation else None,
            student_name=grade.student.user.name if grade.student and grade.student.user else None,
            student_number=grade.student.student_number if grade.student else None
        )
        result.append(resp)

    return result


@router.post("", response_model=GradeResponse, status_code=status.HTTP_201_CREATED)
async def create_grade(
    grade: GradeCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher"]))
):
    """성적 입력"""
    # Check evaluation exists
    evaluation = db.query(Evaluation).filter(Evaluation.id == str(grade.evaluation_id)).first()
    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="평가항목을 찾을 수 없습니다"
        )

    # Check score doesn't exceed max
    if grade.score and grade.score > evaluation.max_score:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"점수가 만점({evaluation.max_score})을 초과합니다"
        )

    # Check if already exists
    existing = db.query(Grade).filter(
        Grade.evaluation_id == str(grade.evaluation_id),
        Grade.student_id == str(grade.student_id)
    ).first()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="해당 학생의 성적이 이미 존재합니다"
        )

    grade_data = grade.model_dump()
    grade_data['evaluation_id'] = str(grade.evaluation_id)
    grade_data['student_id'] = str(grade.student_id)
    db_grade = Grade(
        **grade_data,
        graded_by=str(current_user.id),
        graded_at=datetime.utcnow()
    )
    db.add(db_grade)
    db.commit()
    db.refresh(db_grade)
    return db_grade


@router.post("/bulk", response_model=List[GradeResponse], status_code=status.HTTP_201_CREATED)
async def create_bulk_grades(
    bulk_data: GradeBulkCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher"]))
):
    """성적 일괄 입력"""
    evaluation = db.query(Evaluation).filter(Evaluation.id == str(bulk_data.evaluation_id)).first()
    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="평가항목을 찾을 수 없습니다"
        )

    created_grades = []

    for grade_data in bulk_data.grades:
        student_id_str = str(grade_data["student_id"])
        # Check if already exists
        existing = db.query(Grade).filter(
            Grade.evaluation_id == str(bulk_data.evaluation_id),
            Grade.student_id == student_id_str
        ).first()

        if existing:
            continue

        score = Decimal(str(grade_data.get("score", 0)))
        if score > evaluation.max_score:
            continue

        db_grade = Grade(
            evaluation_id=str(bulk_data.evaluation_id),
            student_id=student_id_str,
            score=score,
            comments=grade_data.get("comments"),
            graded_by=str(current_user.id),
            graded_at=datetime.utcnow()
        )
        db.add(db_grade)
        created_grades.append(db_grade)

    db.commit()
    for grade in created_grades:
        db.refresh(grade)

    return created_grades


@router.get("/{grade_id}", response_model=GradeResponse)
async def get_grade(
    grade_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """성적 상세 조회"""
    grade = db.query(Grade).options(
        joinedload(Grade.evaluation),
        joinedload(Grade.student)
    ).filter(Grade.id == str(grade_id)).first()

    if not grade:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="성적을 찾을 수 없습니다"
        )

    return grade


@router.put("/{grade_id}", response_model=GradeResponse)
async def update_grade(
    grade_id: UUID,
    grade_update: GradeUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher"]))
):
    """성적 수정"""
    grade = db.query(Grade).options(joinedload(Grade.evaluation)).filter(Grade.id == str(grade_id)).first()
    if not grade:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="성적을 찾을 수 없습니다"
        )

    update_data = grade_update.model_dump(exclude_unset=True)

    if "score" in update_data and update_data["score"] > grade.evaluation.max_score:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"점수가 만점({grade.evaluation.max_score})을 초과합니다"
        )

    for field, value in update_data.items():
        setattr(grade, field, value)

    grade.graded_by = str(current_user.id)
    grade.graded_at = datetime.utcnow()

    db.commit()
    db.refresh(grade)
    return grade


@router.delete("/{grade_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_grade(
    grade_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"]))
):
    """성적 삭제"""
    grade = db.query(Grade).filter(Grade.id == str(grade_id)).first()
    if not grade:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="성적을 찾을 수 없습니다"
        )

    db.delete(grade)
    db.commit()


@router.get("/student/{student_id}/summary", response_model=StudentGradeSummary)
async def get_student_grade_summary(
    student_id: UUID,
    course_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """학생별 성적 요약 (자동 산출)"""
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

    course = db.query(Course).filter(Course.id == str(course_id)).first()
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="과정을 찾을 수 없습니다"
        )

    # Get all grades for this student in this course
    grades = db.query(Grade).join(Evaluation).filter(
        Grade.student_id == str(student_id),
        Evaluation.course_id == str(course_id)
    ).options(joinedload(Grade.evaluation)).all()

    # Calculate weighted total
    total_weighted_score = Decimal("0")
    grade_responses = []

    for grade in grades:
        if grade.score is not None:
            # 환산점 = (점수/만점) * 가중치
            weighted = (grade.score / grade.evaluation.max_score) * grade.evaluation.weight
            total_weighted_score += weighted

        grade_responses.append(GradeResponse(
            id=grade.id,
            evaluation_id=grade.evaluation_id,
            student_id=grade.student_id,
            score=grade.score,
            comments=grade.comments,
            graded_by=grade.graded_by,
            graded_at=grade.graded_at,
            created_at=grade.created_at,
            updated_at=grade.updated_at,
            evaluation_name=grade.evaluation.name,
            evaluation_type=grade.evaluation.type,
            max_score=grade.evaluation.max_score,
            weight=grade.evaluation.weight
        ))

    letter_grade = calculate_letter_grade(total_weighted_score)

    return StudentGradeSummary(
        student_id=student_id,
        student_name=student.user.name if student.user else "Unknown",
        student_number=student.student_number,
        course_id=course_id,
        course_name=course.name,
        grades=grade_responses,
        total_weighted_score=round(total_weighted_score, 2),
        letter_grade=letter_grade
    )
