from sqlalchemy import Column, String, Boolean, Integer, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class Role(Base):
    """역할 테이블"""
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)  # admin, teacher, staff, student
    description = Column(String(255))
    permissions = Column(Text, default="[]")  # JSON as Text for SQLite

    # Relationships
    users = relationship("User", back_populates="role")

    def __repr__(self):
        return f"<Role {self.name}>"


class User(Base):
    """사용자 테이블"""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(100), nullable=False)
    phone = Column(String(20))

    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False)
    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)

    # Relationships
    role = relationship("Role", back_populates="users")
    student = relationship("Student", back_populates="user", uselist=False)
    taught_courses = relationship("Course", back_populates="teacher")
    graded_items = relationship("Grade", back_populates="grader", foreign_keys="Grade.graded_by")

    def __repr__(self):
        return f"<User {self.email}>"
