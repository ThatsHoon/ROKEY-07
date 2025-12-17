from sqlalchemy import Column, String, ForeignKey, DateTime, Text
from datetime import datetime
import uuid

from app.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class AuditLog(Base):
    """감사 로그 테이블"""
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)

    user_id = Column(String(36), ForeignKey("users.id"))  # 수행자
    action = Column(String(50), nullable=False)  # CREATE, UPDATE, DELETE

    table_name = Column(String(100), nullable=False)  # 대상 테이블
    record_id = Column(String(100))  # 대상 레코드 ID

    old_values = Column(Text)  # JSON as Text for SQLite
    new_values = Column(Text)  # JSON as Text for SQLite

    ip_address = Column(String(45))  # IPv4/IPv6
    user_agent = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<AuditLog {self.action} on {self.table_name}>"
