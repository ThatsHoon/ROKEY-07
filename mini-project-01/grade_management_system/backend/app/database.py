from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

# Create database engine
# Handle SQLite vs PostgreSQL
if settings.DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables and seed data"""
    Base.metadata.create_all(bind=engine)

    # Seed initial data
    db = SessionLocal()
    try:
        from app.models.user import Role, User
        from passlib.context import CryptContext

        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # Create roles if not exist
        roles_data = [
            {"id": 1, "name": "admin", "description": "시스템 관리자", "permissions": '["*"]'},
            {"id": 2, "name": "teacher", "description": "강사", "permissions": '["course:read", "attendance:*", "grade:*"]'},
            {"id": 3, "name": "staff", "description": "교무 담당자", "permissions": '["course:read", "student:*", "report:*"]'},
            {"id": 4, "name": "student", "description": "학생", "permissions": '["read:own"]'},
        ]

        for role_data in roles_data:
            existing = db.query(Role).filter(Role.id == role_data["id"]).first()
            if not existing:
                role = Role(**role_data)
                db.add(role)

        db.commit()

        # Create admin user if not exist
        admin_email = "admin@example.com"
        existing_admin = db.query(User).filter(User.email == admin_email).first()
        if not existing_admin:
            admin_user = User(
                email=admin_email,
                password_hash=pwd_context.hash("admin123"),
                name="System Admin",
                role_id=1,
                is_active=True
            )
            db.add(admin_user)
            db.commit()
            print(f"Created admin user: {admin_email} / admin123")

    except Exception as e:
        print(f"Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()
