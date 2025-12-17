from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
import json

from app.database import SessionLocal
from app.models.audit_log import AuditLog


class AuditMiddleware(BaseHTTPMiddleware):
    """감사 로그 미들웨어"""

    AUDIT_METHODS = ["POST", "PUT", "PATCH", "DELETE"]

    async def dispatch(self, request: Request, call_next):
        # Only audit modifying operations
        if request.method not in self.AUDIT_METHODS:
            return await call_next(request)

        # Skip certain paths
        skip_paths = ["/api/v1/auth/login", "/api/v1/auth/refresh", "/health", "/docs", "/openapi.json"]
        if any(request.url.path.startswith(path) for path in skip_paths):
            return await call_next(request)

        # Get request body
        body = None
        try:
            body = await request.body()
            # Store body for later use
            request.state.body = body
        except:
            pass

        response = await call_next(request)

        # Log only successful operations
        if 200 <= response.status_code < 300:
            try:
                await self._create_audit_log(request, body)
            except Exception as e:
                print(f"Audit log error: {e}")

        return response

    async def _create_audit_log(self, request: Request, body: bytes):
        """감사 로그 생성"""
        db = SessionLocal()
        try:
            # Parse action from method
            action_map = {
                "POST": "CREATE",
                "PUT": "UPDATE",
                "PATCH": "UPDATE",
                "DELETE": "DELETE"
            }
            action = action_map.get(request.method, "UNKNOWN")

            # Parse table name from path
            path_parts = request.url.path.split("/")
            table_name = path_parts[3] if len(path_parts) > 3 else "unknown"

            # Get user ID from token (if available)
            user_id = None
            if hasattr(request.state, "user"):
                user_id = request.state.user.id

            # Parse body
            new_values = None
            if body:
                try:
                    new_values = json.loads(body.decode())
                    # Remove sensitive fields
                    if "password" in new_values:
                        new_values["password"] = "***"
                except:
                    pass

            audit_log = AuditLog(
                user_id=user_id,
                action=action,
                table_name=table_name,
                new_values=new_values,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent")
            )
            db.add(audit_log)
            db.commit()
        finally:
            db.close()
