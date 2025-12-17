-- Initial database setup for Grade Management System

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Insert default roles
INSERT INTO roles (id, name, description, permissions) VALUES
(1, 'admin', '시스템 관리자', '["*"]'),
(2, 'teacher', '강사', '["course:read", "course:write", "attendance:*", "grade:*", "student:read"]'),
(3, 'staff', '교무 담당자', '["course:read", "student:*", "attendance:read", "grade:read", "report:*"]'),
(4, 'student', '학생', '["course:read:own", "attendance:read:own", "grade:read:own"]')
ON CONFLICT (id) DO NOTHING;

-- Insert default admin user (password: admin123)
INSERT INTO users (id, email, password_hash, name, role_id, is_active) VALUES
(uuid_generate_v4(), 'admin@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4T/L5TqRxQ4QZLSS', 'System Admin', 1, true)
ON CONFLICT (email) DO NOTHING;

-- Reset sequences
SELECT setval('roles_id_seq', (SELECT MAX(id) FROM roles));
