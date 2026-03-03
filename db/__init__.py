import os
from flask import Flask, send_from_directory
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    # ✅ 프로젝트 루트 기준 경로 계산
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # ✅ Flask 앱 생성 시 templates, static 경로 명시
    app = Flask(
        __name__,
        template_folder=os.path.join(BASE_DIR, 'templates'),
        static_folder=os.path.join(BASE_DIR, 'static'),
        static_url_path='/static'
    )

    # ✅ Railway 환경변수 DATABASE_URL 사용 (로컬 테스트 대비)
    database_url = os.getenv("DATABASE_URL")

    # PostgreSQL URL 변환 (Railway 기본값은 postgres:// 형태)
    if database_url and database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    # ✅ 로컬 테스트용 SQLite 기본 DB (환경변수가 없을 때 자동 생성)
    if not database_url:
        instance_dir = os.path.join(BASE_DIR, "instance")
        os.makedirs(instance_dir, exist_ok=True)
        database_url = f"sqlite:///{os.path.join(instance_dir, 'app.db')}"

    # ✅ DB 설정 적용
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.secret_key = os.getenv("SECRET_KEY", "studyroom-secret-key")

    # ✅ DB 초기화
    db.init_app(app)

    # ✅ 앱 컨텍스트 내에서 모델 import 및 테이블 생성
    with app.app_context():
        from db import models
        db.create_all()

    # ✅ static 직접 라우트 추가 (Railway에서 PNG/CSS/JS 안 뜨는 현상 해결)
    @app.route('/static/<path:filename>')
    def static_files(filename):
        """Production 환경(Gunicorn)에서도 static 파일 직접 서빙"""
        static_dir = os.path.join(BASE_DIR, 'static')
        return send_from_directory(static_dir, filename)

    return app
