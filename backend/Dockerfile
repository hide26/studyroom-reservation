FROM python:3.11-slim
WORKDIR /app
# 의존성 먼저 (레이어 캐시)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# 백엔드 + 프론트엔드 복사
COPY backend/ ./backend/
COPY frontend/ ./frontend/
WORKDIR /app/backend
EXPOSE 8000
# Railway/Render 등은 $PORT를 주입한다. 없으면 8000 기본값.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
