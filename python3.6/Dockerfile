FROM python:3.6-slim

# Tắt buffer để log ra ngay
ENV PYTHONUNBUFFERED=1

# Thư mục làm việc
WORKDIR /app

# Copy file requirements trước (để cache layer)
COPY requirements.txt .

# Cài dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code
COPY . .

# Không set CMD (bạn tự chạy khi cần)