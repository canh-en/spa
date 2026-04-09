FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.2.2+cu121.html

RUN pip install torch-geometric==2.6.1 \
  -f https://data.pyg.org/whl/torch-2.2.2+cu121.html

# ❌ KHÔNG copy model ở đây

COPY serve.py .

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]