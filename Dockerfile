FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY frontend ./frontend

# model.h5 is expected to be mounted or copied in at deploy time
# (kept out of the image so it doesn't bloat every rebuild).
ENV MODEL_PATH=/srv/model.h5

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
