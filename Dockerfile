FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app
COPY research research
RUN useradd --create-home lab && mkdir -p /data && chown lab:lab /data
USER lab
ENV APP_DATA_DIR=/data MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-proxy-headers"]
