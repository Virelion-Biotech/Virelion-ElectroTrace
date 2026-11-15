FROM python:3.15-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY . /app
RUN python -m pip install --upgrade pip && python -m pip install .
EXPOSE 5000
CMD ["python", "server.py"]
