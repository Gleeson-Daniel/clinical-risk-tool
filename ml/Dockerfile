FROM python:3.11-slim
WORKDIR /ml
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD ["python", "-u", "train.py"]
