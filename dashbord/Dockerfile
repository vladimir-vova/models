FROM python:3.10-slim-buster

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY dashboard.py ./
COPY app.py ./

RUN python dashboard.py

EXPOSE 9050
CMD ["python", "./app.py"]