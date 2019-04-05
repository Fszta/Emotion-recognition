FROM ubuntu:16.04

FROM python:3

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

ADD predict_webcam.py /

ADD requirements.txt /

CMD [ "python", "./predict_webcam.py"]
