FROM python:3.9

COPY ./api /api
COPY requirements.txt /requirements.txt

RUN pip3 install -r requirements.txt

ENV PYTHONPATH=/api
WORKDIR /api

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host" , "0.0.0.0" , "--port" , "8080"]