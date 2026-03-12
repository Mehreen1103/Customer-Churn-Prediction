FROM python:3.11

WORKDIR /app

COPY . .
COPY packages/ /packages/

RUN pip install --no-index --find-links=/packages -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]