FROM python:3.7-slim

RUN apt-get update

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install -r requirements.txt

RUN ls -la $APP_HOME/

EXPOSE 8501

CMD ["streamlit","run","--server.enableCORS","false","--server.enableXsrfProtection","false","main.py"]