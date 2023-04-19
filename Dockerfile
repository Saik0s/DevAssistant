FROM python:3.11

ARG openai_key

ENV PYTHONBUFFERED 1
ENV OPENAI_API_KEY $openai_key

RUN apt update
RUN apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev -y

RUN ln -s /usr/bin/python3.11 /usr/bin/python

RUN apt install python3-pip -y
RUN apt install python3-venv -y

RUN python -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

WORKDIR /app
RUN python -c "import nltk; nltk.download('punkt')"
CMD [ "python", "-u", "-m" , "main"]
