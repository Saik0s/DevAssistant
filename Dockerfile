FROM python:3.11

ENV PYTHONBUFFERED 1

RUN apt update
RUN apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev -y

RUN ln -s /usr/bin/python3.11 /usr/bin/python

RUN apt install python3-pip -y
RUN apt install python3-venv -y

RUN python -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install -U SQLAlchemy

COPY . .

RUN python -c "import nltk; nltk.download('punkt')"
CMD [ "python", "-u", "-m" , "main"]
# CMD [ "python", "-u", "-m" , "main", "--test", "--verbose"]
