FROM ubuntu:18.04

LABEL name="animegan"
LABEL version="0.1.0"
LABEL description="Generate random anime characters."

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt update -y && apt install python3.8 python3-pip git -y && pip3 install --no-cache-dir pipenv
RUN python3 --version

ADD . .
RUN pipenv install

EXPOSE 80

CMD ["pipenv", "run", "gunicorn", "-b", "0.0.0.0:80", "app:app", "-k", "gevent"]