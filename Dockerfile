FROM python:3.8-buster

RUN mkdir /home/guest
WORKDIR /home/guest

COPY requirements.4dock ./
RUN pip install -r requirements.4dock

COPY py ./py
COPY pytest ./pytest
COPY experiments.ipynb ./

ENV PYTHONPATH=.
