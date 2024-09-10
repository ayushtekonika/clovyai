FROM python:3.10.12-bullseye

WORKDIR /code

COPY . /code/

ENV PATH="/code:$PATH"

COPY ./requirements.txt ./

RUN apt-get -y update \
    && apt-get -y install --no-install-recommends \
       curl \
    && pip3 install --no-cache --upgrade pip setuptools \
    && pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . .

COPY bin/* /code/bin/

CMD ["bash", "bin/start.sh"]