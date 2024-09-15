FROM python:3.10-bookworm

WORKDIR /code

COPY . /code/

ENV PATH="/code:$PATH"

RUN apt-get -y update \
    && apt-get -y install --no-install-recommends curl \
    && pip3 install --no-cache --upgrade pip setuptools \
    && pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY bin/* /code/bin/

CMD ["bash", "bin/start.sh"]
