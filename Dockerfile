# Use Python base image
FROM python:3.10.12-bullseye

# Install necessary packages and update SQLite to version >= 3.35.0
RUN apt-get update && apt-get install -y wget build-essential libsqlite3-dev \
    && wget https://www.sqlite.org/2024/sqlite-autoconf-3410200.tar.gz \
    && tar -xzf sqlite-autoconf-3410200.tar.gz \
    && cd sqlite-autoconf-3410200 \
    && ./configure --prefix=/usr/local \
    && make && make install \
    && cd .. && rm -rf sqlite-autoconf-3410200 sqlite-autoconf-3410200.tar.gz \
    && apt-get remove --purge -y wget build-essential \
    && apt-get clean


RUN sqlite3 --version

WORKDIR /code

COPY . /code/

COPY ./requirements.txt ./

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["bash", "bin/start.sh"]
