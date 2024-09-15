
# Use Python base image
FROM python:3.10.12-bullseye

# Install necessary packages and update SQLite to version >= 3.35.0
RUN apt-get update && apt-get install -y wget build-essential libsqlite3-dev \
    && wget https://www.sqlite.org/2023/sqlite-autoconf-3430100.tar.gz \
    && tar -xzf sqlite-autoconf-3430100.tar.gz \
    && cd sqlite-autoconf-3430100 \
    && ./configure --prefix=/usr/local \
    && make && make install \
    && cd .. && rm -rf sqlite-autoconf-3430100 sqlite-autoconf-3430100.tar.gz \
    && apt-get remove --purge -y wget build-essential \
    && apt-get clean

# Verify the correct SQLite version
RUN sqlite3 --version

# Set working directory
WORKDIR /code

# Copy the application code
COPY . /code/

# Copy requirements file
COPY ./requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Start the application
CMD ["bash", "bin/start.sh"]
