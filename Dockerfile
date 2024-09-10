FROM python:3.10.12-bullseye

WORKDIR /code

ENV PATH="/code/app:$PATH"

COPY ./requirements.txt ./

RUN apt-get -y update \
    && apt-get -y install --no-install-recommends \
       curl \
    && pip3 install --no-cache --upgrade pip setuptools \
    && pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . .
# Create the .aws directory in the container
# RUN mkdir -p /root/.aws

# Copy the credentials and config files from the local machine to the container
# COPY creds/credentials /root/.aws/credentials

COPY bin/* /code/bin/

CMD ["bash", "bin/start.sh"]