FROM ubuntu:22.04


WORKDIR /MalSynGen


RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    unzip \
    && rm -rf /var/lib/apt/lists/*


RUN rm /bin/sh && ln -s /bin/bash /bin/sh


RUN pip3 install pipenv
RUN pip install urllib3

COPY ./ /MalSynGen/

# Install dependencies using pipenv and requirements.txt

RUN pip3 install -r requirements.txt

RUN chmod +x /MalSynGen/shared/app_run.sh
CMD ["/MalSynGen/shared/app_run.sh"]

