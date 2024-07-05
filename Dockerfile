FROM  ubuntu:20.04
USER root
WORKDIR /SynTabData
RUN apt-get update
RUN apt-get -y install python3-pip
RUN apt install unzip 
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
#RUN pip3 install pipenv==2023.7.23
#RUN export PATH="/home/root/.local/bin:$PATH" 
COPY ./ /SynTabData/
RUN apt install python3.8-venv
RUN python3 -m pip install --user virtualenv
RUN apt-get install python3-venv
RUN python3 -m venv .venv
RUN  source .venv/bin/activate
RUN pip3 install -r requirements.txt
#CMD while true; do sleep 1000; done
