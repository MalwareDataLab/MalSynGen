FROM  ubuntu:22.04
USER root
WORKDIR /MalSynGen
RUN apt-get update
RUN apt-get -y install python3-pip
RUN apt install unzip 
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
COPY ./ /MalSynGen/
RUN pip3 install pipenv
RUN pipenv lock
RUN pipenv install -r requirements.txt
#CMD while true; do sleep 1000; done

RUN mv /MalSynGen/scripts/run_app_in_docker.sh /usr/bin/docker_run.sh
RUN chmod +x /usr/bin/docker_run.sh 
CMD ["docker_run.sh"]



