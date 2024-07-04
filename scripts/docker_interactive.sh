#!/bin/bash
docker build -f Dockerfile_novo -t droidaug
sudo docker run -it droidaug /bin/bash
