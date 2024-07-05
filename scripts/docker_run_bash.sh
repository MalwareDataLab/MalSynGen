#!/bin/bash
sudo docker run -it --name=syntabdata-$RANDOM -e DISPLAY=unix$DISPLAY sf24/syntabdata:latest bash
