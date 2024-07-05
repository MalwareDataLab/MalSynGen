#!/bin/bash
sudo docker run -it --name=syntabdata-$RANDOM -e DISPLAY=unix$DISPLAY sf24/syntabdata:latest bash
#sudo docker run -it --name=droidaumentor-$RANDOM -e DISPLAY=unix$DISPLAY sf23/droidaugmentor:latest python3 /droidaugmentor/setup/main.py --xx --yy --zz --kk ls ls ls 
