#!/bin/bash
sudo docker run -it --name=syn_tab_data-$RANDOM -e DISPLAY=unix$DISPLAY sf24/syn_tab_data:latest
