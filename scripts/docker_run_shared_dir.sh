#!/bin/bash
[ $1 ] && [ -d $1 ] || {
	echo "Usage: $0 <code_directory>"
	echo " example: $0 ."
	exit
}
sudo docker run -it --name=syntabdata-$RANDOM -v $(readlink -f $1):/droidaugmentor/shared -e DISPLAY=unix$DISPLAY sf24/syntabdata:latest bash 
