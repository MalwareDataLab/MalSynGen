#!/bin/bash

echo "=============================================================="
echo "Running app with parameters: $*"
echo "=============================================================="
#USER_ID=$1
#shift
cd /MalSynGen/
python3 main.py  $*
#chown -R $USER_ID shared 
