#!/bin/bash
#USER_ID=$(id -u $USER)
if docker info >/dev/null 2>&1; then
        DIR=$(readlink -f campanhas_SF24)
        if [ -z "$(docker images -q sf24/malsyngen:latest 2> /dev/null)" ]; then
        docker build -t sf24/malsyngen:latest . 
        fi
        echo $DIR
        docker run -it --name=MalSynGen-$RANDOM -v $DIR:/MalSynGen/campanhas_SF24 -e DISPLAY=unix$DISPLAY sf24/malsyngen:latest /MalSynGen/run_reproduce_sf24_venv.sh
        #ls shared/outputs/
else
    echo "Seu usuário atual não possui permissões para executar docker sem sudo, execute o seguinte comando e reinicialize a máquina: sudo usermod -aG docker SEU_USER "
fi
if command -v jupyter &> /dev/null
then
    jupyter notebook plots.ipynb
else
   pip install notebook
   jupyter notebook plots.ipynb
fi







