#!/bin/bash
[ $1 ] && [ -d $1 ] || {
	echo "Usage: $0 <code_directory>"
	echo " example: $0 ."
	exit
}
if docker info >/dev/null 2>&1; then
	docker run -it --name=MalSynGen-$RANDOM -v $(readlink -f $1):/MalSynGen/shared -e DISPLAY=unix$DISPLAY sf24/malsyngen:latest bash 
else
    echo "Seu usuário atual não possui permissões para executar docker sem sudo, execute o seguinte comando e reinicialize a máquina: sudo usermod -aG docker SEU_USER "
fi
