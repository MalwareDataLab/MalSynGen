#!/bin/bash
if docker info >/dev/null 2>&1; then
	for i in $( docker ps -a | awk '{print $1}' | grep -v CONTAINER)
	do 
		 docker rm -f $i
	done
else
    echo "Seu usuário atual não possui permissões para executar docker sem sudo, execute o seguinte comando e reinicialize a máquina: sudo usermod -aG docker SEU_USER "
fi

