#!/bin/bash
if docker info >/dev/null 2>&1; then
	if [ -z "$(docker images -q sf24/malsyngen:latest 2> /dev/null)" ]; then
	 ./scripts/docker_build.sh
	fi

	docker run -it --name=MalSynGen-$RANDOM -e DISPLAY=unix$DISPLAY sf24/malsyngen:latest
else
    echo "Seu usuário atual não possui permissões para executar docker sem sudo, execute o seguinte comando e reinicialize a máquina: sudo usermod -aG docker SEU_USER "
fi
