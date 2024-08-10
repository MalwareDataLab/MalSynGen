#!/bin/bash
if docker info >/dev/null 2>&1; then
	docker build -t sf24/malsyngen:latest . 
else
    echo "Seu usuário atual não possui permissões para executar docker sem sudo, execute o seguinte comando e reinicialize a máquina: sudo usermod -aG docker SEU_USER "
fi
