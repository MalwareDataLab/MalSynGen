#!/bin/sh
if  ! command -v mlflow > /dev/null 2>&1;then
	pip install mlflow
fi
if [ $# -eq 0 ]; then
    echo "Você  deve fornecer a porta para o servidor mlflow"
    exit 1
fi
PORT=$1
echo "Iniciando o servidor mlflow na porta $PORT..."
mlflow server --host 127.0.0.1 --port $PORT
