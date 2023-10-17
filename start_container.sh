#!/bin/bash

if [ $# -eq 0 ]; then

    echo "No container name provided"

    echo "Usage: $0 <containername>"

    exit 1

fi

CONTAINER_NAME=$1

echo "Start Building $CONTAINER_NAME ..."

docker compose -f docker-compose.yml -p $CONTAINER_NAME down

echo $2

if [ $# -gt 1 ]; then
    if [ $2 = "-b" ] || [ $2 = "--build" ]; then
        docker compose -f docker-compose.yml -p $CONTAINER_NAME up --build
    fi
else
    docker compose -f docker-compose.yml -p $CONTAINER_NAME up
fi 


show_help() {
    echo -e "Usage: $0 <containername> -b, or $0 <containername> --build" \
        "\n\t Rebuild the docker images" \
        "\n-h, --help" \
        "\n\t Display this help and exit"
}
