#!/bin/bash

# Usage: rundock.sh <dir-containing-docker-file> ["int"]

# User's params
IMG_TAG="csl"
HOST_DIR_NAME="out"   # only name, no dir
VIRTUAL_DIR="/home/guest/out"
# end of user's params

set -e

# process params
context=$(readlink -f -- "$1")
host_dir="${context}/$HOST_DIR_NAME"
interactive_mode=0
if [[ $2 = "int" ]]; then interactive_mode=1; fi

# build
echo "---------------------------------"
echo "Building with context: ""$context"
echo "---------------------------------"
docker build -t ${IMG_TAG} "$context"

# run
echo "---------------------------------"
echo "Binding: ${host_dir} and ${VIRTUAL_DIR}"
echo "---------------------------------"

if [[ $interactive_mode = 1 ]]; then
  docker run -v "$host_dir":$VIRTUAL_DIR -it ${IMG_TAG} /bin/bash
else
  docker run -v "$host_dir":$VIRTUAL_DIR -it ${IMG_TAG} python py/main.py
fi
