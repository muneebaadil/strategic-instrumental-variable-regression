#!/bin/bash

# Usage: rundock.sh <dir-containing-docker-file> ["int"|"nb"|"python"] [<some-python-file>]
# E.g.,
#       bash scripts/rundock.sh . int
#       bash scripts/rundock.sh . nb
#       bash scripts/rundock.sh . python pytest/data_gen_test.py

# User's params
IMG_TAG="csl"
HOST_DIR_NAME="out"   # only name, no dir
VIRTUAL_DIR="/home/guest/out"
# end of user's params

set -e

# process params
context=$(readlink -f -- "$1")
host_dir="${context}/$HOST_DIR_NAME"
launch_mode=$2

# build
echo "---------------------------------"
echo "Building with context: ""$context"
echo "---------------------------------"
docker build -t ${IMG_TAG} "$context"

# run
echo "---------------------------------"
echo "Binding: ${host_dir} and ${VIRTUAL_DIR}"
echo "---------------------------------"

if [[ $launch_mode = "int" ]]; then
  docker run --rm -v "$host_dir":$VIRTUAL_DIR -it ${IMG_TAG} /bin/bash
elif [[ $launch_mode = "nb" ]]; then
  docker run --rm -p 8888:8888 ${IMG_TAG} /bin/bash -c "jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"
elif [[ $launch_mode = "python" ]]; then
  docker run --rm -v "$host_dir":$VIRTUAL_DIR -it ${IMG_TAG} python "$3"
else
  echo "Doing nothing!"
fi
