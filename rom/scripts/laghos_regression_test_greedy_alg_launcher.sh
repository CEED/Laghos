#!/bin/bash

while [ "$?" -eq 0 ]
do
    if [[ -f "run/${OUTPUT_DIR}/hyperreduce.txt" ]]; then
        $LAGHOS_SERIAL "$@"
        rm -rf run/${OUTPUT_DIR}/hyperreduce.txt
    else
        $LAGHOS "$@"
    fi
done
exit 0
