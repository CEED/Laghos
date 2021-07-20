#!/bin/bash

while [ "$?" -eq 0 ]
do
    if [[ -f "$SCRIPT_DIR/run/hyperreduce.txt" ]]; then
        $LAGHOS_SERIAL "$@"
        rm -rf ${BASELINE_LAGHOS_DIR}/run/${OUTPUT_DIR}/hyperreduce.txt
        rm -rf ${BASE_DIR}/run/${OUTPUT_DIR}/hyperreduce.txt
    else
        $LAGHOS "$@"
    fi
done
