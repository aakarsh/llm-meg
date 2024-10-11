#!/usr/bin/env bash


export SRC="/content/llm-meg"
python $SRC/scripts/ptags.py $(find ${SRC} -iname '*.py.')
echo "refresh done."
