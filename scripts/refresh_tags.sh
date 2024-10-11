#!/usr/bin/env bash 


export SRC="/content/llm-meg"
python $SRC/scripts/ptags.py $(find /content/llm-meg -iname '*.py.')
echo "refresh done."
