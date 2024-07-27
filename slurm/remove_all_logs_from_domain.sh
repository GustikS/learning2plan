#!/bin/bash

domain=$1

if [ -z $domain ]; then
    echo "Usage: $0 <domain>"
    exit 1
fi

timestamp=$(date +%Y%m%d-%H%M%S)

for dir in test_logs; do
# for dir in baseline_logs test_logs train_logs models; do
    mkdir -p backups/$timestamp/$dir
    mv __experiments/${dir}/*${domain}* backups/$timestamp/$dir
done
