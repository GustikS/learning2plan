#!/bin/bash

reset_time=600
script=submit_test.py

while true
do
    num_queued=$(qstat | grep "Q " | wc -l)
    num_running=$(qstat | grep "R " | wc -l)
    echo "$(date) Q=${num_queued} R=${num_running}" | tee -a job_info.log
    to_submit=$((1000 - ${num_queued}))
    python3 $script $to_submit
    date
    python3 ~/num_jobs_info.py; nci_account -P cd85
    echo " ... submitting jobs again in ${reset_time} seconds ... "
    sleep $reset_time
done
