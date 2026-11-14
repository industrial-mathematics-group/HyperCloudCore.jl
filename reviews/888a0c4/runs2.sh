#!/bin/bash
# usage: runs2.sh <env> <run_sim.jl> <save dir> "name|vary" ...
P=/home/agent/.claude/jobs/a4b4be7c/tmp/rr6/probes
ENV=$1; RS=$2; SAVE=$3; shift 3
for a in "$@"; do
  name=${a%%|*}; vary=""; [[ $a == *"|"* ]] && vary=${a#*|}
  echo "=== $name ($vary) env=$(basename $ENV) $(date +%T)"
  if [ -n "$vary" ]; then export SIM_VARY="$vary"; else unset SIM_VARY; fi
  ( cd $P && SIM_BACKEND=cpu SIM_T=Float64 SIM_SAVE=$SAVE SIM_PROGRESS=600 \
      flock /tmp/hcc_test.lock timeout 90m julia --project=$ENV -t 6 $RS $name > $P/logs/$name.$(basename $ENV).log 2>&1 )
  echo "exit $? $(date +%T)"; grep -E '^(OK|FAIL)' $P/logs/$name.$(basename $ENV).log
done
