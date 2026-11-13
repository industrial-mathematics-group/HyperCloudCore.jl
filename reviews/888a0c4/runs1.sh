#!/bin/bash
# Selected physical tests on the review code (CPU Float64), one at a time under the shared lock
P=/home/agent/.claude/jobs/a4b4be7c/tmp/rr6/probes
ENV=/home/agent/.claude/jobs/a4b4be7c/tmp/rr6/env
S=/workspace/julia/Simulations
run() {  # name [vary]
  echo "=== $1 ($2) $(date +%T)"
  if [ -n "$2" ]; then export SIM_VARY="$2"; else unset SIM_VARY; fi
  ( cd $P && SIM_BACKEND=cpu SIM_T=Float64 SIM_SAVE=$P/data SIM_PROGRESS=600 \
      flock /tmp/hcc_test.lock timeout 90m julia --project=$ENV -t 6 $S/tools/run_sim.jl $1 > $P/logs/$1.log 2>&1 )
  echo "exit $? $(date +%T)"; grep -E '^(OK|FAIL)' $P/logs/$1.log
}
for a in "$@"; do run ${a%%|*} "$( [[ $a == *"|"* ]] && echo "${a#*|}")"; done
