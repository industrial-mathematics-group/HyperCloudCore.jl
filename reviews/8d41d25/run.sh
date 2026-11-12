#!/bin/bash
W=/workspace/julia/HyperCloudDev/HyperCloudCore.jl/.claude/worktrees/rr-moving2
cd /tmp/brev3/rr4/probes
for p in "$@"; do
  n=${p%.jl}
  s=$(date +%s)
  timeout 40m julia -t 2 --project=$W/test $p > logs/$n.log 2>&1
  echo "$n exit=$? $(( $(date +%s)-s ))s" >> logs/_runs.txt
done
