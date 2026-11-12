#!/bin/bash
# run_on.sh <tag: v3|v2|prev|parent> probe.jl ...
case $1 in v3) W=rr-moving3;; v2) W=rr-moving2;; prev) W=rr-moving;; parent) W=review-mb;; esac
W=/workspace/julia/HyperCloudDev/HyperCloudCore.jl/.claude/worktrees/$W
tag=$1; shift
cd /tmp/brev3/rr5/probes
for p in "$@"; do
  n=${p%.jl}; s=$(date +%s)
  timeout 40m julia -t 2 --project=$W/test $p > logs/$n.$tag.log 2>&1
  echo "$n [$tag] exit=$? $(( $(date +%s)-s ))s" >> logs/_runs.txt
done
