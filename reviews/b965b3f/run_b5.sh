#!/bin/bash
# B5 with the previous state (025b0df/15277eb, rr6 env) and the parent (rr6 penv), to locate the EPD1 v_perp change
Q=/home/agent/.claude/jobs/a4b4be7c/tmp/rr7/probes
export SIM_BACKEND=cpu SIM_PROGRESS=600 SIM_T=Float64
for pair in "/home/agent/.claude/jobs/a4b4be7c/tmp/rr6/env:data_prev" "/home/agent/.claude/jobs/a4b4be7c/tmp/rr6/probes/penv:data_parent"; do
  env=${pair%%:*}; save=$Q/${pair##*:}
  echo "=== piston_channel_2d $save"; date
  SIM_SAVE=$save timeout 60m julia --project=$env -t 6 /workspace/julia/Simulations/tools/run_sim.jl piston_channel_2d > $Q/logs/piston_channel_2d.$(basename $save).log 2>&1
  grep -E "^(OK|FAIL)" $Q/logs/piston_channel_2d.$(basename $save).log | cut -c1-230
done
echo B5DONE
