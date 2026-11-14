#!/bin/bash
Q=/home/agent/.claude/jobs/a4b4be7c/tmp/rr7/probes
export SIM_BACKEND=cpu SIM_PROGRESS=600 SIM_T=Float64
echo "=== X5 parent 200,400"
SIM_SAVE=$Q/data_xparent SIM_VARY="Grid_Ns=[(200,200),(400,400)]" timeout 60m julia --project=/home/agent/.claude/jobs/a4b4be7c/tmp/rr6/probes/penv -t 6 $Q/simroot/tools/run_sim.jl x_disc_mach2_2d > $Q/logs/x5_parent.log 2>&1
grep -E "^(OK|FAIL)" $Q/logs/x5_parent.log | cut -c1-230
echo X5DONE
