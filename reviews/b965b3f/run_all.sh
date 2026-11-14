#!/bin/bash
# all physical runs, sequential, under the shared lock
Q=/home/agent/.claude/jobs/a4b4be7c/tmp/rr7/probes
E=/home/agent/.claude/jobs/a4b4be7c/tmp/rr7/env
S=/workspace/julia/Simulations
export SIM_BACKEND=cpu SIM_PROGRESS=600
run() { # env save tool config [extra env...]
  local env=$1 save=$2 tool=$3 cfg=$4; shift 4
  echo "=== $cfg ($save) $*" ; date
  env SIM_T=Float64 SIM_SAVE=$save "$@" timeout 90m julia --project=$env -t 6 $tool $cfg > $Q/logs/$cfg.$(basename $save).log 2>&1
  grep -E "^(OK|FAIL)" $Q/logs/$cfg.$(basename $save).log | cut -c1-230
}
X=$Q/simroot/tools/run_sim.jl; T=$S/tools/run_sim.jl
run $E $Q/data_x $X x_disc_comoving_2d
run $E $Q/data_x $X x_disc_comoving_spin_2d
run $E $Q/data_x $X x_disc_mach2_2d SIM_VARY="Grid_Ns=[(100,100),(200,200),(400,400)]"
run $E $Q/data_x32 $X x_disc_mach2_2d SIM_T=Float32 SIM_VARY="Grid_Ns=[(100,100),(200,200)]"
run $E $Q/data_x32 $X x_disc_comoving_spin_2d SIM_T=Float32
run $E $Q/data $T oscillating_piston_1d
run $E $Q/data $T annulus_vortex_2d SIM_VARY="Grid_Ns=[(50,50),(100,100),(200,200)]"
run $E $Q/data $T piston_channel_2d
run $E $Q/data $T disc_rotating_2d
run $E $Q/data $T disc_static_2d
run $E $Q/data $T disc_moving_2d
run $E $Q/data $T ball_static_3d
run $E $Q/data $T ball_moving_3d
echo ALLDONE; date
