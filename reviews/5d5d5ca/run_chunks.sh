#!/bin/bash
cd /tmp/brev3/rr5/probes
W=/workspace/julia/HyperCloudDev/HyperCloudCore.jl/.claude/worktrees/rr-moving3
J="julia -t 2 --project=$W/test"
export VARIANT=chunks
{ $J t1_insert_onesweep_chunks.jl; NS=20,40,80 $J o2_periodic_chunks.jl; $J o3_plate_variants_chunks.jl; CASE=r8 NS=20,40,80,160 $J o1_variants_chunks.jl; CASE=p14 NS=20,40,80,160 $J o1_variants_chunks.jl; CASE=r8c NS=40,80,160 $J o1_variants_chunks.jl; } 2>&1 | grep -v "Warning\|└\|┌" > logs/chunks.v3.log
