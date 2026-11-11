cd /tmp/brev3/rr2/probes
for t in /tmp/brev3/r1/old /tmp/brev3/rr2/trees/new /tmp/brev3/rr2/trees/mask /tmp/brev3/rr2/trees/new /tmp/brev3/rr2/trees/mask /tmp/brev3/r1/old; do
  julia -t 8 --project=$t/test cost.jl 2>&1 | tail -6
done
