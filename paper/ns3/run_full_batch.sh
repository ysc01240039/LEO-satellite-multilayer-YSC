#!/bin/bash
# Full ns-3 batch: fig6 (3-gamma CBDP validation), fig_cmp (protocol comparison),
# fig8 (failure recovery). Simple concurrency pool.
# Logs are written to the Windows drive (logs_eval/) so they survive WSL restarts.
cd ~/ns-3-dev
BIN=./build/scratch/ns3-dev-leo_cbdp_eval-default
OUT=/mnt/e/pytorchFile/YSC_2/paper/ns3/logs_eval
mkdir -p $OUT
rm -f $OUT/*.log $OUT/progress.log $OUT/jobs.txt
POOL=${POOL:-10}

run_job() {
  local tag=$1; shift
  echo "START $tag $(date +%H:%M:%S)" >> $OUT/progress.log
  timeout 9000 $BIN "$@" > $OUT/$tag.log 2>&1
  local rc=$?
  echo "END $tag rc=$rc $(date +%H:%M:%S)" >> $OUT/progress.log
}

# job list: tag|args
JOBS=$(cat <<'EOF'
EOF
)

# ---------- Batch A: fig6, N=1000, 3 gamma x 3 seeds, CBDP + Dijkstra ref ----
# nCores from C++ PDE scan (beta=0.6, kappa=0.01), matched per seed:
#   g0.8: s42=75 s123=92 s456=68 ; g1.0: s42=57 s123=51 s456=72 ; g1.5: s42=199 s123=154 s456=176
declare -A NC
NC[0.8,42]=75;  NC[0.8,123]=92;  NC[0.8,456]=68
NC[1.0,42]=57;  NC[1.0,123]=51;  NC[1.0,456]=72
NC[1.5,42]=199; NC[1.5,123]=154; NC[1.5,456]=176

for g in 0.8 1.0 1.5; do
  for s in 42 123 456; do
    nc=${NC[$g,$s]}
    echo "fig6_g${g}_s${s}|--mode=compare --protocol=CBDP --nSats=1000 --nCores=$nc --simTime=50 --flowStart=30 --seed=$s" >> $OUT/jobs.txt
  done
done
for s in 42 123 456; do
  echo "fig6_dij_s${s}|--mode=compare --protocol=Dijkstra --nSats=1000 --nCores=93 --simTime=50 --flowStart=30 --seed=$s" >> $OUT/jobs.txt
done

# ---------- Batch C: fig8 failure recovery (run early: long poles) ----------
# N=1000, gamma=1.0, nCores=57, seed 42, fail at t=40, detect+reconfig 3s
for ff in 0.01 0.02 0.05 0.10; do
  echo "fig8_ff${ff}|--mode=failure --protocol=CBDP --nSats=1000 --nCores=57 --simTime=60 --flowStart=10 --failTime=40 --detectDelay=3 --failFrac=$ff --seed=42" >> $OUT/jobs.txt
done

# ---------- Batch B: protocol comparison, N=200/400/600 --------------------
# nCores from C++ PDE N-scan (gamma=6.0, beta=0.6): 200->137, 400->117, 600->108
declare -A NCN
NCN[200]=137; NCN[400]=117; NCN[600]=108
for N in 200 400 600; do
  for p in Dijkstra CBDP OLSR AODV; do
    for s in 42 123 456; do
      echo "cmp_N${N}_${p}_s${s}|--mode=compare --protocol=$p --nSats=$N --nCores=${NCN[$N]} --simTime=50 --flowStart=30 --seed=$s" >> $OUT/jobs.txt
    done
  done
done

total=$(wc -l < $OUT/jobs.txt)
echo "TOTAL JOBS: $total, pool=$POOL" | tee -a $OUT/progress.log

# ---------- simple pool ----------
i=0
while IFS='|' read -r tag args; do
  [ -z "$tag" ] && continue
  i=$((i+1))
  while [ "$(jobs -rp | wc -l)" -ge "$POOL" ]; do sleep 5; done
  run_job "$tag" $args &
  echo "QUEUED $i/$total $tag" >> $OUT/progress.log
done < $OUT/jobs.txt
wait
echo "ALL DONE $(date +%H:%M:%S)" | tee -a $OUT/progress.log
grep -h RESULT $OUT/*.log > $OUT/all_results.txt
echo "results collected: $(wc -l < $OUT/all_results.txt)"
