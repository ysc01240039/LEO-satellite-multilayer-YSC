#!/bin/bash
# Run routing comparison across scales, protocols, and multiple seeds
cd /home/mark/ns-3-dev
B=./build/scratch/ns3-dev-leo_route_compare-default
OUT=/home/mark/ns-3-dev/compare_results.txt
: > "$OUT"
for N in 80 160 320; do
  for p in Dijkstra OLSR AODV CBDP; do
    for seed in 42 7 123 2024; do
      echo "N=$N proto=$p seed=$seed" >> "$OUT"
      timeout 300 "$B" --nSats=$N --protocol=$p --seed=$seed --simTime=30 --flowStart=5 2>&1 | grep -E "RESULT" >> "$OUT"
    done
  done
done
echo "DONE" >> "$OUT"
cat "$OUT"