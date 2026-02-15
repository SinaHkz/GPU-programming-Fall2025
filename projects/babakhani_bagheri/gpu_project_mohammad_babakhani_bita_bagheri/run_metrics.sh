#!/usr/bin/env bash
set -euo pipefail

BIN="./build/nbody"
OUT="perf_runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"/{logs,nsys,ncu,extract}

export __GL_SYNC_TO_VBLANK=0

# Nها تا 32768 (اگر 32768 خیلی سنگین بود، حذفش کن)
Ns=(1024 2048 4096 8192 16384 32768)

STEPS_BENCH=200       # برای چاپ time_per_step_ms
STEPS_NSYS=2000       # برای capture درست timeline
STEPS_NCU=8           # سریع برای ncu

# برای سناریوهای رندر بهتره N کمتر باشه
Ns_present=(4096 8192 16384)

# ---------- helpers ----------
run_cmd() {
  local name="$1"; shift
  echo "=== Running: $name ==="
  echo "$@" | tee "$OUT/logs/${name}.cmd.txt"
  ( "$@" ) 2>&1 | tee "$OUT/logs/${name}.log.txt"
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# ---------- env baseline ----------
run_cmd "env" bash -lc "nvidia-smi && uname -a && $BIN --help | head -n 80"

# ---------- CSV header ----------
SUMMARY_CSV="$OUT/extract/metrics_summary.csv"
echo "scenario,N,steps,dt,eps,blockSize,time_per_step_ms,IPS,nsys_gpu_time_ms,nsys_top_kernel,nsys_top_kernel_ms,nsys_memcpy_ms,ncu_sm_pct,ncu_dram_pct,ncu_branch_eff" > "$SUMMARY_CSV"

# ---------- function: parse benchmark table from log ----------
# expects lines like: N | dt | eps | blockSize | time_per_step_ms | IPS
parse_bench_row() {
  local logfile="$1"
  # خروجی: N,dt,eps,blockSize,time,IPS
  awk -F'|' '
    $1 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/ && NF>=6 {
      gsub(/[[:space:]]/,"",$1);
      gsub(/[[:space:]]/,"",$2);
      gsub(/[[:space:]]/,"",$3);
      gsub(/[[:space:]]/,"",$4);
      gsub(/[[:space:]]/,"",$5);
      gsub(/[[:space:]]/,"",$6);
      print $1","$2","$3","$4","$5","$6
    }' "$logfile"
}

# ---------- function: extract nsys stats summary ----------
# creates a small text summary we can parse
extract_nsys_stats() {
  local rep="$1"
  local outtxt="$2"

  # روش 1: اگر nsys stats خودش کار کرد
  if have_cmd nsys; then
    # بعضی نسخه‌ها می‌خوان مسیر rep بدون پسوند یا با پسوند باشد
    # ما هر دو حالت را امتحان می‌کنیم
    (nsys stats --report summary,gpu-kern-summary,gpu-mem-summary "$rep" > "$outtxt" 2>&1) || \
    (nsys stats "$rep" > "$outtxt" 2>&1) || true
  fi
}

# ---------- function: parse nsys stats text for a few numbers ----------
# best effort: parse total gpu time, top kernel name/time, memcpy time
parse_nsys_stats_text() {
  local statstxt="$1"
  local gpu_time_ms="" topk="" topk_ms="" memcpy_ms=""

  # این پارس "best effort" است چون خروجی nsys بین نسخه‌ها فرق می‌کند
  # 1) GPU Kernels summary: pick first kernel row
  topk="$(grep -E '^[[:space:]]*[0-9]+[[:space:]]+[^[:space:]].*compute_accel_tiled|^[[:space:]]*[0-9]+[[:space:]]+[^[:space:]]' "$statstxt" | head -n 1 | sed -E 's/^[[:space:]]*//')"
  # تلاش برای گرفتن زمان کرنل اول (عدد ms یا us) از همان خط
  topk_ms="$(echo "$topk" | grep -Eo '([0-9]+\.[0-9]+|[0-9]+)[[:space:]]*(ms|us|s)' | head -n 1)"

  # 2) memcpy totals
  memcpy_ms="$(grep -iE 'Memcpy|memcpy|HtoD|DtoH|cudaMemcpy' "$statstxt" | grep -Eo '([0-9]+\.[0-9]+|[0-9]+)[[:space:]]*(ms|us|s)' | head -n 1)"

  # 3) total gpu time (اگر در summary باشد)
  gpu_time_ms="$(grep -iE 'Total GPU time|GPU time' "$statstxt" | grep -Eo '([0-9]+\.[0-9]+|[0-9]+)[[:space:]]*(ms|us|s)' | head -n 1)"

  echo "$gpu_time_ms|$topk|$topk_ms|$memcpy_ms"
}

# ---------- function: import ncu rep to csv ----------
import_ncu_csv() {
  local rep="$1"
  local outcsv="$2"
  if have_cmd ncu; then
    # بعضی نسخه‌ها --import را دارند
    (ncu --import "$rep" --csv > "$outcsv" 2>/dev/null) || \
    (sudo ncu --import "$rep" --csv > "$outcsv" 2>/dev/null) || true
  fi
}


# ---------- function: parse our 3 metrics from ncu csv ----------
parse_ncu_metrics() {
  local csv="$1"
  local sm="" dram="" br=""
  # نcu csv فرمتش بزرگه؛ ما فقط مقدار آخر هر metric رو می‌گیریم
  sm="$(grep -F 'sm__throughput.avg.pct_of_peak_sustained_elapsed' "$csv" | tail -n 1 | awk -F',' '{print $NF}')"
  dram="$(grep -F 'dram__throughput.avg.pct_of_peak_sustained_elapsed' "$csv" | tail -n 1 | awk -F',' '{print $NF}')"
  br="$(grep -F 'smsp__branch_efficiency.avg.pct' "$csv" | tail -n 1 | awk -F',' '{print $NF}')"
  echo "$sm|$dram|$br"
}

# =========================================================
# SCENARIOS
# =========================================================

run_scenario_bench() {
  local scenario="$1"
  shift
  local extra_args=("$@")

  for N in "${Ns[@]}"; do
    local name="${scenario}_bench_N${N}"
    run_cmd "$name" bash -lc "$BIN ${extra_args[*]} --render 0 --benchmark 0 --steps $STEPS_BENCH --N $N"
    # parse benchmark row(s)
    parse_bench_row "$OUT/logs/${name}.log.txt" > "$OUT/extract/${name}.bench.csv" || true

    # append to summary (bench only, no nsys/ncu numbers yet)
    while IFS= read -r row; do
      # row: N,dt,eps,blockSize,time,IPS
      echo "${scenario},${row%,*},$STEPS_BENCH,,,,,,,,,," >> "$SUMMARY_CSV"
    done < "$OUT/extract/${name}.bench.csv"
  done
}

run_scenario_nsys() {
  local scenario="$1"
  shift
  local extra_args=("$@")

  for N in 8192 16384 32768; do
    local repbase="$OUT/nsys/${scenario}_N${N}"
    local name="nsys_${scenario}_N${N}"

    run_cmd "$name" bash -lc "nsys profile --stats=true --force-overwrite=true -o $repbase \
      $BIN ${extra_args[*]} --render 0 --steps $STEPS_NSYS --N $N"

    # find produced rep (nsys may add .nsys-rep)
    local rep="${repbase}.nsys-rep"
    local statstxt="$OUT/extract/${scenario}_N${N}.nsys_stats.txt"
    extract_nsys_stats "$rep" "$statstxt"

    # parse key numbers
    local parsed
    parsed="$(parse_nsys_stats_text "$statstxt")"
    local gpu_time="${parsed%%|*}"; parsed="${parsed#*|}"
    local topk="${parsed%%|*}"; parsed="${parsed#*|}"
    local topk_ms="${parsed%%|*}"; parsed="${parsed#*|}"
    local memcpy_ms="${parsed}"

    # write a tiny csv line for nsys
    echo "scenario=$scenario N=$N gpu_time=$gpu_time topKernel='$topk' topKernelTime=$topk_ms memcpy=$memcpy_ms" \
      > "$OUT/extract/${scenario}_N${N}.nsys_parsed.txt"
  done
}

run_scenario_present_nsys() {
  local scenario="present"
  for N in "${Ns_present[@]}"; do
    local repbase="$OUT/nsys/${scenario}_N${N}"
    local name="nsys_${scenario}_N${N}"

    run_cmd "$name" bash -lc "nsys profile --stats=true --force-overwrite=true -o $repbase \
      $BIN --present 1 --steps $STEPS_NSYS --N $N"

    local rep="${repbase}.nsys-rep"
    local statstxt="$OUT/extract/${scenario}_N${N}.nsys_stats.txt"
    extract_nsys_stats "$rep" "$statstxt"

    local parsed
    parsed="$(parse_nsys_stats_text "$statstxt")"
    echo "scenario=$scenario N=$N parsed=$parsed" > "$OUT/extract/${scenario}_N${N}.nsys_parsed.txt"
  done
}

run_ncu_quick() {
  # فقط 2 N کلیدی
  for N in 8192 16384; do
    local rep="$OUT/ncu/accel_quick_N${N}"
    local name="ncu_quick_N${N}"

    run_cmd "$name" bash -lc "sudo ncu \
      --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__branch_efficiency.avg.pct \
      --kernel-name compute_accel_tiled \
      --launch-count 2 \
      -o $rep \
      $BIN --benchmark 1 --steps $STEPS_NCU --collisions 0 --N $N"

    # import to csv
    local repfile="${rep}.ncu-rep"
    local outcsv="$OUT/extract/ncu_quick_N${N}.csv"
    import_ncu_csv "$repfile" "$outcsv"

    # parse metrics
    local metrics
    metrics="$(parse_ncu_metrics "$outcsv")"
    echo "$metrics" > "$OUT/extract/ncu_quick_N${N}.parsed.txt"
  done
}

# ============ RUN ALL ============

# 1) realSolarSystem (collisions off)
run_scenario_bench "realSolarSystem" --realSolarSystem 1 --collisions 0
run_scenario_nsys "realSolarSystem" --realSolarSystem 1 --collisions 0


# 2) solarCollision + fragmentation
run_scenario_bench "solarCollision_frag" --solarCollision 1 --fragmentation 1
run_scenario_nsys "solarCollision_frag" --solarCollision 1 --fragmentation 1

# 3) present (render on) -> فقط nsys
run_scenario_present_nsys

# 4) NCU quick metrics for gravity kernel
run_ncu_quick

# 5) Create a minimal baseline json (best effort)
# اگر jq داری بهتره؛ اگر نداری، یک فایل ساده می‌سازیم
if have_cmd jq; then
  jq -n \
    --arg out "$OUT" \
    --arg date "$(date -Iseconds)" \
    '{
      generated_at: $date,
      out_dir: $out,
      notes: "bench logs + nsys + ncu quick metrics",
      files: {
        summary_csv: ($out + "/extract/metrics_summary.csv"),
        nsys_dir: ($out + "/nsys"),
        ncu_dir: ($out + "/ncu"),
        extract_dir: ($out + "/extract")
      }
    }' > "$OUT/extract/baseline_metrics.json"
else
  cat > "$OUT/extract/baseline_metrics.json" <<EOF
{
  "generated_at": "$(date -Iseconds)",
  "out_dir": "$OUT",
  "files": {
    "summary_csv": "$OUT/extract/metrics_summary.csv",
    "nsys_dir": "$OUT/nsys",
    "ncu_dir": "$OUT/ncu",
    "extract_dir": "$OUT/extract"
  }
}
EOF
fi

echo "DONE."
echo "All outputs saved in: $OUT"
echo "Key files:"
echo " - $OUT/extract/metrics_summary.csv"
echo " - $OUT/extract/baseline_metrics.json"
echo " - $OUT/nsys/*.nsys-rep"
echo " - $OUT/ncu/*.ncu-rep"

