# Sourced by a launcher under KT_PREFILL_LEND=1 (patches/kt_lend.py): picks the
# prefill chunk from what the server measured, calibrating once when there is
# nothing to go on. Flash-Next's launcher has its own copy of the same steps.
#
#   kt_lend_auto <launcher> <chunk env var> <port> <key part>...
#
# Sets KT_PREFILL_LEND_HINT (prefix of the per-rank hint files the server
# writes, keyed by the key parts: model, context, mem fraction, GPUs, hot
# experts, ...) and KT_LEND_CHUNK:
#   - the chunk env var when the user gave one;
#   - else the smaller of the ranks' hints from an earlier launch;
#   - else a calibration: the launcher started once in the background with
#     KT_PREFILL_LEND_CALIBRATING=1 at a first guess from the card size, one
#     ~50K-token prompt, the server stopped; out of memory halves the guess
#     (down to 2048). KT_PREFILL_LEND_CALIBRATE=0 skips it (chunk 8192).
# The launcher must print "Run directory: <dir>" (or "log directory: <dir>")
# and leave <dir>/server.pid
# and <dir>/server.log in --background mode.

_kt_lend_read_hint() {
  python3 - "$KT_PREFILL_LEND_HINT" <<'PY'
import glob, json, sys
hints = [json.load(open(f))["chunk"] for f in sorted(glob.glob(sys.argv[1] + "-rank*.json"))]
print(min(hints) if len(hints) >= 2 else "")
PY
}

kt_lend_auto() {
  local launcher="$1" chunk_var="$2" port="$3"
  shift 3
  local key
  key="$(printf '%s|' "$@" | sha1sum | cut -c1-16)"
  mkdir -p "$HOME/.cache/kt-lend"
  export KT_PREFILL_LEND_HINT="$HOME/.cache/kt-lend/chunk-$key"
  local user="${!chunk_var:-}"
  if [[ -n "$user" ]]; then
    KT_LEND_CHUNK="$user"
    return 0
  fi
  local hint
  hint="$(_kt_lend_read_hint)"
  if [[ -n "$hint" ]]; then
    echo "chunk $hint from the hint $KT_PREFILL_LEND_HINT-rank*.json" >&2
    KT_LEND_CHUNK="$hint"
    return 0
  fi
  if [[ "${KT_PREFILL_LEND_CALIBRATE:-1}" != 1 || -n "${KT_PREFILL_LEND_CALIBRATING:-}" ]]; then
    KT_LEND_CHUNK=8192
    return 0
  fi
  local cal_log="$HOME/.cache/kt-lend/calibrate-$key.log"
  : >"$cal_log"
  local vram_mb cal_chunk cal_run cal_pid cal_ok
  vram_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)"
  cal_chunk=16384
  (( vram_mb < 15000 )) && cal_chunk=8192
  (( vram_mb > 23000 )) && cal_chunk=24576
  while (( cal_chunk >= 2048 )); do
    echo "no chunk hint for this setup yet: calibration launch at chunk $cal_chunk (${vram_mb} MiB cards; $cal_log)" >&2
    env KT_PREFILL_LEND_CALIBRATING=1 "$chunk_var=$cal_chunk" bash "$launcher" --background >>"$cal_log" 2>&1 || {
      echo "calibration launch failed to start; see $cal_log" >&2; return 5; }
    cal_run="$(sed -n -e 's/^Run directory: //p' -e 's/^log directory: //p' "$cal_log" | tail -1)"
    cal_pid="$(cat "$cal_run/server.pid" 2>/dev/null)"
    [[ -n "$cal_pid" ]] || { echo "calibration launch left no pid; see $cal_log" >&2; return 5; }
    cal_ok=0
    for _ in $(seq 1 360); do
      kill -0 "$cal_pid" 2>/dev/null || break
      if curl -sf -m 3 "http://127.0.0.1:$port/health" >/dev/null; then cal_ok=1; break; fi
      sleep 10
    done
    if [[ "$cal_ok" == 1 ]]; then
      python3 - "$port" <<'PY' >>"$cal_log" 2>&1 || true
import json, secrets, sys, time, urllib.request
para = ("灯りの落ちた編集部で、彼女は原稿の束を繰っていた。窓の外は雨で、アスファルトの匂いが"
        "換気口から上がってくる。赤を入れる手は速い。迷った箇所には印だけ置いて先へ進む。\n")
text = "識別子 " + secrets.token_hex(24) + "\n" + para * 900 + "\n以上を一行で要約して。"
body = json.dumps({"text": text, "sampling_params": {"max_new_tokens": 4, "temperature": 0.0}}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/generate", body, {"Content-Type": "application/json"})
t = time.time()
r = json.load(urllib.request.urlopen(req, timeout=3600))
print(f"calibration prompt: {r['meta_info']['prompt_tokens']} tokens in {time.time() - t:.1f} s")
PY
      for _ in $(seq 1 60); do
        [[ "$(ls "$KT_PREFILL_LEND_HINT"-rank*.json 2>/dev/null | wc -l)" -ge 2 ]] && break
        sleep 2
      done
    fi
    kill -TERM "$cal_pid" 2>/dev/null
    for _ in $(seq 1 120); do kill -0 "$cal_pid" 2>/dev/null || break; sleep 2; done
    kill -KILL "$cal_pid" 2>/dev/null || true
    # the GPUs, the port and the server processes, as the launchers check them
    for _ in $(seq 1 90); do
      nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || \
        ss -ltn | grep -q ":$port " || pgrep -f 'sglang[. ](serve|launch_server)' >/dev/null || \
        pgrep '^sglang::' >/dev/null || break
      sleep 5
    done
    hint="$(_kt_lend_read_hint)"
    [[ -n "$hint" ]] && break
    if grep -q -E "OutOfMemoryError|out of memory|cannot come back" "$cal_run/server.log" 2>/dev/null; then
      echo "calibration at chunk $cal_chunk ran out of memory; halving" >&2
      cal_chunk=$((cal_chunk / 2))
    else
      break
    fi
  done
  if [[ -n "$hint" ]]; then
    echo "calibrated: chunk $hint (calibration ran at $cal_chunk)" >&2
    KT_LEND_CHUNK="$hint"
  else
    echo "calibration left no hint; starting at chunk 2048 (see $cal_log)" >&2
    KT_LEND_CHUNK=2048
  fi
}

# The DeepSeek V4 / MiMo SWA pool is full_tokens x ratio and must hold one
# chunk plus a page, or the server refuses to start: the ratio for a chunk,
# never below the launcher's own.
kt_lend_swa_ratio() {  # chunk max_total_tokens ratio
  python3 -c "import sys; c, m, r = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]); \
print(f'{max(r, (c + 512) * 1.02 / m):.4f}')" "$1" "$2" "$3"
}
