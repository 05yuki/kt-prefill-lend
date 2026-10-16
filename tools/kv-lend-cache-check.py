#!/usr/bin/env python3
"""Does a prefill that borrows the KV pool's free pages (and an SWA pool's) leave
the KV pages in use alone? Against a running server started with
KT_PREFILL_KV_LEND_VERIFY=1 (and KT_PREFILL_KV_LEND_FORCE=1, so every window
lends): the server hashes every page in use when a window lends and again after
it is mapped back, and logs "verify window N: P pages in use checked, C changed".
This tool fills the pool with cached prompts around lending windows:

  1. a few long prompts (cached in the radix tree afterwards), each with a greedy
     continuation recorded;
  2. a long prompt of other tokens, whose prefill lends the free pages while
     those cached ones are in use;
  3. the first prompts again: every greedy continuation must be the same token
     for token as the reference, which was also served off the cache (a miss,
     the prompt evicted, is reported).

Then 2 and 3 once more with the cached prompts taking most of the pool, so the
loan has the least room.

  kv-lend-cache-check.py <tokenizer dir> [--url URL] [--n 4] [--len 24000] [--other 35000] [--new 48]
                        [--fill 3] [--prompt-file FILE]

The judge is the server's log (0 pages changed). The continuations are printed
as well, but they only judge a model whose cached path repeats itself: the
tool checks that first (the cached prompt read twice), and some do not (a
non-deterministic MoE, or a hybrid whose cached linear-attention state differs
between reads). Exit 0 when every judged continuation matched.
"""
import argparse
import json
import random
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("tokenizer")
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--len", type=int, default=24000)
ap.add_argument("--other", type=int, default=35000)
ap.add_argument("--new", type=int, default=48)
ap.add_argument("--fill", type=int, default=0, help="extra cached prompts for the full-pool round (0: none)")
ap.add_argument("--prompt-file", help="text the prompts are cut from (default: random token ids)")
a = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
rng = random.Random(20261016)
if a.prompt_file:
    base = tok.encode(open(a.prompt_file).read())
else:
    base = [rng.randrange(1000, min(50000, len(tok) - 1)) for _ in range(4096)]


def prompt(n: int, salt: int):
    # a distinct head, so no two prompts share a cached prefix
    head = [rng.randrange(1000, 50000) for _ in range(64)] + [salt % 1000 + 1000]
    body = (base * (n // max(1, len(base)) + 2))
    return (head + body)[:n]


SUFFIX = base[:256]


def _post(body):
    req = urllib.request.Request(a.url + "/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=7200))


def gen(ids, new):
    """((greedy continuation ids, logprobs of a 256-token suffix read over the
    prompt), cached tokens of each of the two requests, seconds)."""
    t = time.time()
    r = _post({"input_ids": ids, "sampling_params": {"max_new_tokens": new, "temperature": 0, "ignore_eos": True}})
    out = r.get("output_ids") or tok.encode(r.get("text", ""), add_special_tokens=False)
    c1 = int(r.get("meta_info", {}).get("cached_tokens", 0))
    r = _post({"input_ids": ids + SUFFIX, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
               "return_logprob": True, "logprob_start_len": len(ids)})
    m = r.get("meta_info", {})
    lp = [x[0] for x in m.get("input_token_logprobs", []) if x and x[0] is not None]
    return (list(out), lp), (c1, int(m.get("cached_tokens", 0))), time.time() - t


def diff(x, y):
    (ids, lp), (rids, rlp) = x, y
    first = next((k for k, (u, v) in enumerate(zip(ids, rids)) if u != v), None)
    dlp = max((abs(u - v) for u, v in zip(lp, rlp)), default=0.0)
    return ids == rids and lp == rlp, first, dlp


ok = True
refs = {}  # prompt index -> (fresh, off the cache)


def reference(i, p):
    """The fresh result (the prompt computed whole) and the result off the cache,
    twice: the cached path must repeat itself before anything else is judged."""
    global ok
    fresh, c0, _ = gen(p, a.new)
    cached1, c1, _ = gen(p, a.new)
    cached2, c2, dt = gen(p, a.new)
    same, first, dlp = diff(cached1, cached2)
    f_same, f_first, f_dlp = diff(fresh, cached2)
    refs[i] = (fresh, cached2, c2)
    if not same:
        ok = False
    print(f"reference {i}: {len(p)} tokens; cached {c1} / {c2}: off-cache twice "
          f"{'identical' if same else f'DIFFERS (token {first}, logprob {dlp:.2e})'}; fresh vs off-cache "
          f"{'identical' if f_same else f'differ (token {f_first}, logprob {f_dlp:.2e})'}")


def round_(name, idx):
    global ok
    other = prompt(a.other, 999 + len(name))
    t = time.time()
    _post({"input_ids": other, "sampling_params": {"max_new_tokens": 1, "temperature": 0}})
    print(f"{name}: other prompt {len(other)} tokens prefilled in {time.time() - t:.1f} s "
          "(its window lent around the cached ones)")
    for i in idx:
        out, (c1, c2), dt = gen(prompts[i], a.new)
        fresh, cref, cr = refs[i]
        if c1 > 0 and c2 > 0 and cr[1] > 0:
            same, first, dlp = diff(out, cref)
            ok &= same
            verdict = "identical to the off-cache reference" if same else \
                f"DIFFERS from the off-cache reference (token {first}, logprob {dlp:.2e})"
        else:
            same, first, dlp = diff(out, fresh)
            verdict = (f"cache missed (evicted); against the fresh reference: "
                       f"{'identical' if same else f'token {first}, logprob {dlp:.2e} (not judged)'}")
        print(f"  prompt {i}: cached {c1} / {c2}, {dt:.1f} s: {verdict}")


prompts = [prompt(a.len, i) for i in range(a.n + a.fill)]
for i in range(a.n):
    reference(i, prompts[i])
round_("round 1", range(a.n))
if a.fill:
    for i in range(a.n, a.n + a.fill):
        reference(i, prompts[i])
    round_("round 2 (fuller pool)", range(a.n + a.fill))
print("ALL IDENTICAL" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
