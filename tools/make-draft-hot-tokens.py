"""Token list for the MTP draft's reduced head (KT_DRAFT_HOT, patches/kt_draft_hot.py).

Every token that occurs in the training corpus (the Japanese prose the serve
writes: ~10.6K distinct tokens cover 99.96% of the validation set), every
special / added token (so the draft can still propose <|im_end|> and the
thinking tags), then the lowest ids (BPE merges in order of frequency, which
covers common English and code) until the list has --size entries. Saved as a
sorted list of ints with torch.save.

A second source can be a directory of plain-text novels (--txt-dir; on our
corpus 36 files, 18.9M tokens, 14,002 distinct tokens; the jsonl-only list
covered 99.56% of them).

Usage: python tools/make-draft-hot-tokens.py <tokenizer dir> <corpus jsonl> <out.pt> [--size 32768] [--limit 3000] [--txt-dir DIR]
"""
import glob
import os
import argparse
import json

import torch
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("tokenizer")
ap.add_argument("corpus")
ap.add_argument("out")
ap.add_argument("--size", type=int, default=32768)
ap.add_argument("--limit", type=int, default=3000)
ap.add_argument("--txt-dir", default=None)
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
hot = set()
texts = []
for i, line in enumerate(open(args.corpus)):
    if i >= args.limit:
        break
    # the written answer only (after the thinking, before the end tag): what the draft proposes
    texts.append(json.loads(line)["text"].split("</think>")[-1].split("<|im_end|>")[0])
for ids in tok(texts, add_special_tokens=False)["input_ids"]:
    hot.update(ids)
from_jsonl = len(hot)
if args.txt_dir:
    for f in sorted(glob.glob(os.path.join(args.txt_dir, "*.txt"))):
        text = open(f, encoding="utf-8", errors="replace").read()
        for i in range(0, len(text), 200000):
            hot.update(tok.encode(text[i:i + 200000], add_special_tokens=False))
    print(f"{from_jsonl} distinct tokens from the jsonl, {len(hot) - from_jsonl} more from {args.txt_dir}")
corpus = len(hot)
hot.update(tok.all_special_ids)
hot.update(tok.convert_tokens_to_ids(list(tok.get_added_vocab().keys())))
special = len(hot) - corpus
i = 0
while len(hot) < args.size:
    hot.add(i)
    i += 1
n_vocab = len(tok)  # len() of this tokenizer recounts the vocabulary on every call
ids = sorted(t for t in hot if 0 <= t < n_vocab)
torch.save(ids, args.out)
print(f"{len(ids)} tokens: {corpus} from the corpus, {special} special/added, filled with ids below {i} -> {args.out}")
