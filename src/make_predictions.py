"""Run best-transformer.pt on the held-out 20% of dataset.csv and write one CSV row per example.

    python src/make_predictions.py                       # held-out split -> data/predictions_test.csv
    python src/make_predictions.py --split train --limit 5000

Columns: row (line index in dataset.csv), split, src, trg, pred, verdict, correct, logprob, min_tok_prob, finished,
         then src_* / trg_* / pred_* structural attributes (len, n_vars, n_terms, max_depth, max_const, n_and, ...)
verdict: exact | equivalent (proved) | wrong (counterexample found) | invalid (unparseable output) | unknown
"""
import argparse
import time
from pathlib import Path

import pandas as pd

import neureduce as nr

ap = argparse.ArgumentParser()
ap.add_argument("--neureduce", default="NeuReduce-original", help="path to the cloned NeuReduce repo")
ap.add_argument("--checkpoint", default="best-transformer.pt")
ap.add_argument("--split", choices=["test", "train"], default="test")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", default=None)
args = ap.parse_args()

t0 = time.time()
model, src_vocab, trg_vocab, pairs, train_idx, test_idx = nr.load(args.neureduce, args.checkpoint)
rows = (test_idx if args.split == "test" else train_idx)[: args.limit]
srcs = [pairs[i][0] for i in rows]
trgs = [pairs[i][1] for i in rows]
print(f"{len(rows)} {args.split} examples on {model.device} ...")

preds = nr.predict(model, srcs, src_vocab, trg_vocab)
df = pd.DataFrame({"row": rows, "split": args.split, "src": srcs, "trg": trgs, **pd.DataFrame(preds)})
df["verdict"] = [nr.verdict(p, t) for p, t in zip(df["pred"], df["trg"])]
df["correct"] = df["verdict"].isin(["exact", "equivalent"])
df = df[["row", "split", "src", "trg", "pred", "verdict", "correct", "logprob", "min_tok_prob", "finished"]]
for col in ["src", "trg", "pred"]:
    attrs = pd.DataFrame([nr.attributes(e) for e in df[col]]).add_prefix(f"{col}_")
    df = pd.concat([df, attrs], axis=1)

out = Path(args.out or f"data/predictions_{args.split}.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(df["verdict"].value_counts().to_string())
print(f"exact: {(df.verdict == 'exact').mean():.3f} | exact+equivalent: "
      f"{df.verdict.isin(['exact', 'equivalent']).mean():.3f} | wrote {out} in {time.time() - t0:.0f}s")
