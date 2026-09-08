"""Patch the one known boundary artifact in the production resample output.

The production job (940569256367) ran code from before the fix in resample.py that
handles a natural `</think>` emitted as the last allowed thinking token. In that case
the old code appended a second `</think>` and labelled the completion think_capped=True.
This script rewrites those entries (strip the duplicated tag, set think_capped=False)
and writes a versioned copy; the raw job output is left untouched.
"""
import json, os, sys
src = os.path.join(os.environ["SILICO_EXPERIMENT_ARTIFACTS_DIR"], "resample", "completions.jsonl")
dst = os.path.join(os.environ["SILICO_EXPERIMENT_ARTIFACTS_DIR"], "resample", "completions_patched.jsonl")
n = 0
with open(src) as f, open(dst, "w") as g:
    for line in f:
        r = json.loads(line)
        for i, c in enumerate(r["completions"]):
            if "</think>\n</think>" in c:
                r["completions"][i] = c.replace("</think>\n</think>", "</think>", 1)
                r["think_capped"][i] = False
                r["think_tokens"][i] = r["think_tokens"][i]  # token count unchanged (tag was the 1024th token)
                n += 1
        g.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"patched {n} completions -> {dst}")
