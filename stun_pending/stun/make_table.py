"""Turn one or more summary.json files into LaTeX rows in the paper's held-out table format:
Arm & held-out acc & held-out R@5 & labeled acc & labeled R@5

python -m stun.make_table results/p0_jina/summary.json results/p0_cleanbge/summary.json
"""
import json
import sys


def fmt(v, sd=None):
    s = f"{v:.3f}".lstrip("0") if v < 1 else f"{v:.3f}"
    return s + (f"\\sd{{{sd:.3f}}}".replace("{0.", "{.") if sd else "")


for path in sys.argv[1:]:
    print(f"% {path}")
    for s in json.load(open(path))["summary"]:
        if "heldout_acc" not in s:
            continue
        print(f"{s['arm']} & {fmt(s['heldout_acc'], s.get('heldout_acc_sd'))} & {fmt(s['heldout_r5'])} & "
              f"{fmt(s['labeled_acc'], s.get('labeled_acc_sd'))} & {fmt(s['labeled_r5'])} \\\\")
