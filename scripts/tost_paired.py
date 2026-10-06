#!/usr/bin/env python3
"""
tost_paired.py — paired two-one-sided-tests (TOST) equivalence test for the
SYNAPSE vs Centralized-SYNAPSE comparison behind Table 27.

Table 27 is a PAIRED design (same seeds for both conditions), so this uses the
paired-differences formula — not independent-samples.

IMPORTANT: pre-specify --margin BEFORE looking at the seed-level differences, and
justify it (e.g., larger than typical seed-to-seed noise on this benchmark).

Inputs: two files, one accuracy per line (or CSV with a column), SAME seed order.
    python tost_paired.py --syn syn.csv --cen cen.csv --margin 0.03
    python tost_paired.py --syn syn.csv --cen cen.csv --margin 0.03 --col acc

Reports: mean paired difference, 90% CI, both one-sided p-values, and PASS/FAIL
(equivalence established iff BOTH one-sided tests reject at alpha, i.e. the (1-2a)
CI lies entirely within [-margin, +margin]).
"""

import argparse
import csv
import math
import statistics as st


def read_col(path, col=None):
    vals = []
    with open(path) as f:
        if path.endswith(".csv") and col is not None:
            for row in csv.DictReader(f):
                vals.append(float(row[col]))
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # tolerate "seed,acc" or bare number
                tok = line.split(",")[-1]
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
    return vals


def t_sf(t, df):
    # survival function P(T>t) for Student-t via regularized incomplete beta
    x = df / (df + t * t)
    return 0.5 * _betai(df / 2.0, 0.5, x)


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    # Lentz continued fraction
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1e-30 if abs(d) < 1e-30 else d
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1 + aa * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1 + aa / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1 + aa * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1 + aa / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-10:
            break
    val = front * h
    return 1 - val if x > (a + 1) / (a + b + 2) else val


def t_crit(df, p):
    # invert two-sided-ish: find t with sf = p via bisection
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_sf(mid, df) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn", required=True)
    ap.add_argument("--cen", required=True)
    ap.add_argument("--margin", type=float, required=True, help="equivalence margin (e.g. 0.03)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--col", default=None)
    args = ap.parse_args()

    s = read_col(args.syn, args.col)
    c = read_col(args.cen, args.col)
    assert len(s) == len(c) and len(s) > 1, "need equal-length paired samples (same seed order)"
    d = [a - b for a, b in zip(s, c)]
    n = len(d)
    df = n - 1
    md = sum(d) / n
    sd = st.stdev(d)
    se = sd / math.sqrt(n)
    m = args.margin

    # TOST: H0a diff <= -m ; H0b diff >= +m
    t_lower = (md + m) / se  # test diff > -m
    t_upper = (md - m) / se  # test diff < +m
    p_lower = t_sf(t_lower, df)  # P(T > t_lower)
    p_upper = t_sf(-t_upper, df)  # P(T < t_upper) = sf(-t_upper)
    p_tost = max(p_lower, p_upper)

    tc = t_crit(df, args.alpha)  # one-sided crit
    ci_lo = md - tc * se  # (1-2alpha) CI
    ci_hi = md + tc * se
    equiv = (p_lower < args.alpha) and (p_upper < args.alpha)

    print(f"n={n}  mean paired diff (SYN-CEN) = {md:+.4f}  sd={sd:.4f}  se={se:.4f}")
    print(f"margin = ±{m}")
    print(f"{int((1-2*args.alpha)*100)}% CI (paired): [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"one-sided p (> -m): {p_lower:.4f}   one-sided p (< +m): {p_upper:.4f}   TOST p = {p_tost:.4f}")
    print(
        "RESULT: "
        + (
            "EQUIVALENT within ±%.3f (both one-sided tests reject at a=%.2f)" % (m, args.alpha)
            if equiv
            else "NOT established — CI not fully inside the margin"
        )
    )
    if not equiv:
        print("  -> report: 'no significant difference at n=%d; equivalence not established at margin %.3f.'" % (n, m))


if __name__ == "__main__":
    main()
