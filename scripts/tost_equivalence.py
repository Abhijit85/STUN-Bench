"""
TOST (Two One-Sided Tests) equivalence test for SYNAPSE vs. Centralized-SYNAPSE
routing accuracy, addressing Reviewer ZkZQ's Q4 / W4:
  "Would an equivalence test at a stated margin support the claim of parity
   with centralized routing at the current number of seeds?"

USAGE
-----
Replace `synapse_scores` and `centralized_scores` below with the real
per-seed routing-accuracy values behind Table 27 (5 seeds currently; add
more if you re-run with additional seeds per Appendix B, Experiment 5).
If the two conditions were evaluated on the *same* 5 seeds / held-out
query sets (a paired design), set PAIRED = True for a more powerful test;
if seeds/test folds differ between conditions, leave PAIRED = False.

Pre-register the equivalence margin (`MARGIN`) BEFORE looking at the
result -- e.g. 0.03 (3 accuracy points), which is defensible given
typical seed-to-seed noise on this benchmark (Table 27 reports SD=0.02
per condition). Do not tune the margin after seeing the p-value.

Requires: pip install scipy statsmodels
"""

import numpy as np
from scipy import stats

# ---- 1. INPUT: replace with real per-seed values ----------------------
synapse_scores = np.array([0.90, 0.92, 0.93, 0.91, 0.94])       # PLACEHOLDER
centralized_scores = np.array([0.91, 0.93, 0.92, 0.92, 0.93])   # PLACEHOLDER
PAIRED = True     # same seeds/test folds for both conditions?
MARGIN = 0.03     # pre-registered equivalence margin (accuracy points)
ALPHA = 0.05


def tost_paired(x, y, margin, alpha=0.05):
    diff = x - y
    n = len(diff)
    mean_diff = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    df = n - 1
    t_lower = (mean_diff - (-margin)) / se   # H0: mean_diff <= -margin
    t_upper = (mean_diff - margin) / se      # H0: mean_diff >= margin
    p_lower = 1 - stats.t.cdf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    p_tost = max(p_lower, p_upper)
    equivalent = p_tost < alpha
    ci = stats.t.interval(1 - 2 * alpha, df, loc=mean_diff, scale=se)
    return dict(
        mean_diff=mean_diff,
        se=se,
        df=df,
        p_lower=p_lower,
        p_upper=p_upper,
        p_tost=p_tost,
        equivalent=equivalent,
        ci_90=ci,
    )


def tost_independent(x, y, margin, alpha=0.05):
    n1, n2 = len(x), len(y)
    mean_diff = x.mean() - y.mean()
    sp2 = ((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2)
    se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
    df = n1 + n2 - 2
    t_lower = (mean_diff - (-margin)) / se
    t_upper = (mean_diff - margin) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    p_tost = max(p_lower, p_upper)
    equivalent = p_tost < alpha
    ci = stats.t.interval(1 - 2 * alpha, df, loc=mean_diff, scale=se)
    return dict(
        mean_diff=mean_diff,
        se=se,
        df=df,
        p_lower=p_lower,
        p_upper=p_upper,
        p_tost=p_tost,
        equivalent=equivalent,
        ci_90=ci,
    )


def main():
    result = (
        tost_paired(synapse_scores, centralized_scores, MARGIN, ALPHA)
        if PAIRED
        else tost_independent(synapse_scores, centralized_scores, MARGIN, ALPHA)
    )

    print(f"Paired design: {PAIRED}")
    print(f"Pre-registered margin: +/-{MARGIN * 100:.1f} points, alpha={ALPHA}")
    print(
        "Mean difference (SYNAPSE - Centralized): "
        f"{result['mean_diff'] * 100:+.2f} points"
    )
    print(f"SE: {result['se'] * 100:.3f} points, df={result['df']}")
    print(
        "90% CI on the difference: "
        f"[{result['ci_90'][0] * 100:+.2f}, {result['ci_90'][1] * 100:+.2f}] points"
    )
    print(f"TOST p-value (max of two one-sided tests): {result['p_tost']:.4f}")
    print(
        f"Equivalent at alpha={ALPHA} and margin={MARGIN * 100:.1f}pt: "
        f"{result['equivalent']}"
    )
    print()
    print('Report this as: "A two-one-sided-tests equivalence test with a')
    print(
        f"pre-registered margin of {MARGIN * 100:.1f} accuracy points "
        f"{'confirms' if result['equivalent'] else 'does not confirm'}"
    )
    print("statistical equivalence between SYNAPSE and centralized routing")
    print(
        f"(90% CI on the difference: [{result['ci_90'][0] * 100:+.2f}, "
        f"{result['ci_90'][1] * 100:+.2f}] pts, p={result['p_tost']:.3f}).\""
    )


if __name__ == "__main__":
    main()
