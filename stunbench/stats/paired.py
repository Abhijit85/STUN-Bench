"""Small paired-stat helpers for released row logs."""

def mcnemar_counts(a_correct, b_correct):
    b01=b10=0
    for a,b in zip(a_correct,b_correct):
        if (not a) and b: b01 += 1
        elif a and (not b): b10 += 1
    return {"a_wrong_b_right": b01, "a_right_b_wrong": b10}
