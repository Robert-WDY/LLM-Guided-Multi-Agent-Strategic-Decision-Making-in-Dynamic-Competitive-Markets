"""Canonical decimal price ranges for constrained numeric decoding."""

from functools import lru_cache


@lru_cache(maxsize=4096)
def legal_grammar(low_tick, high_tick, decimals):
    if low_tick < 0 or low_tick > high_tick:
        raise ValueError("Empty or negative price interval")

    def span(lo, hi, width):
        if not lo:
            return ""
        prefix = '"." ' if decimals and width == decimals else ""

        def digit(a, b):
            return '"' + str(a) + '"' if a == b else f"[{a}-{b}]"

        i, j = int(lo[0]), int(hi[0])
        zeros, nines = "0" * (width - 1), "9" * (width - 1)
        if i == j:
            return prefix + digit(i, i) + " " + span(lo[1:], hi[1:], width - 1)
        alternatives = []
        if lo[1:] != zeros:
            alternatives.append(digit(i, i) + " " + span(lo[1:], nines, width - 1))
            i += 1
        upper_partial = hi[1:] != nines
        last = j - int(upper_partial)
        if i <= last:
            alternatives.append(digit(i, last) + " " + span(zeros, nines, width - 1))
        if upper_partial:
            alternatives.append(digit(j, j) + " " + span(zeros, hi[1:], width - 1))
        return prefix + "(" + " | ".join(alternatives) + ")"

    alternatives = []
    for width in range(decimals + 1, max(decimals + 1, len(str(high_tick))) + 1):
        lo = max(low_tick, 0 if width == decimals + 1 else 10 ** (width - 1))
        hi = min(high_tick, 10**width - 1)
        if lo <= hi:
            alternatives.append(span(str(lo).zfill(width), str(hi).zfill(width), width))
    return "root ::= " + " | ".join(alternatives)
