"""Ask the manager for a directly executable quote."""
from open_price import quote
from game_theory_agent.game_theory.price_language import legal_grammar


def feasible_quote(m, role, h, reference, low, high, seed, j, t, tape, offline):
    decimals = 0 if m == "native" else 2
    scale = 10**decimals
    grammar = legal_grammar(round(low*scale), round(high*scale), decimals)
    return quote(m, role, h, reference, low, high, seed, j, t, tape, offline, price_grammar=grammar)

