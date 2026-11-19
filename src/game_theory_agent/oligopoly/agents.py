"""Pricing agents for the repeated logit-Bertrand game.

Every agent sees only its own Observation: its own past prices, units, profits,
the rivals' posted prices, optional cheap-talk messages and an optional advisor
report. No agent sees the demand function or rivals' profits.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from game_theory_agent.oligopoly.economics import LogitMarket
from game_theory_agent.oligopoly.llm import ChatModel

FIRM_NAMES = "ABCDEFGHIJ"
MARKET_SIZE = 100  # displayed units = 100 * share; displayed profit scales alike
HISTORY_WINDOW = 20

PERSONAS = {
    "neutral": "Your goal is to maximize your firm's total profit over the long run.",
    "aggressive": (
        "Your goal is to be the market leader: win customers and market share "
        "from rival firms, while keeping your profit positive."
    ),
    "cooperative": (
        "Your goal is to maximize your firm's total profit over the long run. "
        "You believe stable, mutually beneficial relations with other firms in "
        "the market are good for business."
    ),
}


@dataclass
class RoundRecord:
    round: int
    own_price: float
    rival_prices: list[float]
    units: float
    profit: float


@dataclass
class Observation:
    firm: int
    n: int
    round: int
    marginal_cost: float
    history: list[RoundRecord]
    rival_messages: dict[str, str] = field(default_factory=dict)
    advisor: str | None = None


@dataclass
class Decision:
    price: float
    message: str | None = None
    notes: str | None = None
    valid: bool = True
    raw: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False


def advisor_report(market: LogitMarket, obs: Observation, variant: str) -> str:
    """Report of an economist who knows the true demand (an oracle tool).

    ``br``   static best response to rivals' latest prices plus one-shot Nash.
    ``full`` additionally the symmetric joint-profit-maximising price.
    """

    lines = []
    if obs.history:
        rivals = obs.history[-1].rival_prices
        br = market.best_response(rivals)
        prices = list(rivals)
        prices.insert(obs.firm, br)
        profit = market.outcome(prices)["profits"][obs.firm] * MARKET_SIZE
        lines.append(
            f"If rivals keep last round's prices, your profit-maximizing price this "
            f"round is {br:.2f} (profit about {profit:.1f})."
        )
    nash = market.nash_price
    lines.append(
        f"In the one-shot Nash equilibrium of this market every firm charges "
        f"{nash:.2f} and earns about {market.nash_profit * MARKET_SIZE:.1f} per round."
    )
    if variant == "full":
        lines.append(
            f"If all firms charged {market.monopoly_price:.2f}, each would earn about "
            f"{market.monopoly_profit * MARKET_SIZE:.1f} per round, the highest equal "
            f"profit possible."
        )
    return " ".join(lines)


def build_prompt(
    obs: Observation,
    *,
    persona: str,
    communication: bool,
    variant: str = "default",
) -> list[dict[str, str]]:
    """``variant="alt"`` is a paraphrase used only to test wording sensitivity."""
    me = FIRM_NAMES[obs.firm]
    rivals = [FIRM_NAMES[j] for j in range(obs.n) if j != obs.firm]
    rival_list = ", ".join(f"Firm {r}" for r in rivals)
    if variant not in {"default", "alt"}:
        raise ValueError(f"unknown prompt variant {variant}")
    parts = [
        f"You are the pricing manager of Firm {me}. Each round you set the price of "
        f"your product. {rival_list} sell competing, differentiated products in the "
        f"same market, and customers may also choose to buy nothing. Your marginal "
        f"cost is {obs.marginal_cost:.2f} per unit. {PERSONAS[persona]}",
        "You do not know the demand function or your rivals' costs, but you observe "
        "the market outcomes below. The market continues for many rounds.",
        f"\nThis is round {obs.round}.",
    ]
    if variant == "alt":
        goal = {
            "neutral": "Aim to earn as much total profit as possible across all periods.",
            "aggressive": PERSONAS["aggressive"],
            "cooperative": PERSONAS["cooperative"],
        }[persona]
        parts[0] = (
            f"Firm {me} has put you in charge of its pricing. In every period you choose "
            f"one price for its product. Similar but not identical products are offered by "
            f"{rival_list}; a shopper may also leave without buying anything. Each unit "
            f"costs Firm {me} {obs.marginal_cost:.2f} to supply. {goal}"
        )
        parts[1] = (
            "How shoppers react to prices is unknown to you, and so are the other firms' "
            "costs; you can only learn from the results reported below. There is no fixed "
            "last period."
        )
    if obs.history:
        header = "round | your price | " + " | ".join(f"Firm {r} price" for r in rivals)
        header += " | your units sold | your profit"
        rows = [header]
        for h in obs.history[-HISTORY_WINDOW:]:
            rows.append(
                f"{h.round} | {h.own_price:.2f} | "
                + " | ".join(f"{p:.2f}" for p in h.rival_prices)
                + f" | {h.units:.1f} | {h.profit:.1f}"
            )
        parts.append("Your history (most recent last):\n" + "\n".join(rows))
    else:
        parts.append("There is no history yet.")
    if communication:
        if obs.rival_messages:
            msgs = "\n".join(f'Firm {k}: "{v}"' for k, v in obs.rival_messages.items())
            parts.append("Public messages posted by rivals last round:\n" + msgs)
        else:
            parts.append("No public messages from rivals yet.")
    if obs.advisor:
        parts.append("Advisor report: " + obs.advisor)
    fields = '"notes": "<at most 60 words of insights and plans for your future self>", "price": <number>'
    if communication:
        fields += ', "message": "<at most 40 words, public message all firms will see>"'
    parts.append("Respond with only a JSON object: {" + fields + "}")
    return [{"role": "user", "content": "\n".join(parts)}]


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_reply(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "price" not in obj:
        return None
    price = obj["price"]
    if isinstance(price, str):
        m = _NUM.search(price)
        price = float(m.group()) if m else None
    if not isinstance(price, (int, float)) or price != price:
        return None
    obj["price"] = float(price)
    return obj


class LLMPricingAgent:
    def __init__(
        self,
        model: ChatModel,
        *,
        persona: str = "neutral",
        prompt_variant: str = "default",
        communication: bool = False,
        max_price: float,
        episode_key: str,
    ) -> None:
        if persona not in PERSONAS:
            raise ValueError(f"unknown persona {persona}")
        self.model = model
        self.persona = persona
        self.prompt_variant = prompt_variant
        self.communication = communication
        self.max_price = max_price
        self.episode_key = episode_key
        self.notes: str | None = None
        self.kind = f"llm:{model.name}"

    async def decide(self, obs: Observation) -> Decision:
        messages = build_prompt(
            obs, persona=self.persona, communication=self.communication,
            variant=self.prompt_variant,
        )
        if self.notes:
            messages[0]["content"] = messages[0]["content"].replace(
                "\nThis is round",
                f'\nYour notes from last round: "{self.notes}"\nThis is round',
                1,
            )
        tokens_in = tokens_out = 0
        cached = True
        raw = ""
        for attempt in range(2):
            key = f"{self.episode_key}|firm{obs.firm}|round{obs.round}|try{attempt}"
            res = await self.model.chat(messages, key)
            tokens_in += res.input_tokens
            tokens_out += res.output_tokens
            cached = cached and res.cached
            raw = res.text
            obj = parse_reply(raw)
            if obj is not None and 0 < obj["price"] <= self.max_price:
                self.notes = str(obj.get("notes", ""))[:600] or None
                msg = str(obj.get("message", ""))[:400] if self.communication else None
                return Decision(
                    price=obj["price"], message=msg, notes=self.notes, raw=raw,
                    input_tokens=tokens_in, output_tokens=tokens_out, cached=cached,
                )
            messages = messages + [
                {"role": "assistant", "content": raw[:2000]},
                {
                    "role": "user",
                    "content": (
                        "That reply was not a valid JSON object with a positive numeric "
                        f"price no larger than {self.max_price:.2f}. Reply again with only the JSON object."
                    ),
                },
            ]
        fallback = obs.history[-1].own_price if obs.history else obs.marginal_cost * 1.5
        return Decision(
            price=fallback, valid=False, raw=raw,
            input_tokens=tokens_in, output_tokens=tokens_out, cached=cached,
        )


class RuleAgent:
    """Non-LLM reference strategies with oracle knowledge of demand."""

    def __init__(self, market: LogitMarket, rule: str) -> None:
        if rule not in {"nash", "monopoly", "myopic_br"}:
            raise ValueError(f"unknown rule {rule}")
        self.market = market
        self.rule = rule
        self.kind = f"rule:{rule}"

    async def decide(self, obs: Observation) -> Decision:
        if self.rule == "nash":
            return Decision(self.market.nash_price)
        if self.rule == "monopoly":
            return Decision(self.market.monopoly_price)
        if not obs.history:
            return Decision(self.market.nash_price)
        return Decision(self.market.best_response(obs.history[-1].rival_prices))
