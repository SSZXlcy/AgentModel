"""Rule side of the inner loop: the fallback drift table, recipe resolution, candidate generation.

The agent may override the drift judgment; everything it returns passes through the same
whitelists defined here.
"""
from __future__ import annotations

from .config import PolicyConfig
from .contracts import Cost, DataRecipe, DriftJudgment, TrainSpec
from . import diagnose

DRIFT_TYPES = ("covariate", "concept", "both", "none")
RECIPES = ("all", "adaptive", "recent:1", "recent:2", "recent:3")


def rule_judgment(evidence: dict) -> DriftJudgment:
    """The 2×2 attribution table, used whenever the LLM is absent, fails or answers out of bounds."""
    shifted = evidence["drift"]["shifted"]
    degraded = evidence["performance"]["degraded"]
    if shifted and degraded:
        return DriftJudgment("both", "adaptive", 0.6, "inputs moved and accuracy fell: widen data", "rule")
    if shifted:
        return DriftJudgment("covariate", "all", 0.6, "inputs moved, accuracy held: keep history", "rule")
    if degraded:
        return DriftJudgment("concept", "recent:2", 0.6, "accuracy fell on similar inputs: favour recent data", "rule")
    return DriftJudgment("none", "adaptive", 0.6, "no shift and no degradation", "rule")


def resolve_recipe(choice: str, window: int, pool: dict[int, int], fit_rows: int, policy: PolicyConfig) -> DataRecipe:
    """Turn a recipe name into concrete windows. `pool` maps each past window to its labelled count."""
    past = sorted(w for w in pool if w < window)
    if choice == "all":
        return DataRecipe(True, tuple(past), "all")
    if choice.startswith("recent:"):
        n = int(choice.split(":")[1])
        chosen = tuple(past[-n:]) if n else ()
        include_base = fit_rows + sum(pool[w] for w in chosen) < policy.min_samples
        return DataRecipe(include_base, chosen, choice)
    # adaptive: walk back until min_samples, fall back to the base set if history is too thin
    chosen, rows = [], fit_rows
    for w in reversed(past):
        if rows >= policy.min_samples:
            break
        chosen.append(w)
        rows += pool[w]
    return DataRecipe(rows < policy.min_samples, tuple(sorted(chosen)), "adaptive")


def candidates(judgment: DriftJudgment, last_diagnosis: str, window: int, pool: dict[int, int],
               fit_rows: int, policy: PolicyConfig) -> list[TrainSpec]:
    """One dimension varies per window: learning rate after an OVERFIT diagnosis, data otherwise."""
    primary = resolve_recipe(judgment.recipe, window, pool, fit_rows, policy)
    epochs = policy.max_epochs + (2 if last_diagnosis == diagnose.NOT_SATURATED else 0)
    if last_diagnosis == diagnose.OVERFIT:
        scales = [1.0] + list(policy.lr_scales_overfit)
        return [TrainSpec(primary, s, epochs) for s in scales[:policy.max_candidates]]
    specs, seen = [TrainSpec(primary, 1.0, epochs)], {primary.recipe_id}
    for choice in RECIPES:
        if len(specs) >= policy.max_candidates:
            break
        recipe = resolve_recipe(choice, window, pool, fit_rows, policy)
        if recipe.recipe_id not in seen:
            seen.add(recipe.recipe_id)
            specs.append(TrainSpec(recipe, 1.0, epochs))
    return specs


def cost_value(cost: Cost, policy: PolicyConfig) -> float:
    return (cost.train_seconds * policy.cost_per_second + cost.labels * policy.cost_per_label
            + cost.llm_tokens * policy.cost_per_token)
