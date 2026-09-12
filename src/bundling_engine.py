"""Multi-Item Outfit Bundling & Constraint Optimization Engine.

Constructs 3- or 4-piece coordinated outfit bundles (topwear, bottomwear, footwear, accessories)
optimizing for budget adherence (knapsack constraint), color harmony, style cohesion, and relevance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from src.intent_parser import ParsedIntent, SemanticSearchIndex

# Neutral colors that pair easily with almost everything
NEUTRALS = {"black", "white", "beige", "navy blue", "navy", "charcoal grey", "charcoal", "tan brown", "tan", "cream"}

# Color palette harmony clusters
COLOR_CLUSTERS = {
    "monochrome_dark": {"black", "charcoal grey", "charcoal"},
    "monochrome_earth": {"beige", "tan brown", "tan", "cream"},
    "pastels": {"pastel pink", "lavender", "mint green", "sky blue", "cream", "white", "beige"},
    "earthy_warm": {"olive green", "tan brown", "tan", "terracotta", "mustard yellow", "beige", "cream"},
    "regal_rich": {"navy blue", "maroon", "burgundy", "black", "white"},
}


def compute_color_harmony(colors: List[str]) -> float:
    """Calculates color harmony score in [0.0, 1.0] for an ensemble of colors."""
    if not colors:
        return 0.5

    clean_colors = [c.lower().strip() for c in colors]
    unique_colors = set(clean_colors)

    # 1. Complete monochrome
    if len(unique_colors) == 1:
        return 0.95

    # 2. Check cluster membership
    for cluster_name, cluster_set in COLOR_CLUSTERS.items():
        if unique_colors.issubset(cluster_set):
            return 0.92

    # 3. Neutral pairing (e.g. 1-2 accents + neutrals)
    neutrals_count = sum(1 for c in clean_colors if c in NEUTRALS)
    non_neutrals = [c for c in clean_colors if c not in NEUTRALS]

    if neutrals_count >= len(clean_colors) - 1:
        # All neutral or only 1 accent with neutrals
        return 0.88
    elif len(set(non_neutrals)) <= 1 and neutrals_count >= 1:
        return 0.85
    elif neutrals_count >= 1:
        # Mixed with at least 1 neutral base
        return 0.72
    else:
        # Clashing or multi-saturated colors
        return 0.50


def compute_style_harmony(styles_list: List[List[str]]) -> float:
    """Calculates style coherence score in [0.0, 1.0]."""
    if not styles_list:
        return 0.5

    all_styles = [s for sublist in styles_list for s in sublist]
    if not all_styles:
        return 0.6

    from collections import Counter
    counts = Counter(all_styles)
    max_shared = max(counts.values()) if counts else 0
    total_items = len(styles_list)

    # If all items share at least one style tag (e.g. all Minimalist)
    if max_shared >= total_items:
        return 0.95
    elif max_shared >= total_items - 1:
        return 0.82
    elif max_shared >= 2:
        return 0.70
    return 0.55


@dataclass
class OutfitBundle:
    """Represents a curated multi-item outfit bundle."""
    items: List[Dict[str, Any]]
    total_price: int
    original_price: int
    savings_amount: int
    budget_max: float
    is_budget_compliant: bool
    categories_included: List[str]
    is_complete_4_piece: bool
    cohesion_score: float
    relevance_score: float
    stylist_rationale: str
    fallback_applied: bool = False
    fallback_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": self.items,
            "total_price": self.total_price,
            "original_price": self.original_price,
            "savings_amount": self.savings_amount,
            "budget_max": self.budget_max,
            "is_budget_compliant": self.is_budget_compliant,
            "categories_included": self.categories_included,
            "is_complete_4_piece": self.is_complete_4_piece,
            "cohesion_score": self.cohesion_score,
            "relevance_score": self.relevance_score,
            "stylist_rationale": self.stylist_rationale,
            "fallback_applied": self.fallback_applied,
            "fallback_message": self.fallback_message
        }


class StylistRationaleGenerator:
    """Generates natural language stylist explanations for outfit pairings."""

    @staticmethod
    def generate(items: List[Dict[str, Any]], intent: ParsedIntent, fallback_msg: Optional[str] = None) -> str:
        if not items:
            return "No items selected."

        # Extract dominant details
        top_item = next((i for i in items if i.get("category") == "topwear"), items[0])
        bottom_item = next((i for i in items if i.get("category") == "bottomwear"), None)
        shoe_item = next((i for i in items if i.get("category") == "footwear"), None)
        acc_item = next((i for i in items if i.get("category") == "accessories"), None)

        top_fabric = top_item.get("fabric", "cotton")
        top_color = top_item.get("color", "neutral")
        top_sub = top_item.get("sub_category", "top")

        occasion = intent.target_occasion or "versatile everyday"
        vibes = ", ".join(intent.style_vibes) if intent.style_vibes else "effortlessly modern"

        parts = [f"Curated for a {occasion.lower()} aesthetic featuring {vibes} vibes."]

        if bottom_item and shoe_item:
            bottom_color = bottom_item.get("color", "")
            bottom_sub = bottom_item.get("sub_category", "bottomwear")
            shoe_sub = shoe_item.get("sub_category", "footwear")
            parts.append(
                f"Paired a {top_color.lower()} {top_fabric.lower()} {top_sub.lower()} "
                f"with structured {bottom_color.lower()} {bottom_sub.lower()} and matching {shoe_sub.lower()}."
            )
        else:
            parts.append(f"Highlighted by a {top_color.lower()} {top_fabric.lower()} {top_sub.lower()}.")

        if acc_item:
            parts.append(f"Completed with a sleek {acc_item.get('sub_category', 'accessory').lower()} for an elevated finish.")

        if fallback_msg:
            parts.append(f"({fallback_msg})")

        return " ".join(parts)


class OutfitBundlingOptimizer:
    """Greedy/Dynamic Knapsack Multi-Item Bundling Engine with Color and Budget Constraints."""

    def __init__(self, search_index: SemanticSearchIndex):
        self.search_index = search_index
        self.categories_order = ["topwear", "bottomwear", "footwear", "accessories"]

    def curate_bundle(
        self,
        intent: ParsedIntent,
        top_k_candidates: int = 15
    ) -> OutfitBundle:
        """Constructs an optimal 3- or 4-piece outfit bundle strictly under intent.budget_max."""
        budget_max = intent.budget_max
        query = intent.clean_query or intent.raw_query
        gender = intent.gender
        occasion = intent.target_occasion

        # 1. Fetch top candidates per category
        candidates_by_cat = {}
        for cat in self.categories_order:
            df_cat = self.search_index.search(
                query=query,
                category=cat,
                gender=gender,
                occasion=occasion,
                top_k=top_k_candidates
            )
            # If no candidates with occasion filter, relax occasion
            if len(df_cat) == 0:
                df_cat = self.search_index.search(
                    query=query,
                    category=cat,
                    gender=gender,
                    top_k=top_k_candidates
                )
            candidates_by_cat[cat] = df_cat.to_dict(orient="records")

        # 2. Try to find the best 4-piece bundle within budget
        best_4_bundle = self._optimize_ensemble(
            candidates_by_cat=candidates_by_cat,
            categories=["topwear", "bottomwear", "footwear", "accessories"],
            budget_max=budget_max
        )

        if best_4_bundle is not None:
            items, total_price, cohesion, rel = best_4_bundle
            orig_price = int(round(total_price * 1.18))
            savings = orig_price - total_price
            rationale = StylistRationaleGenerator.generate(items, intent)

            return OutfitBundle(
                items=items,
                total_price=total_price,
                original_price=orig_price,
                savings_amount=savings,
                budget_max=budget_max,
                is_budget_compliant=True,
                categories_included=[i["category"] for i in items],
                is_complete_4_piece=True,
                cohesion_score=round(cohesion, 3),
                relevance_score=round(rel, 3),
                stylist_rationale=rationale,
                fallback_applied=False,
                fallback_message=None
            )

        # 3. Fallback 1: Budget too tight for 4 pieces -> Drop accessory, find best 3-piece
        best_3_bundle = self._optimize_ensemble(
            candidates_by_cat=candidates_by_cat,
            categories=["topwear", "bottomwear", "footwear"],
            budget_max=budget_max
        )

        fallback_msg = f"Curated 3 essential pieces to stay strictly within your ₹{int(budget_max):,} budget"

        if best_3_bundle is not None:
            items, total_price, cohesion, rel = best_3_bundle
            orig_price = int(round(total_price * 1.15))
            savings = orig_price - total_price
            rationale = StylistRationaleGenerator.generate(items, intent, fallback_msg)

            return OutfitBundle(
                items=items,
                total_price=total_price,
                original_price=orig_price,
                savings_amount=savings,
                budget_max=budget_max,
                is_budget_compliant=True,
                categories_included=[i["category"] for i in items],
                is_complete_4_piece=False,
                cohesion_score=round(cohesion, 3),
                relevance_score=round(rel, 3),
                stylist_rationale=rationale,
                fallback_applied=True,
                fallback_message=fallback_msg
            )

        # 4. Fallback 2: Extremely tight budget -> Select cheapest valid items across categories
        cheapest_items = []
        running_cost = 0
        cats_to_pick = ["topwear", "bottomwear", "footwear"]

        for cat in cats_to_pick:
            pool = sorted(candidates_by_cat.get(cat, []), key=lambda x: x["price"])
            for cand in pool:
                if running_cost + cand["price"] <= budget_max or len(cheapest_items) == 0:
                    cheapest_items.append(cand)
                    running_cost += cand["price"]
                    break

        # If even cheapest exceeds budget_max, take the absolute lowest single items to get closest
        if running_cost > budget_max and len(cheapest_items) > 1:
            # Drop bottom or footwear if needed to strictly obey budget
            while running_cost > budget_max and len(cheapest_items) > 1:
                dropped = cheapest_items.pop()
                running_cost -= dropped["price"]

        total_price = sum(i["price"] for i in cheapest_items)
        orig_price = int(round(total_price * 1.12))
        savings = max(0, orig_price - total_price)
        cohesion = 0.65
        rel = float(np.mean([i.get("relevance_score", 0.5) for i in cheapest_items])) if cheapest_items else 0.5
        strict_compliant = total_price <= budget_max
        fb_text = f"Optimized essential wardrobe selection to meet ₹{int(budget_max):,} limit"

        return OutfitBundle(
            items=cheapest_items,
            total_price=total_price,
            original_price=orig_price,
            savings_amount=savings,
            budget_max=budget_max,
            is_budget_compliant=strict_compliant,
            categories_included=[i["category"] for i in cheapest_items],
            is_complete_4_piece=False,
            cohesion_score=cohesion,
            relevance_score=rel,
            stylist_rationale=StylistRationaleGenerator.generate(cheapest_items, intent, fb_text),
            fallback_applied=True,
            fallback_message=fb_text
        )

    def _optimize_ensemble(
        self,
        candidates_by_cat: Dict[str, List[Dict[str, Any]]],
        categories: List[str],
        budget_max: float
    ) -> Optional[Tuple[List[Dict[str, Any]], int, float, float]]:
        """Evaluates combinations of candidates across specified categories to maximize score <= budget_max."""
        # Check that we have candidates for all requested categories
        for cat in categories:
            if not candidates_by_cat.get(cat):
                return None

        # Filter candidates whose individual price already exceeds budget
        filtered_candidates = {
            cat: [c for c in candidates_by_cat[cat] if c["price"] < budget_max]
            for cat in categories
        }
        for cat in categories:
            if not filtered_candidates[cat]:
                return None

        best_score = -1e9
        best_combination = None
        best_price = 0
        best_cohesion = 0.0
        best_rel = 0.0

        if len(categories) == 4:
            c1, c2, c3, c4 = categories
            list1 = filtered_candidates[c1][:12]
            list2 = filtered_candidates[c2][:12]
            list3 = filtered_candidates[c3][:12]
            list4 = filtered_candidates[c4][:10]

            for i1 in list1:
                p1 = i1["price"]
                if p1 >= budget_max:
                    continue
                for i2 in list2:
                    p12 = p1 + i2["price"]
                    if p12 >= budget_max:
                        continue
                    for i3 in list3:
                        p123 = p12 + i3["price"]
                        if p123 >= budget_max:
                            continue
                        for i4 in list4:
                            total_p = p123 + i4["price"]
                            if total_p > budget_max:
                                continue

                            items = [i1, i2, i3, i4]
                            score, cohesion, rel = self._evaluate_bundle(items)
                            if score > best_score:
                                best_score = score
                                best_combination = items
                                best_price = total_p
                                best_cohesion = cohesion
                                best_rel = rel

        elif len(categories) == 3:
            c1, c2, c3 = categories
            list1 = filtered_candidates[c1][:14]
            list2 = filtered_candidates[c2][:14]
            list3 = filtered_candidates[c3][:14]

            for i1 in list1:
                p1 = i1["price"]
                if p1 >= budget_max:
                    continue
                for i2 in list2:
                    p12 = p1 + i2["price"]
                    if p12 >= budget_max:
                        continue
                    for i3 in list3:
                        total_p = p12 + i3["price"]
                        if total_p > budget_max:
                            continue

                        items = [i1, i2, i3]
                        score, cohesion, rel = self._evaluate_bundle(items)
                        if score > best_score:
                            best_score = score
                            best_combination = items
                            best_price = total_p
                            best_cohesion = cohesion
                            best_rel = rel

        if best_combination is not None:
            return best_combination, best_price, best_cohesion, best_rel
        return None

    def _evaluate_bundle(self, items: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        """Calculates global objective score combining relevance, rating, color harmony, and style coherence."""
        # 1. Relevance & Rating
        relevances = [item.get("relevance_score", 0.5) for item in items]
        ratings = [item.get("rating", 4.0) / 5.0 for item in items]
        mean_rel = float(np.mean(relevances))
        mean_rating = float(np.mean(ratings))

        # 2. Color Harmony
        colors = [item.get("color", "") for item in items]
        color_score = compute_color_harmony(colors)

        # 3. Style Harmony
        styles_list = [item.get("style_tags", "").split(";") for item in items]
        style_score = compute_style_harmony(styles_list)

        cohesion = 0.5 * color_score + 0.5 * style_score

        # Composite Objective Function
        # We weight relevance (40%), rating (20%), and cohesion (40%)
        total_score = (0.40 * mean_rel) + (0.20 * mean_rating) + (0.40 * cohesion)
        return total_score, cohesion, mean_rel

    def swap_item(
        self,
        current_bundle: OutfitBundle,
        category: str,
        new_sku_id: str
    ) -> OutfitBundle:
        """Swaps an individual item in the bundle and recalculates totals and budget compliance."""
        new_item_df = self.search_index.catalog_df[self.search_index.catalog_df["sku_id"] == new_sku_id]
        if len(new_item_df) == 0:
            return current_bundle

        new_item = new_item_df.iloc[0].to_dict()
        updated_items = []
        swapped = False

        for itm in current_bundle.items:
            if itm["category"] == category and not swapped:
                updated_items.append(new_item)
                swapped = True
            else:
                updated_items.append(itm)

        if not swapped:
            updated_items.append(new_item)

        total_price = sum(i["price"] for i in updated_items)
        orig_price = int(round(total_price * 1.18))
        savings = orig_price - total_price
        is_compliant = total_price <= current_bundle.budget_max

        # Recalculate scores
        _, cohesion, rel = self._evaluate_bundle(updated_items)

        intent = ParsedIntent(raw_query="", budget_max=current_bundle.budget_max)
        rationale = StylistRationaleGenerator.generate(updated_items, intent)

        return OutfitBundle(
            items=updated_items,
            total_price=total_price,
            original_price=orig_price,
            savings_amount=savings,
            budget_max=current_bundle.budget_max,
            is_budget_compliant=is_compliant,
            categories_included=[i["category"] for i in updated_items],
            is_complete_4_piece=len(updated_items) == 4,
            cohesion_score=round(cohesion, 3),
            relevance_score=round(rel, 3),
            stylist_rationale=rationale,
            fallback_applied=current_bundle.fallback_applied,
            fallback_message=current_bundle.fallback_message
        )
