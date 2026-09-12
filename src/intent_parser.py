"""Natural Language Intent Parser and Semantic Search Index for ContextStyle.

Deconstructs free-text fashion search queries into structured attributes
and retrieves relevant catalog items via TF-IDF vector embeddings and cosine similarity.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Canonical Occasion Mapping
OCCASION_MAPPINGS = {
    "brunch": "Casual",
    "coffee": "Casual",
    "coffee date": "Casual",
    "weekend": "Casual",
    "casual": "Casual",
    "outing": "Casual",
    "office": "Formal",
    "corporate": "Formal",
    "interview": "Formal",
    "formal": "Formal",
    "business": "Formal",
    "meeting": "Formal",
    "work": "Formal",
    "cocktail": "Party",
    "party": "Party",
    "club": "Party",
    "clubbing": "Party",
    "night out": "Party",
    "rooftop": "Party",
    "mixer": "Party",
    "dinner": "Party",
    "beach": "Beach",
    "vacation": "Beach",
    "resort": "Beach",
    "pool": "Beach",
    "goa": "Beach",
    "summer trip": "Beach",
    "wedding": "Ethnic",
    "festive": "Ethnic",
    "diwali": "Ethnic",
    "sangeet": "Ethnic",
    "traditional": "Ethnic",
    "ethnic": "Ethnic",
    "puja": "Ethnic",
    "gym": "Gym",
    "workout": "Gym",
    "fitness": "Gym",
    "running": "Gym",
    "athleisure": "Gym",
}

STYLE_VIBES_VOCAB = [
    "pastel", "minimalist", "minimal", "vibrant", "streetwear", "classic",
    "oversized", "floral", "all black", "classy", "chic", "boho", "aesthetic",
    "smart casual", "monochrome", "clean", "preppy", "retro", "breathable",
    "relaxed", "tailored"
]

COLOR_VOCAB = [
    "black", "white", "navy blue", "navy", "olive green", "olive", "beige",
    "pastel pink", "pink", "lavender", "mustard yellow", "mustard", "charcoal",
    "charcoal grey", "tan brown", "tan", "mint green", "mint", "maroon",
    "burgundy", "terracotta", "cream", "blue", "green", "brown"
]

FABRIC_VOCAB = ["cotton", "linen", "silk", "denim", "leather", "polyester"]


@dataclass
class ParsedIntent:
    """Structured container for parsed search intent."""
    raw_query: str
    budget_max: float = 5000.0
    gender: str = "Unisex"
    target_occasion: Optional[str] = None
    occasion_detected_keyword: Optional[str] = None
    style_vibes: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    fabrics: List[str] = field(default_factory=list)
    clean_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "budget_max": self.budget_max,
            "gender": self.gender,
            "target_occasion": self.target_occasion,
            "occasion_keyword": self.occasion_detected_keyword,
            "style_vibes": self.style_vibes,
            "colors": self.colors,
            "fabrics": self.fabrics,
            "clean_query": self.clean_query,
        }


class IntentParser:
    """Deconstructs free-text natural language fashion prompts into structured intent."""

    def __init__(self, default_budget: float = 5000.0):
        self.default_budget = default_budget

    def parse(self, query: str) -> ParsedIntent:
        """Parses raw user input into a ParsedIntent dataclass."""
        if not query or not query.strip():
            return ParsedIntent(raw_query="", budget_max=self.default_budget)

        raw_query = query.strip()
        q_lower = raw_query.lower()

        # 1. Budget extraction
        budget_max = self._extract_budget(q_lower)

        # 2. Gender extraction
        gender = self._extract_gender(q_lower)

        # 3. Occasion extraction
        target_occasion, occ_keyword = self._extract_occasion(q_lower)

        # 4. Style vibes extraction
        style_vibes = self._extract_styles(q_lower)

        # 5. Colors & Fabrics extraction
        colors = [c for c in COLOR_VOCAB if re.search(r"\b" + re.escape(c) + r"\b", q_lower)]
        fabrics = [f for f in FABRIC_VOCAB if re.search(r"\b" + re.escape(f) + r"\b", q_lower)]

        # 6. Clean query for semantic indexing
        clean_query = self._build_clean_query(q_lower)

        return ParsedIntent(
            raw_query=raw_query,
            budget_max=budget_max,
            gender=gender,
            target_occasion=target_occasion,
            occasion_detected_keyword=occ_keyword,
            style_vibes=style_vibes,
            colors=colors,
            fabrics=fabrics,
            clean_query=clean_query
        )

    def _extract_budget(self, query: str) -> float:
        """Extracts budget limits from variations like 'under 4000', 'below ₹3500', 'budget 5k'."""
        patterns = [
            # Under/below/less than/max/upto/within ₹4000 or 4k or 4500
            r"(?:under|below|less\s+than|max|budget|within|upto|up\s+to|sub|around)\s*(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)\s*(k\b|thousand\b)?",
            # ₹3500 / 3500 rs / 3.5k
            r"(?:rs\.?|inr|₹)\s*(\d+(?:\.\d+)?)\s*(k\b|thousand\b)?",
            # Number followed by k e.g. '4k budget'
            r"\b(\d+(?:\.\d+)?)\s*k\b\s*(?:budget)?",
            # Number followed by rs/inr e.g. '4000 rs'
            r"\b(\d{3,5})\s*(?:rs|inr|rupees)\b",
        ]

        for pat in patterns:
            match = re.search(pat, query, re.IGNORECASE)
            if match:
                val_str = match.group(1)
                is_k = False
                if len(match.groups()) > 1 and match.group(2):
                    is_k = True
                elif "k" in match.group(0).lower():
                    is_k = True

                try:
                    val = float(val_str)
                    if is_k:
                        val *= 1000.0
                    if 300 <= val <= 50000:
                        return float(val)
                except ValueError:
                    continue

        return self.default_budget

    def _extract_gender(self, query: str) -> str:
        """Identifies target gender from query keywords."""
        has_women = bool(re.search(r"\b(women|woman|girls?|female|ladies|lady|her|she)\b", query))
        has_men = bool(re.search(r"\b(men|man|guys?|male|gentlemen|boy|boys|him|he)\b", query))

        # Check for explicit unisex
        if re.search(r"\b(unisex|couples?|everyone)\b", query):
            return "Unisex"

        if has_women and not has_men:
            return "Women"
        if has_men and not has_women:
            return "Men"
        return "Unisex"

    def _extract_occasion(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Extracts canonical occasion from query."""
        # Sort keywords by length descending so longer phrases match first (e.g. "coffee date" before "coffee")
        sorted_keywords = sorted(OCCASION_MAPPINGS.keys(), key=len, reverse=True)
        for kw in sorted_keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", query):
                return OCCASION_MAPPINGS[kw], kw
        return None, None

    def _extract_styles(self, query: str) -> List[str]:
        """Detects style vibes from query."""
        detected = []
        for vibe in STYLE_VIBES_VOCAB:
            if re.search(r"\b" + re.escape(vibe) + r"\b", query):
                detected.append(vibe)
        return detected

    def _build_clean_query(self, query: str) -> str:
        """Strips budget numbers and noise to produce semantic text."""
        # Remove budget mentions
        cleaned = re.sub(r"(?:under|below|less\s+than|max|budget|within|upto|up\s+to|around)?\s*(?:rs\.?|inr|₹)?\s*\d+(?:\.\d+)?\s*(?:k\b|thousand\b)?", " ", query)
        cleaned = re.sub(r"\b(?:rs|inr|rupees)\b", " ", cleaned)
        # Remove common filler words
        cleaned = re.sub(r"\b(for|a|an|the|in|with|of|looking|find|me|outfit|look|wear|set|collection)\b", " ", cleaned)
        # Clean extra whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned if cleaned else query


class SemanticSearchIndex:
    """TF-IDF vector space engine over catalog descriptions and metadata."""

    def __init__(self, catalog_df: pd.DataFrame):
        self.catalog_df = catalog_df.copy().reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            max_features=15000
        )
        self.tfidf_matrix = None
        self._build_index()

    def _build_index(self) -> None:
        """Constructs rich composite text representation and computes TF-IDF matrix."""
        # Compose searchable documents
        corpus = (
            self.catalog_df["title"].fillna("") + " " +
            self.catalog_df["category"].fillna("") + " " +
            self.catalog_df["sub_category"].fillna("") + " " +
            self.catalog_df["brand"].fillna("") + " " +
            self.catalog_df["gender"].fillna("") + " " +
            self.catalog_df["color"].fillna("") + " " +
            self.catalog_df["fabric"].fillna("") + " " +
            self.catalog_df["occasion_tags"].fillna("").str.replace(";", " ") + " " +
            self.catalog_df["style_tags"].fillna("").str.replace(";", " ") + " " +
            self.catalog_df["description"].fillna("")
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        gender: Optional[str] = None,
        max_price: Optional[float] = None,
        occasion: Optional[str] = None,
        top_k: int = 20
    ) -> pd.DataFrame:
        """Ranks catalog items by cosine similarity matching query vector and filters.

        Returns DataFrame of top_k items with 'relevance_score' column in [0, 1].
        """
        if not query or not query.strip():
            # If query is empty, return top rated items within filters
            filtered_df = self.catalog_df.copy()
            if category:
                filtered_df = filtered_df[filtered_df["category"] == category]
            if gender and gender != "Unisex":
                filtered_df = filtered_df[filtered_df["gender"].isin([gender, "Unisex"])]
            if max_price:
                filtered_df = filtered_df[filtered_df["price"] <= max_price]
            if occasion:
                filtered_df = filtered_df[filtered_df["occasion_tags"].str.contains(occasion, case=False, na=False)]

            top_items = filtered_df.sort_values(by="rating", ascending=False).head(top_k).copy()
            top_items["relevance_score"] = 0.5
            return top_items

        # Transform query vector
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Build candidate mask based on filters
        mask = np.ones(len(self.catalog_df), dtype=bool)

        if category:
            mask &= (self.catalog_df["category"] == category).values

        if gender and gender != "Unisex":
            mask &= self.catalog_df["gender"].isin([gender, "Unisex"]).values

        if max_price:
            mask &= (self.catalog_df["price"] <= max_price).values

        if occasion:
            occ_mask = self.catalog_df["occasion_tags"].str.contains(occasion, case=False, na=False).values
            # Give a boost to matching occasion or filter if strictly needed
            # Here we apply soft boost if available, or if filtered subset is large enough
            if np.sum(mask & occ_mask) >= 5:
                scores[mask & occ_mask] *= 1.25

        # Normalize scores to [0, 1]
        max_s = np.max(scores) if np.max(scores) > 0 else 1.0
        norm_scores = scores / max_s

        # Get valid candidate indices
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            # Fallback if hard filters yielded 0: relax max_price
            mask = np.ones(len(self.catalog_df), dtype=bool)
            if category:
                mask &= (self.catalog_df["category"] == category).values
            if gender and gender != "Unisex":
                mask &= self.catalog_df["gender"].isin([gender, "Unisex"]).values
            valid_indices = np.where(mask)[0]

        candidate_scores = norm_scores[valid_indices]
        # Sort top indices
        top_sub_indices = np.argsort(candidate_scores)[::-1][:top_k]
        top_indices = valid_indices[top_sub_indices]

        results = self.catalog_df.iloc[top_indices].copy()
        results["relevance_score"] = np.round(candidate_scores[top_sub_indices], 4)
        return results
