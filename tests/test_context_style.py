"""Unit tests for ContextStyle Phase 1: Catalog Pipeline and Data Schema."""

import pytest
import pandas as pd
from pathlib import Path
from src.catalog_pipeline import generate_catalog, save_catalog, load_catalog

REQUIRED_COLUMNS = [
    "sku_id", "title", "category", "sub_category", "brand", "gender",
    "price", "rating", "occasion_tags", "style_tags", "color", "fabric", "description"
]

EXPECTED_CATEGORIES = {"topwear", "bottomwear", "footwear", "accessories"}
EXPECTED_GENDERS = {"Men", "Women", "Unisex"}


class TestCatalogPipeline:
    """Test suite for catalog generation and schema validation."""

    @pytest.fixture(scope="module")
    def catalog_df(self):
        return generate_catalog(num_items=2500, seed=42)

    def test_catalog_row_count(self, catalog_df):
        assert len(catalog_df) == 2500, f"Expected 2500 items, got {len(catalog_df)}"

    def test_catalog_schema_columns(self, catalog_df):
        for col in REQUIRED_COLUMNS:
            assert col in catalog_df.columns, f"Missing required column: {col}"

    def test_no_null_values_in_critical_fields(self, catalog_df):
        critical_cols = ["sku_id", "title", "category", "price", "gender", "rating"]
        for col in critical_cols:
            assert catalog_df[col].isnull().sum() == 0, f"Found nulls in column: {col}"

    def test_price_bounds_and_validity(self, catalog_df):
        assert (catalog_df["price"] >= 399).all(), "Found price below minimum ₹399"
        assert (catalog_df["price"] <= 8999).all(), "Found price above maximum ₹8999"
        assert (catalog_df["price"] > 0).all(), "Price must be strictly positive"

    def test_rating_bounds(self, catalog_df):
        assert (catalog_df["rating"] >= 3.2).all(), "Found rating below 3.2"
        assert (catalog_df["rating"] <= 4.9).all(), "Found rating above 4.9"

    def test_category_distribution_and_completeness(self, catalog_df):
        unique_categories = set(catalog_df["category"].unique())
        assert unique_categories == EXPECTED_CATEGORIES, f"Mismatch in categories: {unique_categories}"

        counts = catalog_df["category"].value_counts()
        for cat in EXPECTED_CATEGORIES:
            assert counts[cat] == 625, f"Expected 625 items for {cat}, got {counts[cat]}"

    def test_gender_validity(self, catalog_df):
        unique_genders = set(catalog_df["gender"].unique())
        assert unique_genders.issubset(EXPECTED_GENDERS)

    def test_sku_id_uniqueness(self, catalog_df):
        assert catalog_df["sku_id"].nunique() == len(catalog_df), "Duplicate sku_ids found"

    def test_save_and_load_catalog(self, tmp_path, catalog_df):
        saved = save_catalog(catalog_df, base_dir=tmp_path)
        assert saved["parquet"].exists()
        assert saved["csv"].exists()

        loaded_parquet = load_catalog(saved["parquet"])
        assert len(loaded_parquet) == len(catalog_df)

        loaded_csv = load_catalog(saved["csv"])
        assert len(loaded_csv) == len(catalog_df)


class TestIntentParser:
    """Test suite for Natural Language Intent Parser across 8 conversational query variations."""

    @pytest.fixture(scope="module")
    def parser(self):
        from src.intent_parser import IntentParser
        return IntentParser()

    def test_query_1_monsoon_brunch_pastel(self, parser):
        query = "monsoon brunch pastel casual look under ₹3500"
        parsed = parser.parse(query)
        assert parsed.budget_max == 3500.0
        assert parsed.target_occasion == "Casual"
        assert "pastel" in parsed.style_vibes
        assert parsed.gender == "Unisex"

    def test_query_2_corporate_dinner_men(self, parser):
        query = "smart casual attire for corporate dinner in Delhi for men budget 4500"
        parsed = parser.parse(query)
        assert parsed.budget_max == 4500.0
        assert parsed.gender == "Men"
        assert parsed.target_occasion in ["Formal", "Party"]
        assert "smart casual" in parsed.style_vibes

    def test_query_3_cocktail_party_women_4k(self, parser):
        query = "cocktail party look for women under 4k"
        parsed = parser.parse(query)
        assert parsed.budget_max == 4000.0
        assert parsed.gender == "Women"
        assert parsed.target_occasion == "Party"

    def test_query_4_minimalist_office_guys_below_3000(self, parser):
        query = "minimalist Friday office wear for guys below 3000"
        parsed = parser.parse(query)
        assert parsed.budget_max == 3000.0
        assert parsed.gender == "Men"
        assert parsed.target_occasion == "Formal"
        assert "minimalist" in parsed.style_vibes

    def test_query_5_goa_beach_girls_under_5000(self, parser):
        query = "Goa beach party vacation vibe for girls under 5000"
        parsed = parser.parse(query)
        assert parsed.budget_max == 5000.0
        assert parsed.gender == "Women"
        assert parsed.target_occasion in ["Beach", "Party"]

    def test_query_6_festive_diwali_max_6000(self, parser):
        query = "festive Diwali ethnic kurta set max 6000"
        parsed = parser.parse(query)
        assert parsed.budget_max == 6000.0
        assert parsed.target_occasion == "Ethnic"

    def test_query_7_streetwear_oversized_under_2500(self, parser):
        query = "streetwear oversized black outfit under 2500 rs"
        parsed = parser.parse(query)
        assert parsed.budget_max == 2500.0
        assert "streetwear" in parsed.style_vibes
        assert "oversized" in parsed.style_vibes
        assert "black" in parsed.colors

    def test_query_8_default_budget_and_gym_men(self, parser):
        query = "running gym athleisure set for men"
        parsed = parser.parse(query)
        assert parsed.budget_max == 5000.0  # default fallback
        assert parsed.gender == "Men"
        assert parsed.target_occasion == "Gym"


class TestSemanticSearchIndex:
    """Test suite for TF-IDF Semantic Search Index."""

    @pytest.fixture(scope="module")
    def search_index(self):
        from src.intent_parser import SemanticSearchIndex
        catalog_df = generate_catalog(num_items=500, seed=123)
        return SemanticSearchIndex(catalog_df)

    def test_search_top_k_and_columns(self, search_index):
        results = search_index.search("linen pastel summer shirt", top_k=10)
        assert len(results) == 10
        assert "relevance_score" in results.columns
        assert (results["relevance_score"] >= 0.0).all()
        assert (results["relevance_score"] <= 1.0).all()

    def test_category_filtering(self, search_index):
        results = search_index.search("casual sneakers", category="footwear", top_k=5)
        assert len(results) == 5
        assert (results["category"] == "footwear").all()

    def test_gender_filtering(self, search_index):
        results = search_index.search("formal office shirt", category="topwear", gender="Women", top_k=5)
        assert len(results) <= 5
        assert set(results["gender"].unique()).issubset({"Women", "Unisex"})


class TestBundlingEngine:
    """Test suite for OutfitBundlingOptimizer, Knapsack constraint satisfaction, and Cohesion."""

    @pytest.fixture(scope="module")
    def bundling_setup(self):
        from src.intent_parser import IntentParser, SemanticSearchIndex
        from src.bundling_engine import OutfitBundlingOptimizer
        catalog_df = generate_catalog(num_items=1000, seed=999)
        search_index = SemanticSearchIndex(catalog_df)
        parser = IntentParser()
        optimizer = OutfitBundlingOptimizer(search_index)
        return parser, optimizer

    def test_generates_bundle_with_distinct_categories(self, bundling_setup):
        parser, optimizer = bundling_setup
        intent = parser.parse("cocktail party look for women under 7000")
        bundle = optimizer.curate_bundle(intent)

        assert bundle is not None
        assert len(bundle.items) in [3, 4]
        categories = [item["category"] for item in bundle.items]
        # Verify no duplicate categories
        assert len(categories) == len(set(categories))
        assert "topwear" in categories
        assert "bottomwear" in categories
        assert "footwear" in categories

    def test_100_percent_budget_compliance_standard(self, bundling_setup):
        parser, optimizer = bundling_setup
        test_budgets = [4000, 5000, 6500, 8000]

        for b in test_budgets:
            intent = parser.parse(f"summer brunch outfit under {b}")
            bundle = optimizer.curate_bundle(intent)
            assert bundle.total_price <= b, f"Bundle price ₹{bundle.total_price} exceeded budget ₹{b}"
            assert bundle.is_budget_compliant is True

    def test_tight_budget_fallback_behavior(self, bundling_setup):
        parser, optimizer = bundling_setup
        # Tight budget where 4 pieces usually exceed ₹2,500
        intent = parser.parse("minimalist office wear under 2500")
        bundle = optimizer.curate_bundle(intent)

        assert bundle.total_price <= 2500
        assert bundle.is_budget_compliant is True
        # Verify fallback transparent message is set
        assert bundle.fallback_applied is True
        assert "2,500" in bundle.fallback_message or "limit" in bundle.fallback_message

    def test_stylist_rationale_generation(self, bundling_setup):
        parser, optimizer = bundling_setup
        intent = parser.parse("pastel beach vacation look under 4500")
        bundle = optimizer.curate_bundle(intent)

        assert isinstance(bundle.stylist_rationale, str)
        assert len(bundle.stylist_rationale) > 20
        assert "beach" in bundle.stylist_rationale.lower() or "pastel" in bundle.stylist_rationale.lower()

    def test_swap_item_functionality(self, bundling_setup):
        parser, optimizer = bundling_setup
        intent = parser.parse("casual weekend look for men under 6000")
        bundle = optimizer.curate_bundle(intent)

        # Pick topwear item to swap
        orig_top = next(i for i in bundle.items if i["category"] == "topwear")
        # Find an alternative topwear SKU
        top_skus = optimizer.search_index.catalog_df[
            (optimizer.search_index.catalog_df["category"] == "topwear") &
            (optimizer.search_index.catalog_df["sku_id"] != orig_top["sku_id"])
        ]
        new_sku = top_skus.iloc[0]["sku_id"]

        swapped_bundle = optimizer.swap_item(bundle, "topwear", new_sku)
        swapped_top = next(i for i in swapped_bundle.items if i["category"] == "topwear")

        assert swapped_top["sku_id"] == new_sku
        assert swapped_top["sku_id"] != orig_top["sku_id"]
        assert swapped_bundle.total_price == sum(i["price"] for i in swapped_bundle.items)


