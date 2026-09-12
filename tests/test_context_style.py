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
