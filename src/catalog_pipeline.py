"""Catalog Pipeline: Generates and manages the synthetic fashion catalog for ContextStyle.

Produces 2,500 diverse, realistic fashion items across 4 categories:
- topwear
- bottomwear
- footwear
- accessories
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Seeds for reproducibility
RANDOM_SEED = 42

CATEGORIES_CONFIG = {
    "topwear": {
        "sub_categories": {
            "Shirts": {"genders": ["Men", "Women"], "fabrics": ["Cotton", "Linen", "Silk"], "min_p": 699, "max_p": 2999},
            "T-Shirts": {"genders": ["Men", "Women", "Unisex"], "fabrics": ["Cotton", "Polyester"], "min_p": 399, "max_p": 1599},
            "Kurtas": {"genders": ["Men", "Women"], "fabrics": ["Cotton", "Silk", "Linen"], "min_p": 799, "max_p": 3499},
            "Blouses": {"genders": ["Women"], "fabrics": ["Silk", "Polyester", "Cotton"], "min_p": 599, "max_p": 2499},
            "Crop Tops": {"genders": ["Women"], "fabrics": ["Cotton", "Polyester"], "min_p": 449, "max_p": 1799},
            "Blazers": {"genders": ["Men", "Women"], "fabrics": ["Linen", "Polyester", "Cotton"], "min_p": 2499, "max_p": 8999},
        }
    },
    "bottomwear": {
        "sub_categories": {
            "Jeans": {"genders": ["Men", "Women"], "fabrics": ["Denim", "Cotton"], "min_p": 999, "max_p": 4499},
            "Trousers": {"genders": ["Men", "Women"], "fabrics": ["Cotton", "Linen", "Polyester"], "min_p": 899, "max_p": 3999},
            "Chinos": {"genders": ["Men", "Women"], "fabrics": ["Cotton", "Linen"], "min_p": 899, "max_p": 3299},
            "Skirts": {"genders": ["Women"], "fabrics": ["Cotton", "Polyester", "Silk"], "min_p": 699, "max_p": 2799},
            "Palazzos": {"genders": ["Women"], "fabrics": ["Cotton", "Silk", "Polyester"], "min_p": 599, "max_p": 2199},
        }
    },
    "footwear": {
        "sub_categories": {
            "Sneakers": {"genders": ["Men", "Women", "Unisex"], "fabrics": ["Leather", "Polyester", "Cotton"], "min_p": 1299, "max_p": 6999},
            "Loafers": {"genders": ["Men", "Women"], "fabrics": ["Leather", "Polyester"], "min_p": 1199, "max_p": 4999},
            "Heels": {"genders": ["Women"], "fabrics": ["Leather", "Polyester"], "min_p": 1099, "max_p": 5499},
            "Flats": {"genders": ["Women"], "fabrics": ["Leather", "Cotton"], "min_p": 499, "max_p": 2299},
            "Boots": {"genders": ["Men", "Women"], "fabrics": ["Leather", "Polyester"], "min_p": 1899, "max_p": 7999},
        }
    },
    "accessories": {
        "sub_categories": {
            "Watches": {"genders": ["Men", "Women", "Unisex"], "fabrics": ["Leather", "Polyester"], "min_p": 1499, "max_p": 8999},
            "Bags": {"genders": ["Women", "Men", "Unisex"], "fabrics": ["Leather", "Polyester", "Cotton"], "min_p": 799, "max_p": 4999},
            "Sunglasses": {"genders": ["Men", "Women", "Unisex"], "fabrics": ["Polyester"], "min_p": 499, "max_p": 3499},
            "Belts": {"genders": ["Men", "Women"], "fabrics": ["Leather", "Polyester"], "min_p": 399, "max_p": 1999},
            "Earrings": {"genders": ["Women"], "fabrics": ["Silk", "Polyester"], "min_p": 399, "max_p": 1799},
        }
    },
}

BRANDS_BY_GENDER = {
    "Men": ["Roadster", "HRX by Hrithik Roshan", "Mast & Harbour", "WROGN", "HIGHLANDER", "Tommy Hilfiger", "Louis Philippe", "Peter England", "Puma", "Nike", "Fossil", "Titan"],
    "Women": ["Tokyo Talkies", "DressBerry", "Anouk", "Sangria", "Mango", "ONLY", "VERO MODA", "Carlton London", "Forever 21", "Fossil", "Bata", "Puma"],
    "Unisex": ["Puma", "Nike", "Fastrack", "Fossil", "Casio", "Converse", "Vans", "Roadster", "Ray-Ban", "Lavie Sport"]
}

COLORS = [
    "Black", "White", "Navy Blue", "Olive Green", "Beige", "Pastel Pink",
    "Lavender", "Mustard Yellow", "Charcoal Grey", "Tan Brown", "Mint Green",
    "Maroon", "Sky Blue", "Terracotta", "Burgundy", "Cream"
]

OCCASIONS = ["Casual", "Formal", "Party", "Beach", "Ethnic", "Gym"]

STYLES = ["Minimalist", "Vibrant", "Pastel", "Streetwear", "Classic"]

STYLE_DESCRIPTORS = {
    "Minimalist": "clean tailored silhouette with subtle contemporary accents",
    "Vibrant": "eye-catching high-energy color palette suited for expressive moods",
    "Pastel": "soft muted hues providing a calm, breezy and sophisticated aesthetic",
    "Streetwear": "urban relaxed drape engineered with durable modern textures",
    "Classic": "timeless quintessential design crafted for versatile everyday elegance"
}


def _generate_item_description(brand: str, sub_cat: str, color: str, fabric: str,
                               occasions: List[str], styles: List[str], gender: str) -> str:
    """Generates rich semantic product description for vector embedding."""
    primary_style = styles[0]
    style_desc = STYLE_DESCRIPTORS.get(primary_style, "versatile modern fit")
    occ_str = ", ".join(occasions)
    gender_str = f"for {gender.lower()}" if gender != "Unisex" else "for all genders"

    return (
        f"{brand} {color.lower()} {sub_cat.lower()} crafted from premium {fabric.lower()} {gender_str}. "
        f"Features a {style_desc}. Ideal for {occ_str.lower()} occasions. "
        f"Styled in a {primary_style.lower()} aesthetic, offering both all-day comfort and trend-forward appeal."
    )


def generate_catalog(num_items: int = 2500, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generates a synthetic fashion catalog of specified length (default 2,500 items).

    Guarantees balanced representation across topwear, bottomwear, footwear, and accessories.
    """
    random.seed(seed)
    np.random.seed(seed)

    categories = list(CATEGORIES_CONFIG.keys())
    items_per_cat = num_items // len(categories)
    remainder = num_items % len(categories)

    records = []
    sku_counter = 1000

    for cat_idx, cat in enumerate(categories):
        cat_items_count = items_per_cat + (1 if cat_idx < remainder else 0)
        sub_cats_dict = CATEGORIES_CONFIG[cat]["sub_categories"]
        sub_cat_names = list(sub_cats_dict.keys())

        for _ in range(cat_items_count):
            sku_counter += 1
            sku_id = f"SKU-{cat[:3].upper()}-{sku_counter}"

            sub_cat = random.choice(sub_cat_names)
            cfg = sub_cats_dict[sub_cat]

            gender = random.choice(cfg["genders"])
            brand = random.choice(BRANDS_BY_GENDER[gender])
            color = random.choice(COLORS)
            fabric = random.choice(cfg["fabrics"])

            # Price within subcategory limits, rounded to 99 or 49
            base_price = random.randint(cfg["min_p"], cfg["max_p"])
            price = round(base_price / 50.0) * 50 - 1  # e.g., 999, 1249, 1499
            if price < 399:
                price = 399
            elif price > 8999:
                price = 8999

            # Realistic rating: normal distribution truncated between 3.2 and 4.9
            rating = round(np.clip(np.random.normal(loc=4.2, scale=0.35), 3.2, 4.9), 1)

            # Assign 1-2 occasions
            num_occ = random.choices([1, 2], weights=[0.6, 0.4])[0]
            occasions = random.sample(OCCASIONS, k=num_occ)

            # Assign 1-2 style tags
            num_style = random.choices([1, 2], weights=[0.7, 0.3])[0]
            styles = random.sample(STYLES, k=num_style)

            # Title matching Myntra style: [Brand] [Color] [Gender] [Fabric] [SubCategory]
            gender_title_part = f"{gender}'s" if gender in ["Men", "Women"] else "Unisex"
            title = f"{brand} {color} {gender_title_part} {fabric} {sub_cat}"

            description = _generate_item_description(
                brand=brand,
                sub_cat=sub_cat,
                color=color,
                fabric=fabric,
                occasions=occasions,
                styles=styles,
                gender=gender
            )

            records.append({
                "sku_id": sku_id,
                "title": title,
                "category": cat,
                "sub_category": sub_cat,
                "brand": brand,
                "gender": gender,
                "price": int(price),
                "rating": float(rating),
                "occasion_tags": ";".join(occasions),
                "style_tags": ";".join(styles),
                "color": color,
                "fabric": fabric,
                "description": description
            })

    df = pd.DataFrame(records)
    # Shuffle catalog
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def save_catalog(df: pd.DataFrame, base_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Saves catalog to both Parquet and CSV in data/processed."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    else:
        base_dir = Path(base_dir)

    base_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = base_dir / "catalog.parquet"
    csv_path = base_dir / "catalog.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    return {
        "parquet": parquet_path,
        "csv": csv_path
    }


def load_catalog(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Loads catalog from parquet if available, otherwise csv."""
    if filepath is None:
        processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
        parquet_path = processed_dir / "catalog.parquet"
        csv_path = processed_dir / "catalog.csv"

        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        elif csv_path.exists():
            return pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"No catalog found at {parquet_path} or {csv_path}. Run catalog_pipeline first.")
    else:
        filepath = Path(filepath)
        if filepath.suffix == ".parquet":
            return pd.read_parquet(filepath)
        return pd.read_csv(filepath)


if __name__ == "__main__":
    print("Generating ContextStyle fashion catalog...")
    catalog_df = generate_catalog(num_items=2500, seed=RANDOM_SEED)
    saved_paths = save_catalog(catalog_df)
    print(f"Catalog generated successfully! Saved {len(catalog_df)} items to:")
    print(f" - {saved_paths['parquet']}")
    print(f" - {saved_paths['csv']}")
    print("\nCategory distribution:")
    print(catalog_df["category"].value_counts())
