"""ContextStyle: GenAI Intent-Based Styling & Outfit Bundling Engine.

Hugging Face Spaces Native Gradio Web Application.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.catalog_pipeline import load_catalog, generate_catalog, save_catalog
from src.intent_parser import IntentParser, SemanticSearchIndex
from src.bundling_engine import OutfitBundlingOptimizer, OutfitBundle
from src.analytics_simulator import CommercialImpactModel, ABExperimentSimulator

# Load Engine
try:
    catalog_df = load_catalog()
except Exception:
    catalog_df = generate_catalog(num_items=2500, seed=42)
    save_catalog(catalog_df)

search_index = SemanticSearchIndex(catalog_df)
intent_parser = IntentParser(default_budget=5000.0)
optimizer = OutfitBundlingOptimizer(search_index)
impact_model = CommercialImpactModel()
ab_simulator = ABExperimentSimulator(impact_model)

# Custom Myntra CSS
CUSTOM_CSS = """
.gradio-container {
    font-family: 'Assistant', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
.myntra-badge {
    background-color: #fff0f3;
    color: #ff3f6c;
    padding: 4px 12px;
    border-radius: 16px;
    font-weight: 700;
    font-size: 0.85rem;
    border: 1px solid #ffccd5;
    display: inline-block;
}
.bundle-card {
    background: #ffffff;
    border: 1px solid #eaeaec;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 4px 12px rgba(40,44,63,0.06);
}
.item-card {
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}
.item-price {
    font-size: 1.15rem;
    font-weight: 800;
    color: #282c3f;
}
.item-brand {
    font-size: 0.85rem;
    color: #ff3f6c;
    font-weight: 700;
}
"""

CAT_ICONS = {
    "topwear": "👔",
    "bottomwear": "👖",
    "footwear": "👟",
    "accessories": "🕶️"
}


def curate_outfit_ui(query, budget, gender, vibe):
    """Processes search query and returns outfit bundle HTML and details."""
    active_query = query if query and query.strip() else "casual stylish outfit"
    intent = intent_parser.parse(active_query)

    intent.budget_max = float(budget)
    if gender != "Auto-Detect / Unisex":
        intent.gender = gender
    if vibe != "Auto-Detect":
        intent.style_vibes = [vibe.lower()]

    bundle = optimizer.curate_bundle(intent)

    # Header HTML
    header_html = f"""
    <div style="background:#fff0f3; border:1px solid #ffccd5; border-radius:12px; padding:16px; margin-bottom:16px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:0.8rem; font-weight:800; color:#ff3f6c; text-transform:uppercase;">Curated Ensemble</span>
                <div style="font-size:1.8rem; font-weight:800; color:#282c3f;">
                    ₹{bundle.total_price:,} 
                    <span style="font-size:1.1rem; color:#94969f; text-decoration:line-through; font-weight:400; margin-left:8px;">₹{bundle.original_price:,}</span>
                    <span style="background:#03a685; color:white; padding:3px 8px; border-radius:4px; font-size:0.8rem; font-weight:700; margin-left:8px;">
                        SAVE ₹{bundle.savings_amount:,} (BUNDLE DEAL)
                    </span>
                </div>
            </div>
            <div style="text-align:right;">
                <span style="color:#03a685; font-weight:700; font-size:0.95rem;">✓ 100% Budget Compliant</span>
                <div style="font-size:0.8rem; color:#696e79;">{len(bundle.items)} coordinated pieces</div>
            </div>
        </div>
        <div style="display:flex; gap:8px; margin-top:12px; flex-wrap:wrap;">
            <span style="background:#fff; border:1px solid #ddd; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600;">
                🎯 Intent: {intent.clean_query.title() if intent.clean_query else 'Casual Look'}
            </span>
            <span style="background:#fff; border:1px solid #ddd; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600;">
                🏷️ Budget Cap: ₹{int(intent.budget_max):,}
            </span>
            <span style="background:#fff; border:1px solid #ddd; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600;">
                👤 Target: {intent.gender}
            </span>
            <span style="background:#fff; border:1px solid #ddd; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600;">
                🎨 Cohesion: {int(bundle.cohesion_score * 100)}% Harmony
            </span>
        </div>
    </div>
    """

    # Items Grid HTML
    items_cards = []
    for item in bundle.items:
        cat = item.get("category", "item")
        icon = CAT_ICONS.get(cat, "✨")
        items_cards.append(f"""
        <div style="flex:1; min-width:200px; background:#ffffff; border:1px solid #eaeaec; border-radius:12px; padding:14px; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
            <div style="font-size:0.75rem; font-weight:800; text-transform:uppercase; color:#7e818c; margin-bottom:6px;">{icon} {cat}</div>
            <div style="background:#f7f7f9; border-radius:8px; height:100px; display:flex; flex-direction:column; align-items:center; justify-content:center; margin-bottom:10px;">
                <span style="font-size:2.2rem;">{icon}</span>
                <span style="font-size:0.72rem; color:#7e818c; font-weight:600; margin-top:4px;">{item.get('color')} • {item.get('fabric')}</span>
            </div>
            <div style="font-size:0.82rem; font-weight:700; color:#ff3f6c;">{item.get('brand')}</div>
            <div style="font-size:0.92rem; font-weight:700; color:#282c3f; margin-bottom:8px; line-height:1.3;">{item.get('title')}</div>
            <div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid #f0f0f2; padding-top:8px;">
                <span style="font-size:1.1rem; font-weight:800; color:#282c3f;">₹{item.get('price'):,}</span>
                <span style="background:#fff; border:1px solid #d4e8e7; color:#14958f; padding:1px 6px; border-radius:3px; font-weight:700; font-size:0.78rem;">★ {item.get('rating')}</span>
            </div>
        </div>
        """)

    grid_html = f"""
    <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px;">
        {''.join(items_cards)}
    </div>
    """

    # Stylist Note HTML
    rationale_html = f"""
    <div style="background:#f9f9fb; border-left:4px solid #ff3f6c; border-radius:0 10px 10px 0; padding:14px 18px; margin-top:8px;">
        <div style="font-size:0.85rem; font-weight:800; color:#ff3f6c; text-transform:uppercase; margin-bottom:4px;">💡 AI Stylist Curation Note</div>
        <div style="font-size:0.95rem; color:#3e4152; line-height:1.5;">{bundle.stylist_rationale}</div>
    </div>
    """

    return header_html + grid_html + rationale_html


def simulate_business_metrics(sessions):
    """Calculates commercial metrics and returns charts."""
    impact = impact_model.calculate_annual_impact(monthly_search_sessions=int(sessions))
    ab_results = ab_simulator.simulate_30_day_experiment(daily_traffic=10000, seed=42)
    daily_df = ab_results["daily_df"]

    # Scorecards HTML
    scorecards_html = f"""
    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin-bottom:16px;">
        <div style="background:#fff; border:1px solid #eaeaec; border-top:4px solid #ff3f6c; border-radius:10px; padding:14px; text-align:center;">
            <div style="font-size:0.75rem; color:#7e818c; font-weight:700; text-transform:uppercase;">Annual GMV Uplift</div>
            <div style="font-size:1.6rem; font-weight:800; color:#282c3f;">₹{impact['annual_gmv_uplift']/1e7:.2f} Cr</div>
            <div style="font-size:0.75rem; color:#03a685; font-weight:700;">+{impact['annual_gmv_lift_pct']:.1f}% vs Lexical</div>
        </div>
        <div style="background:#fff; border:1px solid #eaeaec; border-top:4px solid #ff3f6c; border-radius:10px; padding:14px; text-align:center;">
            <div style="font-size:0.75rem; color:#7e818c; font-weight:700; text-transform:uppercase;">Average Order Value</div>
            <div style="font-size:1.6rem; font-weight:800; color:#282c3f;">₹{impact['variant_aov']:,.0f}</div>
            <div style="font-size:0.75rem; color:#03a685; font-weight:700;">+₹{impact['aov_absolute_lift']:,.0f} (+{impact['aov_lift_pct']:.1f}%)</div>
        </div>
        <div style="background:#fff; border:1px solid #eaeaec; border-top:4px solid #ff3f6c; border-radius:10px; padding:14px; text-align:center;">
            <div style="font-size:0.75rem; color:#7e818c; font-weight:700; text-transform:uppercase;">Attach Rate</div>
            <div style="font-size:1.6rem; font-weight:800; color:#282c3f;">{impact['variant_attach_rate']:.2f}x</div>
            <div style="font-size:0.75rem; color:#03a685; font-weight:700;">+{impact['attach_rate_lift_pct']:.1f}% Units / Order</div>
        </div>
        <div style="background:#fff; border:1px solid #eaeaec; border-top:4px solid #ff3f6c; border-radius:10px; padding:14px; text-align:center;">
            <div style="font-size:0.75rem; color:#7e818c; font-weight:700; text-transform:uppercase;">Zero-Result Reduction</div>
            <div style="font-size:1.6rem; font-weight:800; color:#282c3f;">8.5%</div>
            <div style="font-size:0.75rem; color:#03a685; font-weight:700;">-19.5% pts Churn Drop</div>
        </div>
    </div>
    """

    # Plotly AOV Line Chart
    fig_aov = go.Figure()
    fig_aov.add_trace(go.Scatter(
        x=daily_df["day"], y=daily_df["variant_aov"],
        mode="lines+markers", name="Variant (ContextStyle)",
        line=dict(color="#ff3f6c", width=3)
    ))
    fig_aov.add_trace(go.Scatter(
        x=daily_df["day"], y=daily_df["control_aov"],
        mode="lines+markers", name="Control (Lexical)",
        line=dict(color="#282c3f", width=2, dash="dash")
    ))
    fig_aov.update_layout(
        title="<b>30-Day Simulated A/B Test: Average Order Value (₹)</b>",
        xaxis_title="Experiment Day", yaxis_title="AOV (INR)",
        template="plotly_white", height=340
    )

    # Plotly Donut Chart
    fig_donut = px.pie(
        names=["Topwear", "Bottomwear", "Footwear", "Accessories"],
        values=[38, 32, 22, 8],
        hole=0.55,
        color_discrete_sequence=["#ff3f6c", "#282c3f", "#03a685", "#ffaa00"],
        title="<b>Multi-Item Bundle Category Share</b>"
    )
    fig_donut.update_layout(template="plotly_white", height=340)

    return scorecards_html, fig_aov, fig_donut


# Build Gradio Blocks Application
with gr.Blocks(title="ContextStyle | Myntra Storefront AI Stylist") as demo:
    gr.HTML("""
    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:3px solid #ff3f6c; padding-bottom:12px; margin-bottom:16px;">
        <div>
            <h1 style="margin:0; font-size:1.8rem; font-weight:800; color:#282c3f;">
                Context<span style="color:#ff3f6c;">Style</span>
            </h1>
            <p style="margin:0; font-size:0.9rem; color:#696e79;">GenAI Intent-Based Semantic Search & Outfit Bundling Engine</p>
        </div>
        <div style="text-align:right;">
            <span style="background:#fff0f3; color:#ff3f6c; padding:4px 10px; border-radius:14px; font-weight:700; font-size:0.8rem; border:1px solid #ffccd5;">
                Myntra Storefront CX Showcase
            </span>
        </div>
    </div>
    """)

    with gr.Tabs():
        # TAB 1: Customer View
        with gr.TabItem("🛍️ AI Stylist & Search Experience (Customer View)"):
            with gr.Row():
                with gr.Column(scale=4):
                    query_input = gr.Textbox(
                        label="Natural Language Fashion Search",
                        value="Pastel brunch outfit for humid weather under ₹3500",
                        placeholder="e.g. 'monsoon brunch pastel casual look under ₹3500'"
                    )
                with gr.Column(scale=1):
                    search_btn = gr.Button("✨ Curate Outfit", variant="primary")

            with gr.Row():
                gr.Markdown("**Quick Prompts:**")
                btn_p1 = gr.Button("🌸 Goa Beach Party (₹4k)", size="sm")
                btn_p2 = gr.Button("💼 Tech Offsite Smart Casual", size="sm")
                btn_p3 = gr.Button("☕ Monsoon Coffee Date (₹3k)", size="sm")
                btn_p4 = gr.Button("✨ Festive Diwali Ethnic (₹5k)", size="sm")

            with gr.Row():
                budget_slider = gr.Slider(minimum=1500, maximum=12000, value=3500, step=250, label="Total Outfit Budget (₹)")
                gender_select = gr.Radio(["Auto-Detect / Unisex", "Women", "Men"], value="Auto-Detect / Unisex", label="Target Demographic")
                vibe_select = gr.Dropdown(["Auto-Detect", "Pastel", "Minimalist", "Streetwear", "Classic", "Vibrant"], value="Auto-Detect", label="Aesthetic Filter")

            outfit_output = gr.HTML()

            # Event listeners
            search_btn.click(
                fn=curate_outfit_ui,
                inputs=[query_input, budget_slider, gender_select, vibe_select],
                outputs=outfit_output
            )
            btn_p1.click(lambda: ("Goa beach party pastel summer vacation vibe under 4000", 4000), None, [query_input, budget_slider]).then(
                fn=curate_outfit_ui, inputs=[query_input, budget_slider, gender_select, vibe_select], outputs=outfit_output
            )
            btn_p2.click(lambda: ("smart casual attire for corporate dinner and tech offsite for men under 4500", 4500), None, [query_input, budget_slider]).then(
                fn=curate_outfit_ui, inputs=[query_input, budget_slider, gender_select, vibe_select], outputs=outfit_output
            )
            btn_p3.click(lambda: ("minimalist monsoon coffee date breathable outfit under 3000", 3000), None, [query_input, budget_slider]).then(
                fn=curate_outfit_ui, inputs=[query_input, budget_slider, gender_select, vibe_select], outputs=outfit_output
            )
            btn_p4.click(lambda: ("festive Diwali ethnic royal kurta celebration look under 5000", 5000), None, [query_input, budget_slider]).then(
                fn=curate_outfit_ui, inputs=[query_input, budget_slider, gender_select, vibe_select], outputs=outfit_output
            )

            # Trigger initial render
            demo.load(
                fn=curate_outfit_ui,
                inputs=[query_input, budget_slider, gender_select, vibe_select],
                outputs=outfit_output
            )

        # TAB 2: PM Business Dashboard
        with gr.TabItem("📊 Storefront Business & A/B Dashboard (PM View)"):
            sessions_slider = gr.Slider(minimum=50000, maximum=1000000, value=250000, step=25000, label="Monthly Search Sessions")
            scorecards_display = gr.HTML()
            with gr.Row():
                aov_chart = gr.Plot(label="AOV Lift Over 30 Days")
                donut_chart = gr.Plot(label="Category Basket Share")

            sessions_slider.change(
                fn=simulate_business_metrics,
                inputs=[sessions_slider],
                outputs=[scorecards_display, aov_chart, donut_chart]
            )
            demo.load(
                fn=simulate_business_metrics,
                inputs=[sessions_slider],
                outputs=[scorecards_display, aov_chart, donut_chart]
            )

        # TAB 3: PRD Documentation
        with gr.TabItem("📑 PRD & Architecture"):
            with open(ROOT_DIR / "docs" / "PRD.md", "r", encoding="utf-8") as f:
                prd_text = f.read()
            gr.Markdown(prd_text)

if __name__ == "__main__":
    demo.launch()
