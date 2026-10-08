from html import escape
from pathlib import Path

import joblib
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIRECTORY = PROJECT_ROOT / "models"
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"

ARTIFACT_PATH = (
    MODELS_DIRECTORY
    / "hybrid_recommender.joblib"
)

CATALOGUE_PATH = (
    MODELS_DIRECTORY
    / "product_catalogue.csv.gz"
)

MODEL_COMPARISON_PATH = (
    REPORTS_DIRECTORY
    / "model_comparison.csv"
)

HYBRID_VALIDATION_PATH = (
    REPORTS_DIRECTORY
    / "hybrid_validation_results.csv"
)


st.set_page_config(
    page_title="Saidia Game Recommender",
    page_icon="🎮",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .app-subtitle {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 1rem;
    }

    .product-title {
        min-height: 3.8rem;
        max-height: 3.8rem;
        overflow: hidden;
        font-size: 0.98rem;
        font-weight: 700;
        line-height: 1.25rem;
        margin-top: 0.5rem;
        margin-bottom: 0.4rem;
    }

    .product-price {
        min-height: 2.2rem;
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }

    .product-category {
        min-height: 1.7rem;
        color: #9ca3af;
        font-size: 0.82rem;
    }

    [data-testid="stImage"] img {
        height: 165px;
        object-fit: contain;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------

@st.cache_resource
def load_artifacts():
    return joblib.load(
        ARTIFACT_PATH
    )


@st.cache_data
def load_catalogue():
    product_catalogue = pd.read_csv(
        CATALOGUE_PATH
    )

    product_catalogue["title"] = (
        product_catalogue["title"]
        .fillna("Untitled product")
        .astype(str)
    )

    product_catalogue["main_category"] = (
        product_catalogue["main_category"]
        .fillna("Category unavailable")
        .astype(str)
    )

    return product_catalogue


@st.cache_data
def load_model_comparison():
    if not MODEL_COMPARISON_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        MODEL_COMPARISON_PATH
    )


@st.cache_data
def load_hybrid_validation_results():
    if not HYBRID_VALIDATION_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        HYBRID_VALIDATION_PATH
    )


if not ARTIFACT_PATH.exists():
    st.error(
        f"Model artifact not found: {ARTIFACT_PATH}"
    )
    st.stop()

if not CATALOGUE_PATH.exists():
    st.error(
        f"Product catalogue not found: {CATALOGUE_PATH}"
    )
    st.stop()


artifacts = load_artifacts()
catalogue = load_catalogue()
model_comparison = load_model_comparison()
hybrid_validation_results = (
    load_hybrid_validation_results()
)

catalogue_by_id = (
    catalogue
    .drop_duplicates(
        subset="parent_asin",
        keep="first",
    )
    .set_index("parent_asin")
)

title_lookup = (
    catalogue_by_id["title"]
    .to_dict()
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "selected_product_ids" not in st.session_state:
    st.session_state.selected_product_ids = set()

if "focused_product_id" not in st.session_state:
    st.session_state.focused_product_id = None

if "generated_recommendations" not in st.session_state:
    st.session_state.generated_recommendations = None


# ---------------------------------------------------------
# Navigation and selection callbacks
# ---------------------------------------------------------

def add_product_to_likes(product_id):
    selected = set(
        st.session_state.selected_product_ids
    )

    selected.add(product_id)

    st.session_state.selected_product_ids = (
        selected
    )

    st.session_state.generated_recommendations = None


def remove_product_from_likes(product_id):
    selected = set(
        st.session_state.selected_product_ids
    )

    selected.discard(product_id)

    st.session_state.selected_product_ids = (
        selected
    )

    st.session_state.generated_recommendations = None


def clear_likes():
    st.session_state.selected_product_ids = set()
    st.session_state.generated_recommendations = None


def open_related_products(product_id):
    st.session_state.focused_product_id = (
        product_id
    )


def return_to_catalogue():
    st.session_state.focused_product_id = None


def get_title(product_id):
    return title_lookup.get(
        product_id,
        product_id,
    )


# ---------------------------------------------------------
# Personalized hybrid recommendation
# ---------------------------------------------------------

def create_recommendations(
    history_products,
    number_of_recommendations=10,
):
    seen_products = set(
        history_products
    )

    collaborative_scores = {}
    content_scores = {}

    collaborative_sources = {}
    content_sources = {}

    for seen_product_id in seen_products:
        for candidate_id, similarity in artifacts[
            "item_neighbors"
        ].get(
            seen_product_id,
            [],
        ):
            if candidate_id in seen_products:
                continue

            collaborative_scores[candidate_id] = (
                collaborative_scores.get(
                    candidate_id,
                    0.0,
                )
                + similarity
            )

            previous_source = collaborative_sources.get(
                candidate_id,
                (0.0, None),
            )

            if similarity > previous_source[0]:
                collaborative_sources[candidate_id] = (
                    similarity,
                    seen_product_id,
                )

        for candidate_id, similarity in artifacts[
            "content_neighbors"
        ].get(
            seen_product_id,
            [],
        ):
            if candidate_id in seen_products:
                continue

            content_scores[candidate_id] = (
                content_scores.get(
                    candidate_id,
                    0.0,
                )
                + similarity
            )

            previous_source = content_sources.get(
                candidate_id,
                (0.0, None),
            )

            if similarity > previous_source[0]:
                content_sources[candidate_id] = (
                    similarity,
                    seen_product_id,
                )

    maximum_collaborative_score = max(
        collaborative_scores.values(),
        default=0.0,
    )

    maximum_content_score = max(
        content_scores.values(),
        default=0.0,
    )

    if maximum_collaborative_score <= 0:
        maximum_collaborative_score = 1.0

    if maximum_content_score <= 0:
        maximum_content_score = 1.0

    candidate_ids = (
        set(collaborative_scores)
        | set(content_scores)
    )

    scored_candidates = []

    for candidate_id in candidate_ids:
        collaborative_score = (
            collaborative_scores.get(
                candidate_id,
                0.0,
            )
            / maximum_collaborative_score
        )

        content_score = (
            content_scores.get(
                candidate_id,
                0.0,
            )
            / maximum_content_score
        )

        hybrid_score = (
            artifacts["collaborative_weight"]
            * collaborative_score
            + artifacts["content_weight"]
            * content_score
        )

        collaborative_source = (
            collaborative_sources.get(
                candidate_id,
                (0.0, None),
            )[1]
        )

        content_source = (
            content_sources.get(
                candidate_id,
                (0.0, None),
            )[1]
        )

        if (
            collaborative_score > 0
            and content_score > 0
        ):
            explanation = (
                "Supported by both user-behaviour patterns "
                "and product-metadata similarity."
            )

        elif collaborative_score > 0:
            explanation = (
                "Users who liked related products also "
                "interacted positively with this product."
            )

        elif content_score > 0:
            explanation = (
                "Its title, categories, description or features "
                "are similar to products you selected."
            )

        else:
            explanation = (
                "Included as a popular fallback product."
            )

        strongest_source = (
            collaborative_source
            or content_source
        )

        scored_candidates.append(
            {
                "product_id": candidate_id,
                "hybrid_score": float(
                    hybrid_score
                ),
                "collaborative_score": float(
                    collaborative_score
                ),
                "content_score": float(
                    content_score
                ),
                "source_product_id": strongest_source,
                "explanation": explanation,
            }
        )

    scored_candidates = sorted(
        scored_candidates,
        key=lambda result: (
            -result["hybrid_score"],
            result["product_id"],
        ),
    )

    recommendations = scored_candidates[
        :number_of_recommendations
    ]

    recommended_ids = {
        result["product_id"]
        for result in recommendations
    }

    if len(recommendations) < number_of_recommendations:
        for product_id in artifacts[
            "popularity_order"
        ]:
            if product_id in seen_products:
                continue

            if product_id in recommended_ids:
                continue

            recommendations.append(
                {
                    "product_id": product_id,
                    "hybrid_score": 0.0,
                    "collaborative_score": 0.0,
                    "content_score": 0.0,
                    "source_product_id": None,
                    "explanation": (
                        "Included as a popular fallback product."
                    ),
                }
            )

            recommended_ids.add(
                product_id
            )

            if len(recommendations) == number_of_recommendations:
                break

    return recommendations


# ---------------------------------------------------------
# Product card
# ---------------------------------------------------------

def display_product_card(
    product_id,
    context,
    show_explanation=None,
):
    if product_id not in catalogue_by_id.index:
        return

    product = catalogue_by_id.loc[
        product_id
    ]

    title = product["title"]
    image_url = product.get(
        "image_url"
    )
    price = product.get(
        "numeric_price"
    )

    already_liked = (
        product_id
        in st.session_state.selected_product_ids
    )

    with st.container(
        border=True,
        height=500,
    ):
        if pd.notna(image_url) and image_url:
            st.image(
                image_url,
                width="stretch",
            )
        else:
            st.markdown("## 🎮")
            st.caption(
                "Image unavailable"
            )

        st.markdown(
            (
                f'<div class="product-title">'
                f"{escape(title)}"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )

        if pd.notna(price):
            price_text = (
                f"${float(price):,.2f}"
            )
        else:
            price_text = (
                "Price unavailable"
            )

        st.markdown(
            (
                f'<div class="product-price">'
                f"{escape(price_text)}"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            (
                f'<div class="product-category">'
                f"{escape(product['main_category'])}"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )

        st.button(
            "View related",
            key=(
                f"related_{context}_{product_id}"
            ),
            on_click=open_related_products,
            args=(product_id,),
            width="stretch",
        )

        if already_liked:
            st.button(
                "✓ Liked — remove",
                key=(
                    f"like_{context}_{product_id}"
                ),
                on_click=remove_product_from_likes,
                args=(product_id,),
                width="stretch",
            )
        else:
            st.button(
                "Add to my likes",
                key=(
                    f"like_{context}_{product_id}"
                ),
                on_click=add_product_to_likes,
                args=(product_id,),
                type="primary",
                width="stretch",
            )

        if show_explanation:
            with st.expander(
                "Why recommended?"
            ):
                st.write(
                    show_explanation[
                        "explanation"
                    ]
                )

                source_product_id = (
                    show_explanation[
                        "source_product_id"
                    ]
                )

                if source_product_id:
                    st.caption(
                        "Strongest connection: "
                        f"{get_title(source_product_id)}"
                    )

                st.write(
                    "Collaborative evidence: "
                    f"{show_explanation['collaborative_score']:.3f}"
                )

                st.write(
                    "Content evidence: "
                    f"{show_explanation['content_score']:.3f}"
                )


def display_product_grid(
    product_ids,
    context,
    number_of_columns=4,
    explanations=None,
):
    product_ids = list(
        product_ids
    )

    for row_start in range(
        0,
        len(product_ids),
        number_of_columns,
    ):
        row_product_ids = product_ids[
            row_start:
            row_start + number_of_columns
        ]

        columns = st.columns(
            number_of_columns
        )

        for column_number, product_id in enumerate(
            row_product_ids
        ):
            with columns[column_number]:
                product_explanation = None

                if explanations:
                    product_explanation = (
                        explanations.get(
                            product_id
                        )
                    )

                display_product_card(
                    product_id,
                    context=(
                        f"{context}_{row_start}"
                    ),
                    show_explanation=product_explanation,
                )


# ---------------------------------------------------------
# Focused product page
# ---------------------------------------------------------

def display_focused_product(
    product_id,
):
    if product_id not in catalogue_by_id.index:
        st.warning(
            "The selected product is unavailable."
        )
        return

    product = catalogue_by_id.loc[
        product_id
    ]

    navigation_column, likes_column = st.columns(
        [4, 1]
    )

    with navigation_column:
        st.button(
            "← Back to catalogue",
            on_click=return_to_catalogue,
        )

    with likes_column:
        st.metric(
            "My likes",
            len(
                st.session_state.selected_product_ids
            ),
        )

    image_column, details_column = st.columns(
        [1, 2]
    )

    with image_column:
        image_url = product.get(
            "image_url"
        )

        if pd.notna(image_url) and image_url:
            st.image(
                image_url,
                width="stretch",
            )
        else:
            st.markdown("# 🎮")

    with details_column:
        st.title(
            product["title"]
        )

        st.write(
            f"**Category:** "
            f"{product['main_category']}"
        )

        price = product.get(
            "numeric_price"
        )

        if pd.notna(price):
            st.markdown(
                f"## ${float(price):,.2f}"
            )
        else:
            st.markdown(
                "### Price unavailable"
            )

        if (
            product_id
            in st.session_state.selected_product_ids
        ):
            st.button(
                "✓ Remove from my likes",
                key=f"focused_like_{product_id}",
                on_click=remove_product_from_likes,
                args=(product_id,),
                width="stretch",
            )
        else:
            st.button(
                "Add to my likes",
                key=f"focused_like_{product_id}",
                on_click=add_product_to_likes,
                args=(product_id,),
                type="primary",
                width="stretch",
            )

    collaborative_neighbors = artifacts[
        "item_neighbors"
    ].get(
        product_id,
        [],
    )

    collaborative_product_ids = [
        neighbor_id
        for neighbor_id, similarity
        in collaborative_neighbors
        if neighbor_id != product_id
    ][:8]

    content_neighbors = artifacts[
        "content_neighbors"
    ].get(
        product_id,
        [],
    )

    content_product_ids = [
        neighbor_id
        for neighbor_id, similarity
        in content_neighbors
        if (
            neighbor_id != product_id
            and neighbor_id
            not in collaborative_product_ids
        )
    ][:8]

    st.divider()

    st.subheader(
        "Users also liked"
    )

    st.caption(
        "Products connected through overlapping positive "
        "user behaviour."
    )

    if collaborative_product_ids:
        display_product_grid(
            collaborative_product_ids,
            context=(
                f"collaborative_{product_id}"
            ),
            number_of_columns=4,
        )
    else:
        st.info(
            "This product has insufficient collaborative history."
        )

    st.subheader(
        "Similar products"
    )

    st.caption(
        "Products with similar titles, categories, "
        "descriptions or features."
    )

    if content_product_ids:
        display_product_grid(
            content_product_ids,
            context=f"content_{product_id}",
            number_of_columns=4,
        )
    else:
        st.info(
            "No content-similar products were found."
        )


# ---------------------------------------------------------
# My likes and personalized recommendations
# ---------------------------------------------------------

def display_likes_and_recommendations():
    selected_products = set(
        st.session_state.selected_product_ids
    )

    st.divider()

    if not selected_products:
        st.info(
            "Add at least one product to My likes. "
            "You can add products from the catalogue or "
            "from a related-product page."
        )
        return

    heading_column, clear_column = st.columns(
        [5, 1]
    )

    with heading_column:
        st.subheader(
            f"My likes ({len(selected_products)})"
        )

    with clear_column:
        st.button(
            "Clear all",
            on_click=clear_likes,
            width="stretch",
        )

    ordered_selected_products = sorted(
        selected_products,
        key=lambda product_id: get_title(
            product_id
        ).lower(),
    )

    display_product_grid(
        ordered_selected_products,
        context="selected",
        number_of_columns=4,
    )

    if st.button(
        "Get my 10 personalized recommendations",
        type="primary",
        width="stretch",
    ):
        st.session_state.generated_recommendations = (
            create_recommendations(
                selected_products,
                number_of_recommendations=10,
            )
        )

    recommendations = (
        st.session_state.generated_recommendations
    )

    if recommendations:
        st.divider()

        st.subheader(
            "Recommended for you"
        )

        st.caption(
            "Ranked using 70% collaborative evidence "
            "and 30% content evidence."
        )

        recommendation_ids = [
            result["product_id"]
            for result in recommendations
        ]

        explanation_lookup = {
            result["product_id"]: result
            for result in recommendations
        }

        display_product_grid(
            recommendation_ids,
            context="personalized",
            number_of_columns=5,
            explanations=explanation_lookup,
        )


# ---------------------------------------------------------
# Header and tabs
# ---------------------------------------------------------

st.title(
    "🎮 Saidia Game Recommender"
)

st.markdown(
    """
    <div class="app-subtitle">
        Explore related products, build a likes profile and receive
        ten personalized recommendations.
    </div>
    """,
    unsafe_allow_html=True,
)


recommendation_tab, evidence_tab, methodology_tab = st.tabs(
    [
        "Shop and discover",
        "Model evidence",
        "How it works",
    ]
)


# ---------------------------------------------------------
# Shop and discover tab
# ---------------------------------------------------------

with recommendation_tab:
    focused_product_id = (
        st.session_state.focused_product_id
    )

    if focused_product_id:
        display_focused_product(
            focused_product_id
        )

    else:
        st.subheader(
            "Browse products"
        )

        st.write(
            "Search the catalogue, view related products, "
            "or add products directly to My likes."
        )

        search_column, page_column = st.columns(
            [5, 1]
        )

        with search_column:
            product_search = st.text_input(
                "Search products",
                placeholder=(
                    "Search for FIFA, Nintendo, PlayStation, "
                    "Xbox, headset..."
                ),
            )

        popularity_rank = {
            product_id: rank
            for rank, product_id in enumerate(
                artifacts["popularity_order"]
            )
        }

        browse_catalogue = (
            catalogue.copy()
        )

        browse_catalogue[
            "popularity_rank"
        ] = browse_catalogue[
            "parent_asin"
        ].map(
            popularity_rank
        )

        browse_catalogue[
            "popularity_rank"
        ] = browse_catalogue[
            "popularity_rank"
        ].fillna(
            len(popularity_rank) + 1
        )

        cleaned_search = (
            product_search.strip()
        )

        if cleaned_search:
            search_mask = (
                browse_catalogue["title"]
                .str.contains(
                    cleaned_search,
                    case=False,
                    na=False,
                    regex=False,
                )
            )

            browse_catalogue = (
                browse_catalogue[
                    search_mask
                ]
            )

        browse_catalogue = (
            browse_catalogue
            .sort_values(
                [
                    "popularity_rank",
                    "title",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        products_per_page = 12

        maximum_page = max(
            1,
            int(
                np.ceil(
                    len(browse_catalogue)
                    / products_per_page
                )
            ),
        )

        with page_column:
            selected_page = st.number_input(
                "Page",
                min_value=1,
                max_value=maximum_page,
                value=1,
                step=1,
            )

        page_start = (
            int(selected_page) - 1
        ) * products_per_page

        page_end = (
            page_start
            + products_per_page
        )

        displayed_products = (
            browse_catalogue.iloc[
                page_start:page_end
            ]
        )

        st.caption(
            f"{len(browse_catalogue):,} products found"
        )

        if displayed_products.empty:
            st.info(
                "No products matched your search."
            )
        else:
            display_product_grid(
                displayed_products[
                    "parent_asin"
                ].tolist(),
                context="catalogue",
                number_of_columns=4,
            )

    display_likes_and_recommendations()


# ---------------------------------------------------------
# Evidence tab
# ---------------------------------------------------------

with evidence_tab:
    st.subheader(
        "Offline evaluation evidence"
    )

    st.info(
        "This is a static model report, not a live chart of your "
        "current selections. Before the app was built, each model "
        "was tested on historical users by hiding a later product "
        "they liked and checking whether it appeared in their top "
        "10 recommendations. Your selections only change the "
        "recommendations in Shop and discover."
    )

    metric_columns = st.columns(
        4
    )

    metric_columns[0].metric(
        "Test Hit Rate@10",
        "5.64%",
    )

    metric_columns[1].metric(
        "Test NDCG@10",
        "3.23%",
    )

    metric_columns[2].metric(
        "Catalogue coverage",
        "97.05%",
    )

    metric_columns[3].metric(
        "Test users",
        "88,124",
    )

    st.write(
        "Validation selected the 70% collaborative and "
        "30% content weighting before the untouched test "
        "targets were evaluated."
    )

    st.markdown(
        """
        **What the headline results mean:**

        - **Hit Rate@10 — 5.64%:** the hidden liked product appeared
          in the top 10 for about 1 in every 18 test users.
        - **NDCG@10 — 3.23%:** a ranking score that gives more credit
          when the hidden product appears nearer the top of the list.
        - **Catalogue coverage — 97.05%:** across all test users, the
          model recommended products from almost the entire eligible
          catalogue instead of repeatedly showing only popular items.
        """
    )

    if not model_comparison.empty:
        raw_comparison = (
            model_comparison.copy()
        )

        validation_comparison = (
            raw_comparison[
                raw_comparison[
                    "evaluation_split"
                ].eq("validation")
            ]
            .copy()
        )

        validation_comparison[
            "Model"
        ] = validation_comparison[
            "model"
        ].replace(
            {
                "Collaborative + popularity fallback": (
                    "Collaborative"
                ),
                "Hybrid 70/30": "Hybrid",
                "Popularity baseline": "Popularity",
            }
        )

        st.markdown(
            "### Validation ranking quality"
        )

        st.caption(
            "Higher is better. All three models are compared "
            "on the same chronological validation targets."
        )

        ranking_quality = (
            validation_comparison[
                [
                    "Model",
                    "hit_rate_at_10",
                    "mrr_at_10",
                    "ndcg_at_10",
                ]
            ]
            .rename(
                columns={
                    "hit_rate_at_10": "Hit Rate@10",
                    "mrr_at_10": "MRR@10",
                    "ndcg_at_10": "NDCG@10",
                }
            )
            .melt(
                id_vars="Model",
                var_name="Metric",
                value_name="Score",
            )
        )

        ranking_quality["Score (%)"] = (
            ranking_quality["Score"]
            * 100
        )

        ranking_quality["Value label"] = (
            ranking_quality["Score (%)"]
            .map(lambda value: f"{value:.2f}%")
        )

        model_order = [
            "Popularity",
            "Collaborative",
            "Hybrid",
        ]

        ranking_bars = (
            alt.Chart(ranking_quality)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Model:N",
                    sort=model_order,
                    title=None,
                    axis=alt.Axis(
                        labelAngle=0,
                    ),
                ),
                xOffset=alt.XOffset(
                    "Metric:N",
                ),
                y=alt.Y(
                    "Score (%):Q",
                    title=None,
                    axis=None,
                    scale=alt.Scale(
                        domain=[0, 6.7],
                    ),
                ),
                color=alt.Color(
                    "Metric:N",
                    title="Metric",
                ),
                tooltip=[
                    alt.Tooltip(
                        "Model:N",
                        title="Model",
                    ),
                    alt.Tooltip(
                        "Metric:N",
                        title="Metric",
                    ),
                    alt.Tooltip(
                        "Score (%):Q",
                        title="Score",
                        format=".2f",
                    ),
                ],
            )
        )

        ranking_labels = (
            alt.Chart(ranking_quality)
            .mark_text(
                dy=-9,
                fontSize=11,
                color="#f3f4f6",
            )
            .encode(
                x=alt.X(
                    "Model:N",
                    sort=model_order,
                ),
                xOffset=alt.XOffset(
                    "Metric:N",
                ),
                y=alt.Y("Score (%):Q"),
                text=alt.Text("Value label:N"),
            )
        )

        st.altair_chart(
            (ranking_bars + ranking_labels)
            .properties(height=360)
            .configure_axis(grid=False),
            width="stretch",
        )

        st.markdown(
            "The hybrid model performs best on all three measures. "
            "On validation data, it "
            "finds the hidden liked product for 5.95% of users, "
            "compared with 5.30% for collaborative recommendations "
            "and 3.06% for popularity alone. MRR and NDCG show that "
            "the hybrid also places successful matches nearer the "
            "top of the recommendation list."
        )

        with st.expander(
            "What are Hit Rate, MRR and NDCG?"
        ):
            st.markdown(
                """
                - **Hit Rate@10:** whether the hidden product appeared
                  anywhere in the ten recommendations.
                - **MRR@10:** rewards the model when that product appears
                  very near the top; rank 1 receives the most credit.
                - **NDCG@10:** another ranking-quality measure that gives
                  progressively less credit to products placed lower down.

                Higher values are better for all three metrics. They answer
                related but different questions, so their values should not
                be added together.
                """
            )

        st.markdown(
            "### Recommendation catalogue coverage"
        )

        st.caption(
            "The share of eligible catalogue products that "
            "appeared in at least one user's top-10 list."
        )

        coverage_comparison = (
            validation_comparison[
                [
                    "Model",
                    "catalogue_coverage",
                ]
            ]
            .assign(
                **{
                    "Coverage (%)": lambda frame: (
                        frame[
                            "catalogue_coverage"
                        ]
                        * 100
                    )
                }
            )
            [["Model", "Coverage (%)"]]
        )

        coverage_comparison["Value label"] = (
            coverage_comparison["Coverage (%)"]
            .map(lambda value: f"{value:.2f}%")
        )

        coverage_bars = (
            alt.Chart(coverage_comparison)
            .mark_bar(
                color="#7cc4f8",
            )
            .encode(
                x=alt.X(
                    "Model:N",
                    sort=model_order,
                    title=None,
                    axis=alt.Axis(
                        labelAngle=0,
                    ),
                ),
                y=alt.Y(
                    "Coverage (%):Q",
                    title=None,
                    axis=None,
                    scale=alt.Scale(
                        domain=[0, 105],
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "Model:N",
                        title="Model",
                    ),
                    alt.Tooltip(
                        "Coverage (%):Q",
                        title="Coverage",
                        format=".2f",
                    ),
                ],
            )
        )

        coverage_labels = (
            alt.Chart(coverage_comparison)
            .mark_text(
                dy=-9,
                fontSize=12,
                color="#f3f4f6",
            )
            .encode(
                x=alt.X(
                    "Model:N",
                    sort=model_order,
                ),
                y=alt.Y("Coverage (%):Q"),
                text=alt.Text("Value label:N"),
            )
        )

        st.altair_chart(
            (coverage_bars + coverage_labels)
            .properties(height=320)
            .configure_axis(grid=False),
            width="stretch",
        )

        st.markdown(
            "Popularity repeatedly shows "
            "a very small set of products, covering only 0.07% of the "
            "catalogue. Collaborative and hybrid recommendations "
            "collectively reach about 97% of eligible products. This "
            "measures variety across all users—not how many products "
            "one user receives."
        )

        if not hybrid_validation_results.empty:
            st.markdown(
                "### Hybrid weight selection"
            )

            st.caption(
                "Validation results for different collaborative "
                "weights. The remaining weight is assigned to "
                "content evidence."
            )

            weight_results = (
                hybrid_validation_results.copy()
                .sort_values(
                    "collaborative_weight"
                )
            )

            weight_results[
                "Collaborative weight (%)"
            ] = (
                weight_results[
                    "collaborative_weight"
                ]
                * 100
            ).round().astype(int)

            weight_chart = (
                weight_results[
                    [
                        "Collaborative weight (%)",
                        "hit_rate_at_10",
                        "mrr_at_10",
                        "ndcg_at_10",
                    ]
                ]
                .rename(
                    columns={
                        "hit_rate_at_10": "Hit Rate@10",
                        "mrr_at_10": "MRR@10",
                        "ndcg_at_10": "NDCG@10",
                    }
                )
                .melt(
                    id_vars="Collaborative weight (%)",
                    var_name="Metric",
                    value_name="Score",
                )
            )

            weight_chart["Score (%)"] = (
                weight_chart["Score"]
                * 100
            )

            weight_chart["Value label"] = (
                weight_chart["Score (%)"]
                .map(lambda value: f"{value:.2f}%")
            )

            metric_colors = [
                "#7cc4f8",
                "#0b74d1",
                "#f7a1a1",
            ]

            weight_lines = (
                alt.Chart(weight_chart)
                .mark_line(
                    point=alt.OverlayMarkDef(
                        filled=True,
                        size=80,
                    ),
                    strokeWidth=3,
                )
                .encode(
                    x=alt.X(
                        "Collaborative weight (%):Q",
                        title="Collaborative weight",
                        scale=alt.Scale(
                            domain=[25, 95],
                        ),
                        axis=alt.Axis(
                            values=[30, 50, 70, 90],
                            labelExpr="datum.value + '%'",
                            grid=False,
                        ),
                    ),
                    y=alt.Y(
                        "Score (%):Q",
                        title=None,
                        axis=None,
                        scale=alt.Scale(
                            domain=[0, 6.8],
                        ),
                    ),
                    color=alt.Color(
                        "Metric:N",
                        title="Metric",
                        scale=alt.Scale(
                            domain=[
                                "Hit Rate@10",
                                "MRR@10",
                                "NDCG@10",
                            ],
                            range=metric_colors,
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "Collaborative weight (%):Q",
                            title="Collaborative weight",
                            format=".0f",
                        ),
                        alt.Tooltip(
                            "Metric:N",
                            title="Metric",
                        ),
                        alt.Tooltip(
                            "Score (%):Q",
                            title="Score",
                            format=".2f",
                        ),
                    ],
                )
            )

            weight_labels = (
                alt.Chart(weight_chart)
                .mark_text(
                    dy=-12,
                    fontSize=10,
                    fontWeight="bold",
                )
                .encode(
                    x=alt.X(
                        "Collaborative weight (%):Q"
                    ),
                    y=alt.Y("Score (%):Q"),
                    text=alt.Text("Value label:N"),
                    color=alt.Color(
                        "Metric:N",
                        scale=alt.Scale(
                            domain=[
                                "Hit Rate@10",
                                "MRR@10",
                                "NDCG@10",
                            ],
                            range=metric_colors,
                        ),
                        legend=None,
                    ),
                )
            )

            selected_weight_marker = (
                alt.Chart(
                    pd.DataFrame(
                        {
                            "Collaborative weight (%)": [70]
                        }
                    )
                )
                .mark_rule(
                    color="#42d77d",
                    strokeDash=[6, 5],
                    strokeWidth=2,
                )
                .encode(
                    x="Collaborative weight (%):Q"
                )
            )

            st.altair_chart(
                (
                    weight_lines
                    + weight_labels
                    + selected_weight_marker
                )
                .properties(height=360)
                .configure_axis(grid=False),
                width="stretch",
            )

            st.markdown(
                "Increasing collaborative "
                "evidence improves all three measures up to 70%. At "
                "90%, performance falls slightly. Therefore, the model "
                "uses 70% behavioural evidence and 30% product-content "
                "evidence; the choice was measured rather than guessed."
            )

            selected_weight_result = (
                weight_results.sort_values(
                    [
                        "ndcg_at_10",
                        "hit_rate_at_10",
                    ],
                    ascending=False,
                )
                .iloc[0]
            )

            selected_collaborative_weight = int(
                round(
                    selected_weight_result[
                        "collaborative_weight"
                    ]
                    * 100
                )
            )

            selected_content_weight = (
                100
                - selected_collaborative_weight
            )

            st.success(
                f"Selected configuration: "
                f"{selected_collaborative_weight}% collaborative "
                f"and {selected_content_weight}% content. It had "
                f"the highest validation NDCG@10 "
                f"({selected_weight_result['ndcg_at_10'] * 100:.2f}%)."
            )

        st.markdown(
            "### Detailed evaluation results"
        )

        comparison = raw_comparison.copy()

        comparison["hit_rate_at_10"] *= 100
        comparison["mrr_at_10"] *= 100
        comparison["ndcg_at_10"] *= 100
        comparison["catalogue_coverage"] *= 100

        comparison = comparison.rename(
            columns={
                "model": "Model",
                "evaluation_split": "Split",
                "hit_rate_at_10": "Hit Rate@10 (%)",
                "mrr_at_10": "MRR@10 (%)",
                "ndcg_at_10": "NDCG@10 (%)",
                "catalogue_coverage": "Coverage (%)",
            }
        )

        st.dataframe(
            comparison,
            hide_index=True,
            width="stretch",
        )

    st.warning(
        "These are offline ranking results. They do not "
        "prove production conversion, revenue impact or "
        "real-user satisfaction."
    )


# ---------------------------------------------------------
# Methodology tab
# ---------------------------------------------------------

with methodology_tab:
    st.subheader(
        "Three recommendation experiences"
    )

    st.markdown(
        """
        ### Users also liked

        Collaborative neighbours based on overlapping positive
        user behaviour. This does not mean the products were
        purchased together.

        ### Similar products

        Content neighbours based on titles, categories,
        descriptions and features.

        ### Personalized recommendations

        All products in **My likes** form a temporary preference
        profile. The model combines 70% collaborative evidence
        and 30% content evidence to produce ten recommendations.
        """
    )

    st.subheader(
        "Known limitations"
    )

    st.markdown(
        """
        - Ratings of 4 or 5 are treated as positive feedback.
        - Ratings from 1 to 3 are retained but not used here.
        - Unobserved interactions are unknown, not dislikes.
        - The dataset contains ratings, not verified shopping baskets.
        - Product metadata contains some category noise.
        - Offline evaluation does not measure real user satisfaction.
        """
    )
