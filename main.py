"""
Food Delivery AI — Delivery Time Prediction Dashboard
=======================================================

A production-style Streamlit app around a trained Random Forest
regressor that estimates food delivery time in minutes.

Expected files, next to this script:
    optimized_rf_model.pkl   trained RandomForestRegressor
    label_encoders.pkl       dict[str, LabelEncoder] for the
                              4 categorical features
    Food_Delivery_Times.csv  the training dataset (used here only
                              to show real, live model metrics and
                              a dataset overview — not required for
                              single/batch predictions to work)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Food Delivery AI",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MARKDOWN / HTML RENDER HELPER
# ============================================================

def render_markdown(markup: str, unsafe_allow_html: bool = False):
    """
    Streamlit's Markdown renderer follows normal Markdown rules,
    where any line indented 4+ spaces is treated as a fenced code
    block. Multi-line triple-quoted strings written with nested
    Python indentation trip this every time — the HTML/Markdown
    shows up as raw text in a code box instead of rendering. This
    strips each line's leading whitespace first, which has no
    effect on the rendered output but avoids the false code-block
    detection.
    """

    lines = [line.strip() for line in markup.strip("\n").splitlines()]

    st.markdown("\n".join(lines), unsafe_allow_html=unsafe_allow_html)


# ============================================================
# CUSTOM CSS
# ============================================================

render_markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 35px;
        border-radius: 25px;
        margin-bottom: 25px;
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(14,165,233,0.12));
        border: 1px solid rgba(128,128,128,0.25);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #6b7280;
        line-height: 1.6;
    }

    .pill-row { margin-top: 14px; }

    .pill {
        display: inline-block;
        padding: 6px 14px;
        margin: 4px 6px 4px 0;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        border: 1px solid rgba(99,102,241,0.35);
        background: rgba(99,102,241,0.10);
    }

    .card {
        padding: 22px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 15px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 18px;
        text-align: center;
        border: 1px solid rgba(128,128,128,0.25);
    }

    .metric-value { font-size: 28px; font-weight: 800; }
    .metric-label { color: #6b7280; font-size: 13px; }

    .prediction-card {
        padding: 35px;
        border-radius: 25px;
        text-align: center;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 15px;
        margin-bottom: 25px;
    }

    .prediction-label { font-size: 18px; color: #6b7280; }
    .prediction-value { font-size: 62px; font-weight: 900; }
    .prediction-unit { font-size: 20px; color: #6b7280; }

    .workflow {
        padding: 18px;
        min-height: 150px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.25);
    }

    .workflow-number { font-size: 22px; font-weight: 800; color: #6366f1; }
    .workflow-icon { font-size: 32px; }

    .feature-box {
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.25);
        min-height: 140px;
    }

    .badge-ok {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        background: rgba(34,197,94,0.15);
        color: #16a34a;
        border: 1px solid rgba(34,197,94,0.35);
    }

    .footer { text-align: center; color: #6b7280; padding: 20px; font-size: 13px; }
    .image-credit { text-align: right; color: #9ca3af; font-size: 12px; }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "optimized_rf_model.pkl"
ENCODERS_PATH = BASE_DIR / "label_encoders.pkl"
DATA_PATH = BASE_DIR / "Food_Delivery_Times.csv"

HERO_IMAGE_URL = (
    "https://images.unsplash.com/photo-1572195577046-2f25894c06fc"
    "?fm=jpg&q=80&w=1200&auto=format&fit=crop"
)
HERO_IMAGE_CREDIT = "Photo by Lucian Alexe on Unsplash"

# Must match the exact column order the model was trained on
# (verified against optimized_rf_model.feature_names_in_).
FEATURE_COLUMNS = [
    "Distance_km",
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
]

CATEGORICAL_COLUMNS = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
TARGET_COLUMN = "Delivery_Time_min"


# ============================================================
# CACHED LOADERS
# ============================================================

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_resource(show_spinner="Loading encoders...")
def load_encoders():
    with open(ENCODERS_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_data(show_spinner=False)
def load_raw_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def get_fill_stats(raw_df: pd.DataFrame):
    """
    Same cleaning strategy used to train the model: categorical
    nulls filled with the column mode, numeric null
    (Courier_Experience_yrs) filled with the median. Returned so the
    exact same values can be reused for batch predictions on new
    data.
    """

    return {
        "Weather": raw_df["Weather"].mode()[0],
        "Traffic_Level": raw_df["Traffic_Level"].mode()[0],
        "Time_of_Day": raw_df["Time_of_Day"].mode()[0],
        "Courier_Experience_yrs": raw_df["Courier_Experience_yrs"].median(),
    }


def clean_dataframe(df: pd.DataFrame, fill_stats: dict) -> pd.DataFrame:
    df = df.copy()

    for col in ["Weather", "Traffic_Level", "Time_of_Day"]:
        if col in df.columns:
            df[col] = df[col].fillna(fill_stats[col])

    if "Courier_Experience_yrs" in df.columns:
        df["Courier_Experience_yrs"] = df["Courier_Experience_yrs"].fillna(
            fill_stats["Courier_Experience_yrs"]
        )

    if "Order_ID" in df.columns:
        df = df.drop(columns=["Order_ID"])

    return df


def encode_dataframe(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()

    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])

    return df


@st.cache_data(show_spinner="Evaluating model on hold-out data...")
def compute_model_metrics(_model, _encoders, raw_df: pd.DataFrame):
    """
    Reproduces the exact train/test split used when the model was
    trained (80/20, random_state=42) so the metrics shown in the
    dashboard are real numbers computed against the actual pickled
    model — not placeholders.
    """

    fill_stats = get_fill_stats(raw_df)
    clean_df = clean_dataframe(raw_df, fill_stats)
    encoded_df = encode_dataframe(clean_df, _encoders)

    y = encoded_df[TARGET_COLUMN]
    X = encoded_df[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = _model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(_model, X, y, cv=5, scoring="r2")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "n_test": len(X_test),
        "n_total": len(X),
    }


@st.cache_data(show_spinner=False)
def get_input_bounds(raw_df: pd.DataFrame):
    return {
        "distance_min": float(raw_df["Distance_km"].min()),
        "distance_max": float(raw_df["Distance_km"].max()),
        "distance_default": float(round(raw_df["Distance_km"].median(), 1)),
        "prep_min": int(raw_df["Preparation_Time_min"].min()),
        "prep_max": int(raw_df["Preparation_Time_min"].max()),
        "prep_default": int(raw_df["Preparation_Time_min"].median()),
        "exp_min": int(raw_df["Courier_Experience_yrs"].min()),
        "exp_max": int(raw_df["Courier_Experience_yrs"].max()),
        "exp_default": int(raw_df["Courier_Experience_yrs"].median()),
    }


def get_base_estimator(model):
    """
    Unwraps a sklearn Pipeline to reach the actual regressor, so
    `n_estimators` / `max_depth` / `feature_importances_` still
    resolve correctly if the pickle is ever swapped for a Pipeline
    instead of a bare RandomForestRegressor.
    """

    if hasattr(model, "feature_importances_") or hasattr(model, "n_estimators"):
        return model

    if hasattr(model, "steps"):
        return model.steps[-1][1]

    if hasattr(model, "named_steps"):
        return list(model.named_steps.values())[-1]

    return model


def predict_one(model, encoders, distance_km, weather, traffic, time_of_day, vehicle, prep_time, experience):
    row = pd.DataFrame(
        [[distance_km, weather, traffic, time_of_day, vehicle, prep_time, experience]],
        columns=FEATURE_COLUMNS,
    )

    for col in CATEGORICAL_COLUMNS:
        row[col] = encoders[col].transform(row[col])

    prediction = float(model.predict(row)[0])

    return max(prediction, 0.0)


# ============================================================
# MODEL / DATA INITIALIZATION
# ============================================================

if not MODEL_PATH.exists() or not ENCODERS_PATH.exists():
    st.error("Required model files were not found.")
    st.info(
        "Make sure these files sit next to this script:\n\n"
        f"• {MODEL_PATH.name}\n"
        f"• {ENCODERS_PATH.name}\n\n"
        f"Expected folder: `{BASE_DIR}`"
    )
    st.stop()

try:
    model = load_model()
    label_encoders = load_encoders()
    base_estimator = get_base_estimator(model)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

missing_keys = [k for k in CATEGORICAL_COLUMNS if k not in label_encoders]
if missing_keys:
    st.error("`label_encoders.pkl` is missing expected keys: " + ", ".join(missing_keys))
    st.info(f"Keys found in the file: {list(label_encoders.keys())}")
    st.stop()

raw_data = load_raw_data()
data_available = raw_data is not None

if data_available:
    fill_stats = get_fill_stats(raw_data)
    input_bounds = get_input_bounds(raw_data)
else:
    fill_stats = None
    input_bounds = {
        "distance_min": 0.1, "distance_max": 50.0, "distance_default": 10.0,
        "prep_min": 1, "prep_max": 60, "prep_default": 20,
        "exp_min": 0, "exp_max": 20, "exp_default": 2,
    }


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🚴 Food Delivery AI")
    st.caption("Machine Learning Prediction System")

    st.divider()

    st.subheader("Model Status")
    st.success("Random Forest Loaded")
    st.success("Label Encoders Loaded")

    if data_available:
        st.success(f"Training Data Loaded ({len(raw_data):,} rows)")
    else:
        st.warning("Training data CSV not found — live metrics disabled.")

    st.divider()

    st.subheader("Navigation")

    page = st.radio(
        "Select Page",
        [
            "🏠 Home",
            "🤖 Prediction",
            "🧾 Batch Prediction",
            "📊 Model Insights",
            "📈 Prediction History",
            "🎥 How It Works",
            "ℹ️ About Project",
        ],
    )

    st.divider()

    st.caption("Python • Pandas • NumPy • Scikit-learn • Streamlit")


# ============================================================
# HERO SECTION
# ============================================================

hero_col1, hero_col2 = st.columns([1.4, 1], gap="large")

with hero_col1:
    render_markdown(
        """
        <div class="hero">
            <div class="hero-title">🚴 Food Delivery AI</div>
            <div class="hero-subtitle">
                AI-powered food delivery time prediction using
                an optimized Random Forest regression model.
            </div>
            <div class="pill-row">
                <span class="pill">📍 Distance</span>
                <span class="pill">🚦 Traffic</span>
                <span class="pill">🌦️ Weather</span>
                <span class="pill">🍔 Prep Time</span>
                <span class="pill">🏍️ Vehicle</span>
                <span class="pill">🕐 Time of Day</span>
                <span class="pill">👨‍🚴 Experience</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_col2:
    st.image(HERO_IMAGE_URL, use_container_width=True, caption=HERO_IMAGE_CREDIT)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.subheader("Machine Learning Delivery Prediction")

    st.write(
        "This application uses a trained Random Forest model to estimate "
        "food delivery time from real-world delivery conditions."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    metric_cards = [
        ("7", "Input Features"),
        (str(base_estimator.n_estimators) if hasattr(base_estimator, "n_estimators") else "RF", "Trees in Forest"),
        (f"{len(raw_data):,}" if data_available else "N/A", "Training Records"),
        ("Real-Time", "Prediction"),
    ]

    for col, (value, label) in zip([col1, col2, col3, col4], metric_cards):
        with col:
            render_markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    st.subheader("🔄 Machine Learning Workflow")

    workflow = [
        ("01", "📥", "User Input", "Enter delivery information."),
        ("02", "⚙️", "Preprocessing", "Encode categorical variables."),
        ("03", "🌲", "Random Forest", "Process input using trained trees."),
        ("04", "🤖", "Prediction", "Generate delivery time."),
        ("05", "📊", "Result", "Display estimated minutes."),
    ]

    workflow_cols = st.columns(5)

    for col, (number, icon, title, description) in zip(workflow_cols, workflow):
        with col:
            render_markdown(
                f"""
                <div class="workflow">
                    <div class="workflow-number">{number}</div>
                    <div class="workflow-icon">{icon}</div>
                    <h4>{title}</h4>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    st.subheader("🔍 Model Features")

    features = [
        ("📍", "Distance", "Delivery distance in kilometers."),
        ("🌦️", "Weather", "Weather condition during delivery."),
        ("🚦", "Traffic", "Traffic intensity."),
        ("🕐", "Time of Day", "Morning, afternoon, evening or night."),
        ("🏍️", "Vehicle", "Vehicle used by the courier."),
        ("🍔", "Preparation Time", "Restaurant food preparation duration."),
        ("👨‍🚴", "Experience", "Courier experience in years."),
    ]

    feature_cols = st.columns(4)

    for index, (icon, title, description) in enumerate(features):
        with feature_cols[index % 4]:
            render_markdown(
                f"""
                <div class="feature-box">
                    <div style="font-size:32px;">{icon}</div>
                    <h4>{title}</h4>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🤖 Prediction":

    st.subheader("🚀 Delivery Time Prediction")
    st.write("Enter the delivery information below.")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        distance_km = st.number_input(
            "📍 Distance (km)",
            min_value=0.1,
            max_value=max(100.0, input_bounds["distance_max"]),
            value=input_bounds["distance_default"],
            step=0.1,
            help=f"Training data ranged from {input_bounds['distance_min']:.1f} to {input_bounds['distance_max']:.1f} km.",
        )

    with col2:
        preparation_time_min = st.number_input(
            "🍔 Preparation Time (min)",
            min_value=1,
            max_value=max(90, input_bounds["prep_max"]),
            value=input_bounds["prep_default"],
            step=1,
            help=f"Training data ranged from {input_bounds['prep_min']} to {input_bounds['prep_max']} min.",
        )

    with col3:
        courier_experience_yrs = st.number_input(
            "👨‍🚴 Courier Experience (years)",
            min_value=0,
            max_value=max(20, input_bounds["exp_max"]),
            value=input_bounds["exp_default"],
            step=1,
            help=f"Training data ranged from {input_bounds['exp_min']} to {input_bounds['exp_max']} years.",
        )

    st.write("")

    col4, col5 = st.columns(2)

    with col4:
        weather_selected = st.selectbox("🌦️ Weather", list(label_encoders["Weather"].classes_))

    with col5:
        traffic_selected = st.selectbox("🚦 Traffic Level", list(label_encoders["Traffic_Level"].classes_))

    col6, col7 = st.columns(2)

    with col6:
        time_selected = st.selectbox("🕐 Time of Day", list(label_encoders["Time_of_Day"].classes_))

    with col7:
        vehicle_selected = st.selectbox("🏍️ Vehicle Type", list(label_encoders["Vehicle_Type"].classes_))

    st.divider()

    predict_button = st.button("🚀 Predict Delivery Time", type="primary", use_container_width=True)

    if predict_button:

        try:
            prediction = predict_one(
                model,
                label_encoders,
                distance_km,
                weather_selected,
                traffic_selected,
                time_selected,
                vehicle_selected,
                preparation_time_min,
                courier_experience_yrs,
            )

            render_markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">Estimated Delivery Time</div>
                    <div class="prediction-value">{prediction:.2f}</div>
                    <div class="prediction-unit">minutes</div>
                    <br>
                    <div>🌲 Optimized Random Forest</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if prediction <= 30:
                st.success("⚡ Fast Delivery — estimated delivery time is relatively short.")
            elif prediction <= 60:
                st.info("🚴 Normal Delivery — estimated delivery time is within a typical range.")
            else:
                st.warning("⏳ Longer Delivery — the estimated delivery time is relatively high.")

            st.subheader("📋 Prediction Summary")

            summary = pd.DataFrame(
                {
                    "Feature": [
                        "Distance", "Weather", "Traffic Level", "Time of Day",
                        "Vehicle Type", "Preparation Time", "Courier Experience",
                        "Predicted Delivery Time",
                    ],
                    "Value": [
                        f"{distance_km:.1f} km",
                        weather_selected,
                        traffic_selected,
                        time_selected,
                        vehicle_selected,
                        f"{preparation_time_min} min",
                        f"{courier_experience_yrs} years",
                        f"{prediction:.2f} min",
                    ],
                }
            )

            st.dataframe(summary, use_container_width=True, hide_index=True)

            st.session_state.prediction_history.append(
                {
                    "Distance (km)": distance_km,
                    "Weather": weather_selected,
                    "Traffic": traffic_selected,
                    "Time of Day": time_selected,
                    "Vehicle": vehicle_selected,
                    "Preparation (min)": preparation_time_min,
                    "Experience (yrs)": courier_experience_yrs,
                    "Prediction (min)": round(prediction, 2),
                }
            )

            st.success("Prediction successfully generated!")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ============================================================
# BATCH PREDICTION PAGE
# ============================================================

elif page == "🧾 Batch Prediction":

    st.subheader("🧾 Batch Prediction")

    st.write(
        "Upload a CSV with multiple deliveries to score them all at once. "
        "Required columns:"
    )

    st.code(", ".join(FEATURE_COLUMNS), language=None)

    if data_available:
        with st.expander("See a sample of the expected format"):
            st.dataframe(raw_data[FEATURE_COLUMNS].head(5), use_container_width=True, hide_index=True)

    uploaded_csv = st.file_uploader("Upload deliveries CSV", type=["csv"])

    if uploaded_csv is not None:

        try:
            batch_df = pd.read_csv(uploaded_csv)
        except Exception as e:
            st.error(f"Couldn't read that CSV: {e}")
            st.stop()

        missing_cols = [c for c in FEATURE_COLUMNS if c not in batch_df.columns]

        if missing_cols:
            st.error("The uploaded file is missing required columns: " + ", ".join(missing_cols))

        else:
            st.success(f"Loaded {len(batch_df):,} rows.")

            working_df = batch_df[FEATURE_COLUMNS].copy()

            # Fill missing values the same way the model was trained,
            # falling back to simple defaults if the training CSV
            # isn't available in this deployment.
            batch_fill_stats = fill_stats or {
                "Weather": working_df["Weather"].mode().iloc[0] if not working_df["Weather"].mode().empty else "Clear",
                "Traffic_Level": working_df["Traffic_Level"].mode().iloc[0] if not working_df["Traffic_Level"].mode().empty else "Medium",
                "Time_of_Day": working_df["Time_of_Day"].mode().iloc[0] if not working_df["Time_of_Day"].mode().empty else "Morning",
                "Courier_Experience_yrs": working_df["Courier_Experience_yrs"].median(),
            }

            for col in ["Weather", "Traffic_Level", "Time_of_Day"]:
                working_df[col] = working_df[col].fillna(batch_fill_stats[col])

            working_df["Courier_Experience_yrs"] = working_df["Courier_Experience_yrs"].fillna(
                batch_fill_stats["Courier_Experience_yrs"]
            )

            # Flag rows with categories the encoders have never seen,
            # rather than letting the whole batch fail.
            row_errors = pd.Series(False, index=working_df.index)

            for col in CATEGORICAL_COLUMNS:
                known = set(label_encoders[col].classes_)
                row_errors |= ~working_df[col].isin(known)

            valid_df = working_df[~row_errors].copy()
            invalid_df = batch_df[row_errors]

            if row_errors.any():
                st.warning(
                    f"{row_errors.sum()} row(s) contain categories the model has never seen "
                    "and were skipped. Expand below to see them."
                )

                with st.expander("Skipped rows"):
                    st.dataframe(invalid_df, use_container_width=True, hide_index=True)

            if len(valid_df) == 0:
                st.error("No valid rows left to predict on.")

            else:
                encoded_df = valid_df.copy()

                for col in CATEGORICAL_COLUMNS:
                    encoded_df[col] = label_encoders[col].transform(encoded_df[col])

                predictions = model.predict(encoded_df[FEATURE_COLUMNS])
                predictions = np.clip(predictions, 0, None)

                result_df = batch_df.loc[valid_df.index].copy()
                result_df["Predicted_Delivery_Time_min"] = np.round(predictions, 2)

                st.subheader("📋 Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Rows Scored", len(result_df))
                with col2:
                    st.metric("Average Predicted Time", f"{result_df['Predicted_Delivery_Time_min'].mean():.1f} min")
                with col3:
                    st.metric("Max Predicted Time", f"{result_df['Predicted_Delivery_Time_min'].max():.1f} min")

                st.dataframe(result_df, use_container_width=True, hide_index=True)

                csv_out = result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=csv_out,
                    file_name="batch_delivery_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ============================================================
# MODEL INSIGHTS PAGE
# ============================================================

elif page == "📊 Model Insights":

    st.subheader("📊 Random Forest Model Insights")
    st.write("Technical information, real evaluation metrics, and dataset context for the trained model.")

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Algorithm", "Random Forest")

    with col2:
        st.metric("Trees", str(getattr(base_estimator, "n_estimators", "N/A")))

    with col3:
        max_depth = getattr(base_estimator, "max_depth", None)
        st.metric("Max Depth", "None" if max_depth is None else str(max_depth))

    with col4:
        st.metric("Features", str(len(FEATURE_COLUMNS)))

    st.divider()

    st.subheader("📈 Model Performance")

    if data_available:

        metrics = compute_model_metrics(model, label_encoders, raw_data)

        st.caption(
            f"Computed on a held-out 20% test split ({metrics['n_test']} of "
            f"{metrics['n_total']} rows, random_state=42) — the same split used at training time."
        )

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("R² Score", f"{metrics['r2']:.3f}")
        with m2:
            st.metric("RMSE", f"{metrics['rmse']:.2f} min")
        with m3:
            st.metric("MAE", f"{metrics['mae']:.2f} min")
        with m4:
            st.metric("5-Fold CV R²", f"{metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")

    else:
        st.info("Upload/include `Food_Delivery_Times.csv` next to this script to compute live evaluation metrics.")

    st.divider()

    st.subheader("🌟 Feature Importance")

    if hasattr(base_estimator, "feature_importances_"):

        importances = np.asarray(base_estimator.feature_importances_)

        if len(importances) == len(FEATURE_COLUMNS):

            importance_df = pd.DataFrame(
                {"Feature": FEATURE_COLUMNS, "Importance": importances}
            ).sort_values("Importance", ascending=False)

            st.bar_chart(importance_df.set_index("Feature"))

            display_df = importance_df.copy()
            display_df["Importance"] = display_df["Importance"].round(4)

            st.dataframe(display_df, use_container_width=True, hide_index=True)

        else:
            st.warning(
                f"Model reports {len(importances)} feature importances, "
                f"but this app expects {len(FEATURE_COLUMNS)}."
            )

    else:
        st.info("Feature importance is not available for this model.")

    if data_available:

        st.divider()

        st.subheader("🗂️ Dataset Overview")

        clean_df = clean_dataframe(raw_data, fill_stats)

        d1, d2, d3, d4 = st.columns(4)

        with d1:
            st.metric("Total Records", f"{len(raw_data):,}")
        with d2:
            st.metric("Avg Delivery Time", f"{raw_data[TARGET_COLUMN].mean():.1f} min")
        with d3:
            st.metric("Missing Values (raw)", f"{int(raw_data.isna().sum().sum())}")
        with d4:
            st.metric("Missing After Cleaning", f"{int(clean_df.isna().sum().sum())}")

        tab1, tab2, tab3 = st.tabs(["Delivery Time Distribution", "By Traffic Level", "By Weather"])

        with tab1:
            bins = pd.cut(raw_data[TARGET_COLUMN], bins=15)
            dist = raw_data.groupby(bins, observed=True)[TARGET_COLUMN].count()
            dist.index = [f"{int(i.left)}–{int(i.right)}" for i in dist.index]
            st.bar_chart(dist)

        with tab2:
            by_traffic = clean_df.groupby("Traffic_Level", observed=True)[TARGET_COLUMN].mean().sort_values(ascending=False)
            st.bar_chart(by_traffic)

        with tab3:
            by_weather = clean_df.groupby("Weather", observed=True)[TARGET_COLUMN].mean().sort_values(ascending=False)
            st.bar_chart(by_weather)

        with st.expander("View a sample of the cleaned training data"):
            st.dataframe(clean_df.head(20), use_container_width=True, hide_index=True)


# ============================================================
# PREDICTION HISTORY PAGE
# ============================================================

elif page == "📈 Prediction History":

    st.subheader("📈 Prediction History")

    if len(st.session_state.prediction_history) == 0:
        st.info("No predictions have been made yet.")
        st.write("Go to the Prediction page and generate your first prediction.")

    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            st.metric("Average Delivery Time", f"{history_df['Prediction (min)'].mean():.2f} min")
        with col3:
            st.metric("Maximum Delivery Time", f"{history_df['Prediction (min)'].max():.2f} min")

        st.divider()

        st.dataframe(history_df, use_container_width=True, hide_index=True)

        st.divider()

        st.subheader("📊 Prediction Trend")
        st.line_chart(history_df[["Prediction (min)"]])

        st.divider()

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Prediction History CSV",
            data=csv_data,
            file_name="food_delivery_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.divider()

        if st.button("🗑️ Clear Prediction History"):
            st.session_state.prediction_history = []
            st.rerun()


# ============================================================
# HOW IT WORKS PAGE
# ============================================================

elif page == "🎥 How It Works":

    st.subheader("🎥 How the ML Application Works")
    st.write("This section explains the complete machine learning deployment pipeline.")

    st.divider()

    st.subheader("🔄 End-to-End ML Pipeline")

    pipeline = [
        ("📥", "Input Data", "User enters delivery information."),
        ("🧹", "Preprocessing", "Categorical variables are encoded."),
        ("📦", "Feature Vector", "All seven features are assembled."),
        ("🌲", "Random Forest", "Trained model processes the input."),
        ("🤖", "Prediction", "Delivery time is estimated."),
    ]

    cols = st.columns(5)

    for col, (icon, title, description) in zip(cols, pipeline):
        with col:
            render_markdown(
                f"""
                <div class="workflow">
                    <div class="workflow-icon">{icon}</div>
                    <h4>{title}</h4>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    st.subheader("🎬 Project Demo Video")
    st.write("Upload an MP4/MOV/WebM video to demonstrate your project.")

    uploaded_video = st.file_uploader("Upload Project Demo", type=["mp4", "mov", "webm"])

    if uploaded_video is not None:
        st.video(uploaded_video)
    else:
        st.info("No demo video uploaded yet.")

    st.divider()

    st.subheader("🧠 Technical Explanation")

    render_markdown(
        """
        ### Step 1 — Input
        The user enters seven features related to the delivery.

        ### Step 2 — Cleaning & Encoding
        Missing categorical values are filled with the training
        set's mode, missing courier experience with the median.
        Categorical variables (Weather, Traffic Level, Time of Day,
        Vehicle Type) are transformed using the saved LabelEncoders.

        ### Step 3 — DataFrame
        The encoded values and numerical features are assembled
        into a Pandas DataFrame using the exact feature order the
        model was trained on.

        ### Step 4 — Random Forest
        The optimized Random Forest regression model processes the
        feature vector.

        ### Step 5 — Prediction
        The model returns the estimated delivery time in minutes.

        ### Step 6 — Visualization
        The prediction is displayed through the Streamlit dashboard
        and stored in prediction history.
        """
    )


# ============================================================
# ABOUT PROJECT PAGE
# ============================================================

elif page == "ℹ️ About Project":

    st.subheader("ℹ️ About the Project")

    render_markdown(
        """
        ## Food Delivery Time Prediction

        An end-to-end Machine Learning application that predicts
        estimated food delivery time from real-world delivery
        conditions, deployed as an interactive Streamlit dashboard.

        ### Machine Learning Workflow

        - Data Collection
        - Data Cleaning (mode/median imputation)
        - Exploratory Data Analysis
        - Categorical Encoding
        - Model Training (Decision Tree baseline, Random Forest)
        - Hyperparameter Optimization (RandomizedSearchCV, 5-fold CV)
        - Model Evaluation (MAE, RMSE, R²)
        - Model Serialization (pickle)
        - Streamlit Deployment

        ### Technology Stack

        **Programming** — Python

        **Data Processing** — Pandas, NumPy

        **Machine Learning** — Scikit-learn, Random Forest Regression

        **Deployment** — Streamlit, Pickle

        ### Model Features

        1. Distance_km
        2. Weather
        3. Traffic_Level
        4. Time_of_Day
        5. Vehicle_Type
        6. Preparation_Time_min
        7. Courier_Experience_yrs
        """
    )

    st.divider()

    st.subheader("📋 Model Input Schema")

    schema_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Type": ["Numerical", "Categorical", "Categorical", "Categorical", "Categorical", "Numerical", "Numerical"],
            "Role": [
                "Delivery distance", "Weather condition", "Traffic intensity",
                "Delivery time period", "Courier vehicle", "Restaurant preparation",
                "Courier experience",
            ],
        }
    )

    st.dataframe(schema_df, use_container_width=True, hide_index=True)

    if hasattr(base_estimator, "get_params"):
        st.divider()
        st.subheader("⚙️ Trained Hyperparameters")
        params = base_estimator.get_params()
        key_params = {
            k: params.get(k)
            for k in ["n_estimators", "max_depth", "max_features", "min_samples_split", "min_samples_leaf", "random_state"]
            if k in params
        }
        st.dataframe(
            pd.DataFrame(list(key_params.items()), columns=["Parameter", "Value"]),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# FOOTER
# ============================================================

render_markdown(
    """
    <div class="footer">
        🚴 Food Delivery AI
        <br>
        Machine Learning Prediction System
        <br>
        Built with Python + Scikit-learn + Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
