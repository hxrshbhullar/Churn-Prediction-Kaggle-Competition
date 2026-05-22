import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

# ==========================================
# USER-DEFINED CONFIGURATION PLACEHOLDERS
# ==========================================
# FILL IN: Specify the exact column names your model expects as features (in correct order)
MODEL_FEATURES = [
    'SeniorCitizen',
    'tenure',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
    'MonthlyCharges',
    'TotalCharges',
    'ExpectedCharges',
    'ChargeDifference',
    'Household_Score',
    'high_risk',
    'no_streaming_services',
    'no_services',
    'is_vulnerable'
]

# FILL IN: The column name used to identify customers uniquely (e.g., 'CustomerId', 'Email')
CUSTOMER_ID_COLUMN = None

# Directory where your sample CSV files are stored
SAMPLE_DATA_DIR = "sample_data"
MODEL_PATH = "churn_model.pkl"

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Customer Churn Risk Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimalistic Custom CSS for clean layout
st.markdown(
    """
    <style>
    .reportview-container { background: #fdfdfd; }
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { font-weight: 600; color: #1E293B; }
    div.stAlert { border-radius: 8px; }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# HELPER FUNCTIONS & CACHING
# ==========================================
@st.cache_resource
def load_model(path):
    """Loads the pickled XGBoost model."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None


@st.cache_resource
def calculate_shap_values(_model, _df_processed):
    """Computes SHAP values for local model explanations."""
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer(_df_processed)
    return explainer, shap_values


# ==========================================
# APPLICATION LOGIC
# ==========================================

st.title("Customer Churn Risk Analytics")
st.markdown(
    "Upload or select customer datasets to predict churn risk and understand the underlying drivers behind individual customer behavior."
)
st.markdown("---")

# 1. SIDEBAR: Load Model and Select File
st.sidebar.header("Control Panel")

model = load_model(MODEL_PATH)

if model is not None:
    st.sidebar.success("Model loaded successfully!")
else:
    st.sidebar.error("Failed to load model. Check your model path file.")
    st.stop()

# Ensure sample directory exists
if not os.path.exists(SAMPLE_DATA_DIR):
    os.makedirs(SAMPLE_DATA_DIR)

# Get list of CSVs in sample directory
available_csvs = [f for f in os.listdir(SAMPLE_DATA_DIR) if f.endswith(".csv")]

if not available_csvs:
    st.sidebar.warning(
        f"No sample CSV files found in environment folder: `/{SAMPLE_DATA_DIR}`"
    )
    # Fallback option to let them upload via UI directly if directory is empty
    uploaded_file = st.sidebar.file_uploader(
        "Or upload a customer CSV file directly", type=["csv"]
    )
    selected_file = None
else:
    uploaded_file = None
    selected_file = st.sidebar.selectbox(
        "Choose a sample CSV file:", ["Select a file..."] + available_csvs
    )

# Determine final dataset source
df_raw = None
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
elif selected_file and selected_file != "Select a file...":
    df_raw = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, selected_file))

# Exit early if data isn't loaded yet
if df_raw is None:
    st.info(
        "Please select a sample CSV file from the sidebar or upload a file to begin analysis."
    )
    st.stop()

# Validate that required configurations have been filled out by user safely
is_incomplete_features = any("USER_FILL" in str(f) for f in MODEL_FEATURES)
is_incomplete_id = CUSTOMER_ID_COLUMN is not None and "USER_FILL" in str(CUSTOMER_ID_COLUMN)

if is_incomplete_features or is_incomplete_id:
    st.warning(
        "⚠️ Code configuration incomplete. Please open the script file and update `MODEL_FEATURES` values with your real dataset columns."
    )
    st.stop()

# Validate that required features exist in the uploaded file
# missing_features = [col for col in MODEL_FEATURES if col not in df_raw.columns]
# if missing_features:
#    st.error(
#        f"The selected dataset is missing required model features: {missing_features}"
#    )
#    st.stop()
    
# if CUSTOMER_ID_COLUMN and CUSTOMER_ID_COLUMN in df_raw.columns:
#    customer_options = df_raw[CUSTOMER_ID_COLUMN].unique().tolist()
#    selected_customer = st.selectbox("Select Customer ID to audit:", customer_options)
#    selected_row_idx = df_raw[df_raw[CUSTOMER_ID_COLUMN] == selected_customer].index[0]
# else:
#    # Safe fallback if no identifier column is provided
#    selected_row_idx = st.number_input(
#        "Select Dataset Row Index to analyze:",
#        min_value=0,
#        max_value=len(df_raw) - 1,
#        value=0,
#    )
#    selected_customer = f"Row Index {selected_row_idx}"

# ==========================================
# MAIN SECTION 1: Batch Prediction Display
# ========================================== 
st.header("Batch Prediction Summary")

# List out the columns that your model expects to be categorical/strings
cat_cols = df_raw.select_dtypes(['object']).columns
for col in cat_cols:
    df_raw[col] = df_raw[col].astype('category')

# Filter raw dataframe down to what model expects
X_processed = df_raw[MODEL_FEATURES].copy()

# Generate batch churn probabilities
try:
    # Handle Scikit-Learn wrapper probability output
    probabilities = model.predict_proba(X_processed)[:, 1]
except AttributeError:
    # Fallback if raw booster or alternative layout configuration is passed
    probabilities = model.predict(X_processed)

# Append predictions back to the view dataframe for clean layout display
df_display = df_raw.copy()
df_display["Churn Probability (%)"] = np.round(probabilities * 100, 2)

# Move prediction columns to the front of the display for readability
cols = list(df_display.columns)
if CUSTOMER_ID_COLUMN in cols:
    cols.insert(0, cols.pop(cols.index(CUSTOMER_ID_COLUMN)))
cols.insert(1, cols.pop(cols.index("Churn Probability (%)")))
df_display = df_display[cols]

st.dataframe(df_display, use_container_width=True, hide_index=True)


# ==========================================
# MAIN SECTION 2: SHAP Local Interpretability (Non-Technical)
# ==========================================
st.markdown("---")
st.header("Individual Customer Risk Explainer")
st.markdown(
    "Select a specific customer below to look into the behavioral markers driving or preventing their churn risk."
)

# Dropdown to pick a specific customer row using the specified identity column
if CUSTOMER_ID_COLUMN in df_raw.columns:
    customer_options = df_raw[CUSTOMER_ID_COLUMN].unique().tolist()
    selected_customer = st.selectbox(
        "Select Customer ID to audit:", customer_options
    )
    selected_row_idx = df_raw[df_raw[CUSTOMER_ID_COLUMN] == selected_customer].index[
        0
    ]
else:
    # Fallback index selector if identifier column fails
    selected_row_idx = st.number_input(
        "Select Dataset Row Index:",
        min_value=0,
        max_value=len(df_raw) - 1,
        value=0,
    )
    selected_customer = f"Row Index {selected_row_idx}"

# Isolate the exact selected observation vector
single_obs = X_processed.iloc[[selected_row_idx]]
cust_probability = df_display.loc[
    selected_row_idx, "Churn Probability (%)"
]

# Calculate SHAP structures
explainer, shap_values = calculate_shap_values(model, X_processed)

# Extract specific SHAP forces metrics for the current selection
# Accommodates both old format array types and modern SHAP Explanation data blocks safely
if hasattr(shap_values, "values"):
    row_shap_array = shap_values.values[selected_row_idx]
    base_value = (
        shap_values.base_values[selected_row_idx]
        if hasattr(shap_values, "base_values")
        else 0
    )
else:
    row_shap_array = shap_values[selected_row_idx]
    base_value = getattr(explainer, "expected_value", 0)

# Build dynamic summary structure for clean textual rendering
shap_summary = pd.DataFrame(
    {
        "Feature": MODEL_FEATURES,
        "Impact": row_shap_array,
        "Actual Value": single_obs.values[0],
    }
)

# Isolate Top 2 Positive (Risk Builders) and Top 2 Negative (Risk Counterweights)
top_risk_drivers = (
    shap_summary[shap_summary["Impact"] > 0]
    .sort_values(by="Impact", ascending=False)
    .head(2)
)
top_saving_graces = (
    shap_summary[shap_summary["Impact"] < 0]
    .sort_values(by="Impact", ascending=True)
    .head(2)
)

# Render Non-Technical Explainer Cards
col_metrics, col_drivers, col_graces = st.columns([1, 1, 1], gap="medium")

with col_metrics:
    st.subheader("Current Risk Status")
    if cust_probability >= 50.0:
        st.error(f"### {cust_probability:.2f}% \n High Churn Risk Profile")
    else:
        st.success(f"### {cust_probability:.2f}% \n Stable Retained Profile")
    # st.caption(f"Auditing unique identifier: **{selected_customer}**. Baseline market average risk context falls near: {np.round(base_value*100, 1) if abs(base_value)<=1 else 'N/A'}%.")

with col_drivers:
    st.subheader("Top 2 Churn Risk Drivers")
    if not top_risk_drivers.empty:
        for _, row in top_risk_drivers.iterrows():
            st.markdown(
                f"• **{row['Feature']}** (Value: *{row['Actual Value']}*)\n"
                f"  This metric pushed this specific customer significantly closer toward canceling their account."
            )
    else:
        st.write("No distinct metrics are actively forcing an increased risk level.")

with col_graces:
    st.subheader("Top 2 Saving Graces")
    if not top_saving_graces.empty:
        for _, row in top_saving_graces.iterrows():
            st.markdown(
                f"• **{row['Feature']}** (Value: *{row['Actual Value']}*)\n"
                f"  This metric acts as a strong anchor, keeping this customer loyal and lowering their churn probability."
            )
    else:
        st.write(
            "No active operational metrics are helping to anchor down this customer's risk score."
        )


# ==========================================
# MAIN SECTION 3: Global Feature Importance (Interactive Plotly)
# ==========================================
st.markdown("---")
st.header("Global Model Mechanics")
st.markdown(
    "The chart below illustrates the macro-level importance of features across your entire XGBoost model architecture. Hover or zoom over bars to view individual parameter impacts."
)

if hasattr(model, "feature_importances_"):
    # Extract structural array parameters
    importances = model.feature_importances_

    df_importance = pd.DataFrame(
        {"Feature": MODEL_FEATURES, "Importance Weight": importances}
    ).sort_values(by="Importance Weight", ascending=True)

    # Clean minimalist Plotly layout creation
    fig = px.bar(
        df_importance,
        x="Importance Weight",
        y="Feature",
        orientation="h",
        title="Overall XGBoost Feature Importance Weightings",
        labels={"Importance Weight": "Relative Impact Score", "Feature": ""},
        template="plotly_white",
    )

    fig.update_traces(
        marker_color="#2563EB",
        hovertemplate="<b>%{y}</b><br>Importance Score: %{x:.4f}<extra></extra>",
    )

    fig.update_layout(
        height=min(400 + (len(MODEL_FEATURES) * 15), 800),
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(showgrid=False),
        title_font=dict(size=16, color="#1E293B", family="sans-serif"),
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(
        "Global feature importance configuration vector could not be directly parsed from this pickle type. Ensure model uses the Scikit-Learn wrapper API framework."
    )