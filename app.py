import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
st.set_page_config(page_title="Auto MPG Predictor", page_icon="🚗", layout="wide")

DEFAULT_DATA = Path(__file__).with_name("auto-mpg.csv")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.hero {padding: 1.25rem 1.5rem; border-radius: 16px; background: linear-gradient(120deg,#102a43,#176b87); color: white; margin-bottom: 1rem;}
.hero h1 {margin:0; color:white;}
.hero p {margin:.4rem 0 0; color:#e6f4f7;}
div[data-testid="stMetric"] {border:1px solid #e5e7eb; padding:12px; border-radius:12px;}
</style>
<div class="hero"><h1>🚗 Automobile Fuel Efficiency Predictor</h1><p>Estimate miles per gallon (MPG) from vehicle specifications. Includes automatic preprocessing and comparison of three machine-learning models.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Data & settings")
    uploaded = st.file_uploader("Upload vehicle data (CSV or Excel)", type=["csv", "xlsx", "xls"])
    st.caption("If you don't upload a file, the included Auto MPG dataset is used.")
    test_size = st.slider("Test data (%)", min_value=10, max_value=40, value=20, step=5)
    st.divider()
    st.markdown("**Expected target column:** `mpg` (or choose a target below).")

@st.cache_data(show_spinner=False)
def read_data(file_bytes, file_name):
    if file_bytes is None:
        if not DEFAULT_DATA.exists():
            return None
        return pd.read_csv(DEFAULT_DATA)
    bio = io.BytesIO(file_bytes)
    if file_name.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(bio)
    return pd.read_csv(bio)

try:
    raw = read_data(uploaded.getvalue() if uploaded else None, uploaded.name if uploaded else None)
except Exception as exc:
    st.error(f"Could not read this file: {exc}")
    st.stop()

if raw is None:
    st.error("Dataset not found. Upload the provided auto-mpg.csv file.")
    st.stop()
if raw.empty or len(raw.columns) < 2:
    st.error("Please upload a dataset with at least two columns and some rows.")
    st.stop()

# Normalize headers and common Auto MPG missing-value markers.
df = raw.copy()
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
df = df.loc[:, ~df.columns.duplicated()]
df = df.replace(["?", "NA", "N/A", "na", "null", "None", ""], np.nan)
for col in df.columns:
    if df[col].dtype == "object":
        converted = pd.to_numeric(df[col], errors="coerce")
        non_missing_original = df[col].notna().sum()
        if non_missing_original and converted.notna().sum() / non_missing_original >= 0.85:
            df[col] = converted

# Clean up rows that are entirely empty; preprocessing imputes feature-level missing values.
df = df.dropna(how="all").reset_index(drop=True)
if len(df) < 10:
    st.error("The dataset needs at least 10 usable rows to compare models reliably.")
    st.stop()

preferred = next((c for c in df.columns if c in ("mpg", "fuel_efficiency", "fuel_efficiency_mpg")), None)
with st.expander("📄 Preview and inspect uploaded data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    missing = df.isna().sum().rename("Missing values").to_frame()
    st.dataframe(missing, use_container_width=True)
    st.caption(f"Rows: {len(df):,} · Columns: {len(df.columns)} · Missing cells will be handled in the preprocessing pipeline.")

target_default = df.columns.get_loc(preferred) if preferred else 0
target = st.selectbox("Choose the value to predict (target column)", options=list(df.columns), index=target_default)

# The target must be numeric for regression.
y_all = pd.to_numeric(df[target], errors="coerce")
valid_target = y_all.notna()
model_df = df.loc[valid_target].copy().reset_index(drop=True)
y = pd.to_numeric(model_df[target], errors="coerce").astype(float)
feature_cols = [c for c in model_df.columns if c != target]
if len(feature_cols) == 0:
    st.error("Select a numeric target and ensure the dataset has at least one feature column.")
    st.stop()
if len(model_df) < 10:
    st.error("Not enough rows have a valid numeric target value.")
    st.stop()
X = model_df[feature_cols].copy()

numeric_features = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())])
categorical_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
transformers = []
if numeric_features:
    transformers.append(("numeric", numeric_pipe, numeric_features))
if categorical_features:
    transformers.append(("categorical", categorical_pipe, categorical_features))
preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1),
}

st.subheader("1. Model comparison")
st.write("Three regression algorithms are evaluated using cross-validation. The model with the highest average cross-validation R² score is selected for prediction.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)
cv = KFold(n_splits=min(5, max(2, len(X_train) // 5)), shuffle=True, random_state=42)
results = []
pipelines = {}
progress = st.progress(0, text="Evaluating models…")
for idx, (name, estimator) in enumerate(models.items(), start=1):
    pipe = Pipeline(steps=[("preprocessing", preprocessor), ("model", estimator)])
    try:
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=1)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({"Model": name, "CV R² (mean)": float(np.mean(scores)), "Test R²": float(r2_score(y_test, pred)), "Test MAE": float(mean_absolute_error(y_test, pred)), "Test RMSE": float(np.sqrt(mean_squared_error(y_test, pred)))})
        pipelines[name] = pipe
    except Exception as exc:
        st.warning(f"{name} could not be trained on this dataset: {exc}")
    progress.progress(idx / len(models), text=f"Evaluated {idx} of {len(models)} models")
progress.empty()
if not results:
    st.error("None of the models could be trained. Check the selected target and uploaded data.")
    st.stop()
results_df = pd.DataFrame(results).sort_values("CV R² (mean)", ascending=False).reset_index(drop=True)
best_name = results_df.iloc[0]["Model"]
best_model = pipelines[best_name]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows used", f"{len(model_df):,}")
c2.metric("Features", len(feature_cols))
c3.metric("Best model", best_name)
c4.metric("Test R²", f"{results_df.iloc[0]['Test R²']:.3f}")
st.dataframe(results_df.style.format({"CV R² (mean)":"{:.3f}", "Test R²":"{:.3f}", "Test MAE":"{:.3f}", "Test RMSE":"{:.3f}"}), use_container_width=True, hide_index=True)
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    fig = px.bar(results_df, x="Model", y="CV R² (mean)", title="Cross-validation R² by model", text_auto=".3f")
    fig.update_layout(yaxis_title="Mean CV R²", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
with chart_col2:
    test_predictions = best_model.predict(X_test)
    compare = pd.DataFrame({"Actual MPG / target": y_test.to_numpy(), "Predicted value": test_predictions})
    fig = px.scatter(compare, x="Actual MPG / target", y="Predicted value", title=f"Actual vs predicted — {best_name}", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("2. Predict fuel efficiency")
st.write("Enter vehicle specifications below. Numeric fields start with typical values from the dataset; categorical fields use known categories. Unknown categories are safely encoded by the pipeline.")
with st.form("prediction_form"):
    input_values = {}
    cols = st.columns(2)
    for i, col in enumerate(feature_cols):
        with cols[i % 2]:
            series = X[col]
            if col in numeric_features:
                vals = pd.to_numeric(series, errors="coerce").dropna()
                median = float(vals.median()) if not vals.empty else 0.0
                low = float(vals.min()) if not vals.empty else median - 10
                high = float(vals.max()) if not vals.empty else median + 10
                # Expand bounds so the entered value can be outside the training range.
                span = max(high - low, abs(median) * 0.2, 1.0)
                step = 1.0 if pd.api.types.is_integer_dtype(series.dtype) else 0.1
                input_values[col] = st.number_input(col.replace("_", " ").title(), value=median, min_value=low - span * 0.5, max_value=high + span * 0.5, step=step, key=f"input_{col}")
            else:
                categories = series.dropna().astype(str).value_counts().index.tolist()
                if col == "car_name" or len(categories) > 30:
                    default = str(series.dropna().iloc[0]) if series.notna().any() else "Unknown"
                    input_values[col] = st.text_input(col.replace("_", " ").title(), value=default, key=f"input_{col}")
                else:
                    options = categories or ["Unknown"]
                    input_values[col] = st.selectbox(col.replace("_", " ").title(), options=options, key=f"input_{col}")
    submitted = st.form_submit_button("🚀 Predict target value", type="primary", use_container_width=True)

if submitted:
    one_row = pd.DataFrame([{col: input_values[col] for col in feature_cols}])
    for col in numeric_features:
        one_row[col] = pd.to_numeric(one_row[col], errors="coerce")
    prediction = float(best_model.predict(one_row)[0])
    unit = "MPG" if target.lower() in ("mpg", "fuel_efficiency", "fuel_efficiency_mpg") else f"predicted {target}"
    st.success(f"Estimated {unit}: **{prediction:.2f}**")
    if target.lower() == "mpg":
        if prediction >= 30:
            st.info("Interpretation: relatively fuel-efficient compared with many vehicles in this dataset.")
        elif prediction >= 20:
            st.info("Interpretation: moderate fuel efficiency for this dataset.")
        else:
            st.info("Interpretation: lower fuel efficiency for this dataset.")
    st.caption("This is a machine-learning estimate, not a guarantee of real-world mileage. Actual MPG varies with maintenance, driving style, road conditions and fuel type.")

st.subheader("3. Download cleaned data and model results")
cleaned_csv = model_df.to_csv(index=False).encode("utf-8")
metrics_csv = results_df.to_csv(index=False).encode("utf-8")
d1, d2 = st.columns(2)
with d1:
    st.download_button("⬇️ Download cleaned dataset (CSV)", data=cleaned_csv, file_name="cleaned_auto_mpg.csv", mime="text/csv", use_container_width=True)
with d2:
    st.download_button("⬇️ Download model comparison (CSV)", data=metrics_csv, file_name="model_comparison.csv", mime="text/csv", use_container_width=True)

with st.expander("ℹ️ What the preprocessing pipeline does"):
    st.markdown("""
    - Converts common missing markers such as `?` into missing values.
    - Detects numeric columns stored as text where most values can be converted.
    - Imputes missing numeric values with the median and scales numeric features.
    - Imputes missing categorical values with the most frequent category and one-hot encodes categories.
    - Uses `handle_unknown='ignore'` so new categories won't crash prediction.
    - Compares Linear Regression, Random Forest and Extra Trees; selects the highest mean cross-validation R².
    """)
