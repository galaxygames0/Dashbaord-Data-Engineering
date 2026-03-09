import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(
    page_title="Vehicle Emissions and Scenario Dashboard",
    layout="wide"
)

st.title("Vehicle Emissions, Demand, Revenue, and Scenario Dashboard")
st.caption("Interactive dashboard for the final coursework modelling framework")

# ============================================================
# HELPERS
# ============================================================

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df.drop_duplicates().copy()
    return df

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "Make",
        "Model",
        "Vehicle Class",
        "Engine Size(L)",
        "Cylinders",
        "Transmission",
        "Fuel Type",
        "CO2 Emissions(g/km)"
    ]

    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    df = df[required_cols].dropna().copy()

    df["vehicle_key"] = (
        df["Vehicle Class"].astype(str) + "_" +
        df["Fuel Type"].astype(str) + "_" +
        df["Transmission"].astype(str) + "_" +
        pd.cut(
            df["Engine Size(L)"],
            bins=[0, 1.5, 2.5, 3.5, 5.0, 10.0],
            labels=["small", "medium", "large", "xlarge", "performance"]
        ).astype(str)
    )

    return df

def add_proxy_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Units Sold" in df.columns:
        df["demand_target"] = df["Units Sold"]
    else:
        class_boost = {
            "COMPACT": 1.20,
            "MID-SIZE": 1.10,
            "SUBCOMPACT": 1.05,
            "SUV - SMALL": 1.15,
            "SUV - STANDARD": 0.95,
            "FULL-SIZE": 0.85,
            "PICKUP TRUCK - SMALL": 0.90,
            "PICKUP TRUCK - STANDARD": 0.88,
            "MINIVAN": 0.92,
            "TWO-SEATER": 0.55,
            "STATION WAGON - SMALL": 0.95,
            "STATION WAGON - MID-SIZE": 0.88,
            "VAN - CARGO": 0.72,
            "VAN - PASSENGER": 0.78,
            "MINICOMPACT": 0.60
        }

        fuel_boost = {
            "X": 1.00,
            "Z": 0.92,
            "D": 0.98,
            "E": 0.80,
            "N": 0.70
        }

        np.random.seed(42)
        base_demand = 5000
        emissions_penalty = (df["CO2 Emissions(g/km)"] - df["CO2 Emissions(g/km)"].min()) * 8
        engine_penalty = df["Engine Size(L)"] * 180
        cyl_penalty = df["Cylinders"] * 35
        class_multiplier = df["Vehicle Class"].map(class_boost).fillna(0.90)
        fuel_multiplier = df["Fuel Type"].map(fuel_boost).fillna(0.90)
        noise = np.random.normal(0, 250, len(df))

        demand_proxy = (
            (base_demand - emissions_penalty - engine_penalty - cyl_penalty)
            * class_multiplier
            * fuel_multiplier
            + noise
        )

        df["demand_target"] = np.maximum(150, demand_proxy).round().astype(int)

    if "Revenue" in df.columns:
        df["revenue_target"] = df["Revenue"]
    else:
        price_base = 18000
        engine_price = df["Engine Size(L)"] * 4500
        cyl_price = df["Cylinders"] * 900
        premium_fuel_price = np.where(df["Fuel Type"] == "Z", 3500, 0)
        diesel_price = np.where(df["Fuel Type"] == "D", 2500, 0)
        luxury_class_price = np.where(
            df["Vehicle Class"].isin(["FULL-SIZE", "MINICOMPACT", "TWO-SEATER", "SUV - STANDARD"]),
            12000,
            0
        )
        estimated_price = (
            price_base + engine_price + cyl_price +
            premium_fuel_price + diesel_price + luxury_class_price
        )
        df["revenue_target"] = (df["demand_target"] * estimated_price).round(2)

    return df

def build_preprocessors(numeric_features, categorical_features):
    numeric_preprocessor_basic = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    numeric_preprocessor_poly = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False))
    ])

    categorical_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor_basic = ColumnTransformer([
        ("num", numeric_preprocessor_basic, numeric_features),
        ("cat", categorical_preprocessor, categorical_features)
    ])

    preprocessor_poly = ColumnTransformer([
        ("num", numeric_preprocessor_poly, numeric_features),
        ("cat", categorical_preprocessor, categorical_features)
    ])

    return preprocessor_basic, preprocessor_poly

def build_model_dict(preprocessor_basic, preprocessor_poly):
    return {
        "Linear Regression": Pipeline([
            ("preprocessor", preprocessor_basic),
            ("model", LinearRegression())
        ]),
        "Ridge Regression": Pipeline([
            ("preprocessor", preprocessor_poly),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5))
        ]),
        "Lasso Regression": Pipeline([
            ("preprocessor", preprocessor_poly),
            ("model", LassoCV(alphas=np.logspace(-3, 1, 100), cv=5, random_state=42, max_iter=20000))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor_basic),
            ("model", RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", preprocessor_basic),
            ("model", GradientBoostingRegressor(random_state=42))
        ])
    }

@st.cache_resource
def run_all_models(df: pd.DataFrame):
    emissions_target = "CO2 Emissions(g/km)"
    demand_target = "demand_target"
    revenue_target = "revenue_target"

    feature_cols = [
        "Engine Size(L)",
        "Cylinders",
        "Vehicle Class",
        "Transmission",
        "Fuel Type"
    ]

    numeric_features = ["Engine Size(L)", "Cylinders"]
    categorical_features = ["Vehicle Class", "Transmission", "Fuel Type"]

    preprocessor_basic, preprocessor_poly = build_preprocessors(
        numeric_features, categorical_features
    )

    def evaluate_task(target_name):
        X = df[feature_cols].copy()
        y = df[target_name].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = build_model_dict(preprocessor_basic, preprocessor_poly)

        results = []
        fitted_models = {}
        predictions = {}
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        for model_name, pipeline in models.items():
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            cv_rmse_scores = np.sqrt(
                -cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="neg_mean_squared_error"
                )
            )

            row = {
                "Technique": model_name,
                "Test RMSE": round(float(rmse), 3),
                "Test MAE": round(float(mae), 3),
                "Test R²": round(float(r2), 4),
                "CV RMSE Mean": round(float(cv_rmse_scores.mean()), 3),
                "CV RMSE Std": round(float(cv_rmse_scores.std()), 3)
            }

            fitted_model = pipeline.named_steps["model"]
            if model_name == "Ridge Regression":
                row["Selected Alpha"] = float(getattr(fitted_model, "alpha_", np.nan))
            elif model_name == "Lasso Regression":
                row["Selected Alpha"] = float(getattr(fitted_model, "alpha_", np.nan))
                row["Non-zero Coefs"] = int(np.sum(fitted_model.coef_ != 0))
            else:
                row["Selected Alpha"] = np.nan
                row["Non-zero Coefs"] = np.nan

            results.append(row)
            fitted_models[model_name] = pipeline
            predictions[model_name] = y_pred

        results_df = pd.DataFrame(results).sort_values(by="Test RMSE").reset_index(drop=True)
        best_name = results_df.loc[0, "Technique"]
        best_model = fitted_models[best_name]

        sample_df = X_test.copy().reset_index(drop=True)
        sample_df["Actual"] = y_test.reset_index(drop=True)
        sample_df["Predicted"] = pd.Series(best_model.predict(X_test))
        sample_df["Residual"] = sample_df["Actual"] - sample_df["Predicted"]

        return {
            "results_df": results_df,
            "best_name": best_name,
            "best_model": best_model,
            "sample_df": sample_df,
            "X_test": X_test,
            "y_test": y_test,
            "predictions": predictions
        }

    out = {
        "emissions": evaluate_task(emissions_target),
        "demand": evaluate_task(demand_target),
        "revenue": evaluate_task(revenue_target),
        "feature_cols": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }

    return out

def actual_vs_pred_figure(y_true, y_pred, title, y_label):
    plot_df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    fig = px.scatter(
        plot_df,
        x="Actual",
        y="Predicted",
        title=title,
        opacity=0.55
    )
    low = min(plot_df["Actual"].min(), plot_df["Predicted"].min())
    high = max(plot_df["Actual"].max(), plot_df["Predicted"].max())
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="Perfect fit"
        )
    )
    fig.update_layout(
        xaxis_title=f"Actual {y_label}",
        yaxis_title=f"Predicted {y_label}"
    )
    return fig

def residual_figure(y_true, y_pred, title):
    residuals = y_true - y_pred
    plot_df = pd.DataFrame({"Predicted": y_pred, "Residual": residuals})
    fig = px.scatter(
        plot_df,
        x="Predicted",
        y="Residual",
        title=title,
        opacity=0.55
    )
    fig.add_hline(y=0)
    return fig

def metric_bar_figure(results_df, metric, title):
    fig = px.bar(
        results_df,
        x="Technique",
        y=metric,
        title=title
    )
    fig.update_layout(xaxis_tickangle=-25)
    return fig

def feature_importance_figure(best_pipeline, numeric_features, categorical_features, title):
    model = best_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None, None

    preprocessor = best_pipeline.named_steps["preprocessor"]
    cat_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_names)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(15)

    fig = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title=title
    )
    return fig, importance_df

def run_scenario(df, best_emissions_model, best_demand_model, best_revenue_model, feature_cols, scenario_mix):
    scenario_rows = []

    for vehicle_class, share in scenario_mix.items():
        subset = df[df["Vehicle Class"] == vehicle_class].copy()
        if subset.empty:
            continue

        representative_row = subset.iloc[[0]].copy()
        representative_row["Engine Size(L)"] = subset["Engine Size(L)"].mean()
        representative_row["Cylinders"] = round(subset["Cylinders"].mean())
        representative_row["Transmission"] = subset["Transmission"].mode()[0]
        representative_row["Fuel Type"] = subset["Fuel Type"].mode()[0]
        representative_row["Vehicle Class"] = vehicle_class

        X_rep = representative_row[feature_cols]

        pred_emissions = float(best_emissions_model.predict(X_rep)[0])
        pred_demand = float(best_demand_model.predict(X_rep)[0])
        pred_revenue = float(best_revenue_model.predict(X_rep)[0])

        scenario_rows.append({
            "Vehicle Class": vehicle_class,
            "Production Share": share,
            "Predicted CO2 per Vehicle": pred_emissions,
            "Predicted Demand": pred_demand,
            "Predicted Revenue": pred_revenue,
            "Weighted CO2": pred_emissions * share,
            "Weighted Demand": pred_demand * share,
            "Weighted Revenue": pred_revenue * share
        })

    scenario_df = pd.DataFrame(scenario_rows)

    summary = {
        "Fleet average CO2": float(scenario_df["Weighted CO2"].sum()),
        "Fleet weighted demand": float(scenario_df["Weighted Demand"].sum()),
        "Fleet weighted revenue": float(scenario_df["Weighted Revenue"].sum())
    }

    return scenario_df, summary

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Inputs")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.subheader("Scenario mix")
compact = st.sidebar.slider("Compact share", 0.0, 1.0, 0.30, 0.01)
mid_size = st.sidebar.slider("Mid-size share", 0.0, 1.0, 0.20, 0.01)
small_suv = st.sidebar.slider("Small SUV share", 0.0, 1.0, 0.25, 0.01)
subcompact = st.sidebar.slider("Subcompact share", 0.0, 1.0, 0.10, 0.01)
standard_suv = st.sidebar.slider("Standard SUV share", 0.0, 1.0, 0.10, 0.01)
two_seater = st.sidebar.slider("Two-seater share", 0.0, 1.0, 0.05, 0.01)

scenario_total = compact + mid_size + small_suv + subcompact + standard_suv + two_seater
st.sidebar.write(f"Scenario share total: {scenario_total:.2f}")

# ============================================================
# MAIN
# ============================================================

if uploaded_file is None:
    st.info("Upload your CO2 emissions CSV to begin.")
    st.stop()

try:
    raw_df = load_data(uploaded_file)
    df = prepare_dataframe(raw_df)
    df = add_proxy_targets(df)

    model_outputs = run_all_models(df)

    st.success("Data loaded and models trained successfully.")

except Exception as e:
    st.error(str(e))
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Emissions Model",
    "Demand Model",
    "Revenue Model",
    "Scenario Simulator",
    "Sample Predictions"
])

with tab1:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Unique vehicle classes", df["Vehicle Class"].nunique())

    st.dataframe(df.head(10), use_container_width=True)

    overview_df = pd.DataFrame({
        "Task": ["Emissions", "Demand", "Revenue"],
        "Best Model": [
            model_outputs["emissions"]["best_name"],
            model_outputs["demand"]["best_name"],
            model_outputs["revenue"]["best_name"]
        ]
    })
    st.subheader("Best model by task")
    st.dataframe(overview_df, use_container_width=True)

with tab2:
    st.subheader("Emissions model comparison")
    results_df = model_outputs["emissions"]["results_df"]
    st.dataframe(results_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            metric_bar_figure(results_df, "Test RMSE", "Emissions model, Test RMSE by technique"),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            metric_bar_figure(results_df, "Test R²", "Emissions model, Test R² by technique"),
            use_container_width=True
        )

    best_model = model_outputs["emissions"]["best_model"]
    X_test = model_outputs["emissions"]["X_test"]
    y_test = model_outputs["emissions"]["y_test"]
    y_pred = best_model.predict(X_test)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            actual_vs_pred_figure(y_test, y_pred, "Emissions, actual vs predicted", "CO2"),
            use_container_width=True
        )
    with col4:
        st.plotly_chart(
            residual_figure(y_test, y_pred, "Emissions, residual plot"),
            use_container_width=True
        )

    fig, importance_df = feature_importance_figure(
        best_model,
        model_outputs["numeric_features"],
        model_outputs["categorical_features"],
        "Emissions feature importance"
    )
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(importance_df, use_container_width=True)

with tab3:
    st.subheader("Demand model comparison")
    results_df = model_outputs["demand"]["results_df"]
    st.dataframe(results_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            metric_bar_figure(results_df, "Test RMSE", "Demand model, Test RMSE by technique"),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            metric_bar_figure(results_df, "Test R²", "Demand model, Test R² by technique"),
            use_container_width=True
        )

    best_model = model_outputs["demand"]["best_model"]
    X_test = model_outputs["demand"]["X_test"]
    y_test = model_outputs["demand"]["y_test"]
    y_pred = best_model.predict(X_test)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            actual_vs_pred_figure(y_test, y_pred, "Demand, actual vs predicted", "Demand"),
            use_container_width=True
        )
    with col4:
        st.plotly_chart(
            residual_figure(y_test, y_pred, "Demand, residual plot"),
            use_container_width=True
        )

    fig, importance_df = feature_importance_figure(
        best_model,
        model_outputs["numeric_features"],
        model_outputs["categorical_features"],
        "Demand feature importance"
    )
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(importance_df, use_container_width=True)

with tab4:
    st.subheader("Revenue model comparison")
    results_df = model_outputs["revenue"]["results_df"]
    st.dataframe(results_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            metric_bar_figure(results_df, "Test RMSE", "Revenue model, Test RMSE by technique"),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            metric_bar_figure(results_df, "Test R²", "Revenue model, Test R² by technique"),
            use_container_width=True
        )

    best_model = model_outputs["revenue"]["best_model"]
    X_test = model_outputs["revenue"]["X_test"]
    y_test = model_outputs["revenue"]["y_test"]
    y_pred = best_model.predict(X_test)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            actual_vs_pred_figure(y_test, y_pred, "Revenue, actual vs predicted", "Revenue"),
            use_container_width=True
        )
    with col4:
        st.plotly_chart(
            residual_figure(y_test, y_pred, "Revenue, residual plot"),
            use_container_width=True
        )

    fig, importance_df = feature_importance_figure(
        best_model,
        model_outputs["numeric_features"],
        model_outputs["categorical_features"],
        "Revenue feature importance"
    )
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(importance_df, use_container_width=True)

with tab5:
    st.subheader("Scenario simulator")

    if abs(scenario_total - 1.0) > 1e-6:
        st.warning("Scenario shares must sum to 1.00 to run the simulator.")
    else:
        scenario_mix = {
            "COMPACT": compact,
            "MID-SIZE": mid_size,
            "SUV - SMALL": small_suv,
            "SUBCOMPACT": subcompact,
            "SUV - STANDARD": standard_suv,
            "TWO-SEATER": two_seater
        }

        scenario_df, summary = run_scenario(
            df,
            model_outputs["emissions"]["best_model"],
            model_outputs["demand"]["best_model"],
            model_outputs["revenue"]["best_model"],
            model_outputs["feature_cols"],
            scenario_mix
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Fleet average CO2", f"{summary['Fleet average CO2']:.2f} g/km")
        c2.metric("Fleet weighted demand", f"{summary['Fleet weighted demand']:.0f}")
        c3.metric("Fleet weighted revenue", f"£{summary['Fleet weighted revenue']:,.2f}")

        st.dataframe(scenario_df, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            fig = px.bar(
                scenario_df,
                x="Vehicle Class",
                y="Predicted CO2 per Vehicle",
                title="Predicted emissions by vehicle class"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                scenario_df,
                x="Vehicle Class",
                y="Predicted Demand",
                title="Predicted demand by vehicle class"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            fig = px.bar(
                scenario_df,
                x="Vehicle Class",
                y="Predicted Revenue",
                title="Predicted revenue by vehicle class"
            )
            st.plotly_chart(fig, use_container_width=True)

        csv = scenario_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download scenario results as CSV",
            data=csv,
            file_name="scenario_results.csv",
            mime="text/csv"
        )

with tab6:
    st.subheader("Sample predictions")
    subtab1, subtab2, subtab3 = st.tabs(["Emissions", "Demand", "Revenue"])

    with subtab1:
        st.dataframe(model_outputs["emissions"]["sample_df"].head(20), use_container_width=True)

    with subtab2:
        st.dataframe(model_outputs["demand"]["sample_df"].head(20), use_container_width=True)

    with subtab3:
        st.dataframe(model_outputs["revenue"]["sample_df"].head(20), use_container_width=True)
