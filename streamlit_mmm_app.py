import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Market Mix Modeling (MMM)", layout="wide")

st.title("📊 Market Mix Modeling (MMM) – Streamlit App")

st.markdown("""
Upload your marketing data to:
- Explore spend vs sales
- Build a regression-based MMM
- Estimate ROI by channel
- Run simple scenario planning
""")

# -----------------------------
# Upload data
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Basic EDA
    # -----------------------------
    st.subheader("📈 Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.write("Numeric columns detected:", numeric_cols)

    st.line_chart(df[numeric_cols])

    st.subheader("Correlation Matrix")
    st.dataframe(df[numeric_cols].corr())

    # -----------------------------
    # Modeling
    # -----------------------------
    st.subheader("🧠 MMM Regression Model")

    target = st.selectbox("Select target (Sales)", numeric_cols)
    features = st.multiselect(
        "Select marketing spend channels",
        [c for c in numeric_cols if c != target]
    )

    if target and features:
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.write("R²:", round(r2_score(y_test, y_pred), 3))
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        st.write("RMSE:", round(rmse, 2))

        coef_df = pd.DataFrame({
            "Channel": features,
            "Coefficient": model.coef_
        })

        st.subheader("📌 Channel Coefficients")
        st.dataframe(coef_df)

        # -----------------------------
        # ROI
        # -----------------------------
        st.subheader("💰 ROI Estimation")

        spend_means = X.mean()
        coef_df["Avg Spend"] = spend_means.values
        coef_df["Contribution"] = coef_df["Coefficient"] * coef_df["Avg Spend"]
        coef_df["ROI"] = coef_df["Contribution"] / coef_df["Avg Spend"]

        st.dataframe(coef_df)

        # -----------------------------
        # Scenario Planner
        # -----------------------------
        st.subheader("🔮 Scenario Planner")

        scenario = {}
        for col in features:
            scenario[col] = st.slider(
                f"{col} spend",
                float(X[col].min()),
                float(X[col].max()),
                float(X[col].mean())
            )

        scenario_df = pd.DataFrame([scenario])
        predicted_sales = model.predict(scenario_df)[0]

        st.success(f"📈 Predicted Sales under scenario: {predicted_sales:,.2f}")

else:
    st.info("⬆️ Upload a CSV file to get started.")

