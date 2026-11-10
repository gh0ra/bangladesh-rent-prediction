import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

COLS = ["area_sqft", "beds_n", "bath_n", "type", "city", "locality"]
CAT = ["type", "city", "locality"]

st.set_page_config(page_title="Bangladesh Rent Predictor", page_icon="🏠")


@st.cache_resource
def load():
    models = joblib.load("models/rent_range.joblib")
    data = pd.read_csv("data/clean.csv")
    X = data[COLS]
    cats = {c: sorted(X[c].unique()) for c in CAT}

    def encode(df):
        out = df.copy()
        for c in CAT:
            out[c] = pd.Categorical(out[c], categories=cats[c]).codes
        return out.to_numpy(dtype=float)

    def f(a):
        df = pd.DataFrame(a, columns=COLS)
        for c in CAT:
            df[c] = [cats[c][int(i)] for i in df[c]]
        return models["mid"].predict(df)

    bg = encode(X.sample(100, random_state=42))
    explainer = shap.Explainer(f, shap.maskers.Independent(bg), feature_names=COLS)
    return models, data, explainer, encode

models, data, explainer, encode = load()

st.title("Bangladesh Rent Predictor")
st.write("Estimate the monthly rent of a property from listing data.")

c1, c2 = st.columns(2)
area = c1.number_input("Area (sqft)", 100, 10000, 1000, step=50)
beds = c2.number_input("Bedrooms", 1, 10, 3)
baths = c1.number_input("Bathrooms", 1, 10, 2)
ptype = c2.selectbox("Type", sorted(data["type"].unique()))
city = c1.selectbox("City", sorted(data["city"].unique()))
localities = sorted(data.loc[data["city"] == city, "locality"].unique())
locality = c2.selectbox("Locality", localities)

if st.button("Predict"):
    row = pd.DataFrame([[area, beds, baths, ptype, city, locality]], columns=COLS)

    mid = np.expm1(models["mid"].predict(row)[0])
    low = min(np.expm1(models["low"].predict(row)[0]), mid)
    high = max(np.expm1(models["high"].predict(row)[0]), mid)

    st.metric("Estimated rent", f"{mid:,.0f} BDT")
    st.write(f"Likely range: **{low:,.0f} to {high:,.0f} BDT**")

    sv = explainer(encode(row))
    effects = sv.values[0] * mid
    labels = [f"{c} = {row[c].iloc[0]}" for c in COLS]
    colors = ["green" if v > 0 else "red" for v in effects]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(labels, effects, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Effect on rent (BDT)")
    st.subheader("Why this estimate")
    st.pyplot(fig)
    st.caption("Approximate: each SHAP value (log scale) is multiplied by the predicted rent.")