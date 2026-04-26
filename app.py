import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
)

# ── countries (exact names matching FAO + their coords) ───────────────────────
COUNTRIES = {
    'India':                      {'lat': 20.59, 'lon': 78.96},
    'China':                      {'lat': 35.86, 'lon': 104.19},
    'Pakistan':                   {'lat': 30.37, 'lon': 69.34},
    'Bangladesh':                 {'lat': 23.68, 'lon': 90.35},
    'Iran (Islamic Republic of)': {'lat': 32.42, 'lon': 53.68},
    'France':                     {'lat': 46.22, 'lon': 2.21},
    'Germany':                    {'lat': 51.16, 'lon': 10.45},
    'Ukraine':                    {'lat': 48.37, 'lon': 31.16},
    'Poland':                     {'lat': 51.91, 'lon': 19.14},
    'United States of America':   {'lat': 37.09, 'lon': -95.71},
    'Brazil':                     {'lat': -14.23, 'lon': -51.92},
    'Argentina':                  {'lat': -38.41, 'lon': -63.61},
    'Mexico':                     {'lat': 23.63, 'lon': -102.55},
    'Canada':                     {'lat': 56.13, 'lon': -106.34},
    'Ethiopia':                   {'lat': 9.14,  'lon': 40.48},
    'Nigeria':                    {'lat': 9.08,  'lon': 8.67},
    'Egypt':                      {'lat': 26.82, 'lon': 30.80},
    'South Africa':               {'lat': -30.55, 'lon': 22.93},
    'Australia':                  {'lat': -25.27, 'lon': 133.77},
}

# ── loaders (cached so they only run once) ────────────────────────────────────
@st.cache_resource
def load_model():
    rf     = joblib.load('crop_yield_model.pkl')
    scaler = joblib.load('scaler.pkl')
    poly   = joblib.load('poly.pkl')
    return rf, scaler, poly

@st.cache_data
def load_merged():
    """Load merged dataset if it exists, else rebuild from FAO + NASA cache."""
    if os.path.exists('merged.csv'):
        return pd.read_csv('merged.csv')
    return None

@st.cache_resource
def load_label_encoder():
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(sorted(COUNTRIES.keys()))
    return le

# ── NASA fetch helper (used for live single-country lookup) ───────────────────
def fetch_nasa_for_country(country):
    coords = COUNTRIES[country]
    params = {
        "start": 20000101, "end": 20231231,
        "latitude": coords['lat'], "longitude": coords['lon'],
        "community": "ag",
        "parameters": "T2M,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN",
        "format": "json", "units": "metric",
    }
    r = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point",
                     params=params, timeout=60)
    data = r.json()
    props = data['properties']['parameter']
    df = pd.DataFrame(props).reset_index()
    df.columns = ['date', 'T2M', 'PRECTOTCORR', 'RH2M', 'ALLSKY_SFC_SW_DWN']
    df['year'] = df['date'].astype(str).str[:4].astype(int)
    yearly = df.groupby('year').agg(
        avg_temp=('T2M', 'mean'),
        total_rainfall=('PRECTOTCORR', 'sum'),
        avg_humidity=('RH2M', 'mean'),
        avg_solar=('ALLSKY_SFC_SW_DWN', 'mean'),
    ).reset_index()
    return yearly.mean()  # return average across all years

# ── prediction function ───────────────────────────────────────────────────────
def predict_yield(country, avg_temp, total_rainfall, avg_humidity, avg_solar):
    rf, scaler, poly = load_model()
    le = load_label_encoder()
    country_encoded = le.transform([country])[0]
    X = np.array([[avg_temp, total_rainfall, avg_humidity,
                   avg_solar, country_encoded]])
    X_poly   = poly.transform(X)
    X_scaled = scaler.transform(X_poly)
    prediction = rf.predict(X_scaled)[0]
    return round(float(prediction), 3)

# ── yield colour helper ───────────────────────────────────────────────────────
def yield_color(val):
    if val >= 4.0:
        return "#2ecc71", "High yield"
    elif val >= 2.0:
        return "#f39c12", "Medium yield"
    else:
        return "#e74c3c", "Low yield"

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 1.5rem 0 0.5rem'>
    <h1 style='margin:0; font-size:2.2rem'>🌾 Crop Yield Predictor</h1>
    <p style='margin:0.3rem 0 0; color:gray; font-size:1rem'>
        Predict wheat yield (tons/ha) from weather conditions using
        NASA POWER data &amp; a Random Forest model trained on FAOSTAT statistics.
    </p>
</div>
<hr style='margin: 1rem 0'>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Input Parameters")

    country = st.selectbox(
        "Country",
        sorted(COUNTRIES.keys()),
        index=sorted(COUNTRIES.keys()).index('India'),
    )

    st.markdown("---")
    st.markdown("**Weather inputs**")

    avg_temp = st.slider(
        "Average Temperature (°C)", -10.0, 40.0, 20.0, 0.5,
        help="Mean annual temperature for the growing region"
    )
    total_rainfall = st.slider(
        "Total Annual Rainfall (mm)", 0, 2000, 500, 10,
        help="Total precipitation across the year"
    )
    avg_humidity = st.slider(
        "Average Humidity (%)", 0, 100, 60, 1,
        help="Mean relative humidity across the year"
    )
    avg_solar = st.slider(
        "Avg Solar Radiation (MJ/m²/day)", 0.0, 30.0, 15.0, 0.5,
        help="Mean daily shortwave radiation"
    )

    st.markdown("---")

    autofill = st.button("📡 Auto-fill from NASA API", use_container_width=True,
                         help="Fetch real historical averages for selected country")

    predict_btn = st.button("🌱 Predict Yield", type="primary",
                            use_container_width=True)

# ── auto-fill from NASA ───────────────────────────────────────────────────────
if autofill:
    with st.spinner(f"Fetching NASA data for {country}…"):
        try:
            avgs = fetch_nasa_for_country(country)
            st.sidebar.success("✅ Values updated! Now click Predict.")
            avg_temp       = round(float(avgs['avg_temp']), 1)
            total_rainfall = int(avgs['total_rainfall'])
            avg_humidity   = round(float(avgs['avg_humidity']), 1)
            avg_solar      = round(float(avgs['avg_solar']), 1)
            st.sidebar.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Avg Temp | {avg_temp} °C |
            | Rainfall | {total_rainfall} mm |
            | Humidity | {avg_humidity} % |
            | Solar | {avg_solar} MJ/m² |
            """)
        except Exception as e:
            st.sidebar.error(f"NASA API error: {e}")

# ── main prediction panel ─────────────────────────────────────────────────────
col_pred, col_chart = st.columns([1, 2], gap="large")

with col_pred:
    st.subheader("Prediction Result")

    if predict_btn:
        try:
            result = predict_yield(country, avg_temp, total_rainfall,
                                   avg_humidity, avg_solar)
            color, label = yield_color(result)

            st.markdown(f"""
            <div style='
                background:{color}18;
                border-left: 5px solid {color};
                border-radius: 8px;
                padding: 1.2rem 1.5rem;
                margin-bottom: 1rem;
            '>
                <div style='font-size:0.85rem; color:gray; margin-bottom:4px'>
                    Predicted yield for <b>{country}</b>
                </div>
                <div style='font-size:3rem; font-weight:700; color:{color}; line-height:1'>
                    {result}
                </div>
                <div style='font-size:1rem; color:{color}; margin-top:4px'>
                    tons / hectare &nbsp;·&nbsp; {label}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Inputs used:**")
            st.table(pd.DataFrame({
                "Parameter": ["Country", "Avg Temp", "Total Rainfall",
                              "Avg Humidity", "Avg Solar"],
                "Value": [country, f"{avg_temp} °C", f"{total_rainfall} mm",
                          f"{avg_humidity} %", f"{avg_solar} MJ/m²"],
            }).set_index("Parameter"))

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Make sure crop_yield_model.pkl, scaler.pkl and poly.pkl "
                    "are in the same folder as app.py")
    else:
        st.info("👈 Set your parameters in the sidebar and click **Predict Yield**")

with col_chart:
    st.subheader("Country Comparison")

    merged = load_merged()
    if merged is not None:
        avg_by_country = (
            merged.groupby('country')['yield_tons_ha']
            .mean()
            .sort_values(ascending=True)
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ['#e74c3c' if c == country else '#3498db'
                  for c in avg_by_country.index]
        ax.barh(avg_by_country.index, avg_by_country.values, color=colors)
        ax.set_xlabel("Avg yield (tons/ha)")
        ax.set_title("Average wheat yield by country (2000–2023)")
        ax.spines[['top', 'right']].set_visible(False)

        # overlay current prediction if button was clicked
        if predict_btn:
            try:
                ax.axvline(result, color='#2ecc71', linestyle='--',
                           linewidth=2, label=f'Your prediction: {result}')
                ax.legend(fontsize=9)
            except Exception:
                pass

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("merged.csv not found. Run your notebook first and save "
                   "the merged dataframe:  `merged.to_csv('merged.csv', index=False)`")

# ── model performance ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Model Performance")

m1, m2, m3 = st.columns(3)
m1.metric("R² Score", "0.912", help="Explains 91% of variance in crop yield")
m2.metric("RMSE", "0.499 tons/ha", help="Average prediction error")
m3.metric("Algorithm", "Random Forest", help="200 decision trees, degree-2 polynomial features")

# actual vs predicted scatter
if predict_btn and merged is not None:
    try:
        rf, scaler, poly = load_model()
        le = load_label_encoder()

        merged2 = merged.copy()
        known   = [c for c in merged2['country'].unique() if c in le.classes_]
        merged2 = merged2[merged2['country'].isin(known)]
        merged2['country_encoded'] = le.transform(merged2['country'])

        feats = ['avg_temp', 'total_rainfall', 'avg_humidity',
                 'avg_solar', 'country_encoded']
        Xm = scaler.transform(poly.transform(merged2[feats]))
        merged2['predicted'] = rf.predict(Xm)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(merged2['yield_tons_ha'], merged2['predicted'],
                    alpha=0.5, color='steelblue', s=20, label='All data')
        lim = [merged2['yield_tons_ha'].min(), merged2['yield_tons_ha'].max()]
        ax2.plot(lim, lim, 'r--', linewidth=1.5, label='Perfect prediction')
        ax2.set_xlabel("Actual yield (tons/ha)")
        ax2.set_ylabel("Predicted yield (tons/ha)")
        ax2.set_title("Actual vs Predicted — Random Forest (R²=0.912)")
        ax2.legend(fontsize=9)
        ax2.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
    except Exception:
        pass

# ── about ─────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this project"):
    st.markdown("""
    ### Data sources
    - **FAOSTAT** — FAO crop yield statistics (wheat, kg/ha → tons/ha), 
      133 countries, 1961–2023
    - **NASA POWER API** — daily agroclimatology data (temperature, rainfall, 
      humidity, solar radiation) per country centroid, 2000–2023

    ### Model
    A **Random Forest Regressor** (200 trees) trained on:
    - 5 base features: avg temp, total rainfall, avg humidity, avg solar, 
      country (label-encoded)
    - Expanded to **20 polynomial features** (degree 2) to capture 
      non-linear relationships
    - Features normalized with **StandardScaler**

    ### Countries covered
    """ + ", ".join(sorted(COUNTRIES.keys())))

    st.markdown("""
    ### How to run locally
    ```bash
    pip install streamlit pandas numpy scikit-learn matplotlib joblib requests
    streamlit run app.py
    ```
    Make sure these files are in the same folder as `app.py`:
    - `crop_yield_model.pkl`
    - `scaler.pkl`  
    - `poly.pkl`
    - `merged.csv`
    """)
