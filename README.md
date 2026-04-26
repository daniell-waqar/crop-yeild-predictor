# Crop Yield Predictor 🌾

A machine learning web app that predicts wheat yield (tons per hectare) based on weather conditions and location.

I built this as a beginner ML project to learn how to collect real data from APIs, clean it, train a model, and deploy it as a web app.

---

## What it does

You select a country and enter weather conditions like temperature, rainfall, humidity, and solar radiation — and the app predicts how many tons of wheat per hectare that region would likely produce.

---

## How I built it

**Data Collection**
- Downloaded historical wheat yield data from [FAOSTAT](https://www.fao.org/faostat/) (covers 20 countries, 2000–2023)
- Fetched daily weather data from [NASA POWER API](https://power.larc.nasa.gov/) for each country and averaged it into yearly values

**Data Cleaning**
- Merged both datasets on country + year
- Converted yield from kg/ha to tons/ha
- Handled missing values

**Machine Learning**
- Used polynomial features (degree 2) to capture non-linear relationships
- Tried Ridge Regression first (R² = 0.75), then switched to Random Forest
- Final model: Random Forest with 200 trees
- R² Score: 0.912 — RMSE: 0.499 tons/ha

**Web App**
- Built with Streamlit
- Deployed on Streamlit Cloud

---

## Tech stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- NASA POWER API
- FAOSTAT dataset

---

## How to run locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install -r requirements.txt
streamlit run app.py
```

---

## Countries covered

Argentina, Australia, Bangladesh, Brazil, Canada, Egypt, Ethiopia,
France, Germany, India, Iran, Mexico, Nigeria, Pakistan, Poland,
South Africa, Ukraine, United States, China, Turkey

---

## What I learned

- How to collect and merge data from multiple real-world sources
- How to engineer features for machine learning
- Why Random Forest outperforms linear models on complex data
- How to deploy a machine learning app for free
