# Bangladesh Rent Prediction

Predicts monthly apartment rent (BDT) in Dhaka and Chattogram from area, bedrooms, bathrooms, property type and locality.

## Data

7,557 rent listings from bproperty ([8K+ Property Listing Data in Bangladesh](https://www.kaggle.com/datasets/ijajdatanerd/property-listing-data-in-bangladesh)). After removing duplicates and impossible rows (e.g. 36 bedrooms, 31,500 sqft), 7,448 listings remain, covering 50 localities.

To rerun: download the CSV, save it as `data/listings.csv`, and run the notebooks in order.

## What I did

1. **EDA** (`notebooks/01_eda.ipynb`): parsed price, area, beds and bath from text, cleaned outliers, and looked at rent by city, area and locality.
2. **Modeling** (`notebooks/02_modeling.ipynb`): trained Ridge, Random Forest and HistGradientBoosting on log(rent) with a Scikit-learn pipeline, compared them with 5-fold CV, and tuned the best one with random search.

## Results (held-out test set, 20%)

| Model           | R²     | MAE (BDT) |
| --------------- | ------ | --------- |
| Median baseline | -0.055 | 15,655    |
| Ridge           | 0.874  | 6,068     |
| Random Forest   | 0.887  | 5,203     |
| HistGB          | 0.896  | 5,344     |
| HistGB (tuned)  | 0.899  | 5,298     |

Median error is about 9.8% of the actual rent.

Example: 1,200 sqft, 3 beds, 2 baths, Dhaka (Mirpur) → about 18,000 BDT/month.

## Findings

- Area is the most important feature, then locality. Type adds almost nothing.
- The model is weakest on high-rent flats (17% error against 7 to 10% for the rest), especially in Gulshan, Banani and Baridhara.
- Furnishing, floor and building age aren't in the data, which probably explains some of the worst misses.

## Project structure

- `notebooks/`: EDA and modeling
- `models/rent_model.joblib`: trained pipeline
- `requirements.txt`

## Setup

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
