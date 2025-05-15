# app.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="🏠 House Price Prediction", layout="wide")

# ========== STEP 1: LOAD PICKLE FILES ========== #
st.title("🏠 House Price Prediction - Model Evaluation App")
st.subheader("📦 Loading model and data...")

try:
    with open('final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('final_train.pkl', 'rb') as f:
        final_train = pickle.load(f)
    with open('test_reduced.pkl', 'rb') as f:
        test_reduced = pickle.load(f)
    with open('X_reduced_columns.pkl', 'rb') as f:
        selected_columns = pickle.load(f)
    with open('model_results.pkl', 'rb') as f:
        model_results = pickle.load(f)
    st.success("✅ All files loaded successfully.")
except FileNotFoundError as e:
    st.error(f"❌ Missing file: {e.filename}")
    st.stop()

# ========== STEP 2: MODEL EVALUATION ========== #

st.subheader("🔍 Model Evaluation")

X = final_train.drop('SalePrice', axis=1)
y = final_train['SalePrice']

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred_log = model.predict(X_val)
y_pred = np.expm1(y_pred_log)
y_val_original = np.expm1(y_val)

rmse = np.sqrt(mean_squared_error(y_val_original, y_pred))
mae = mean_absolute_error(y_val_original, y_pred)
r2 = r2_score(y_val_original, y_pred)

st.metric("RMSE", f"{rmse:.2f}")
st.metric("MAE", f"{mae:.2f}")
st.metric("R² Score", f"{r2:.4f}")

# ========== STEP 3: PREDICT ON TEST SET ========== #

st.subheader("📈 Predict on Test Set")

test_scaled = scaler.transform(test_reduced[selected_columns])
test_preds_log = model.predict(test_scaled)
test_preds = np.expm1(test_preds_log) * 1e5  # converting lakhs to rupees

test_ids = test_reduced.index if 'Id' not in test_reduced.columns else test_reduced['Id']

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice (in lakhs)': test_preds
})

st.dataframe(submission.head())

# 📁 Download as CSV
csv_buffer = BytesIO()
submission.to_csv(csv_buffer, index=False)
st.download_button("⬇️ Download Predictions CSV", data=csv_buffer.getvalue(),
                   file_name="local_submission.csv", mime='text/csv')

# ========== STEP 4: SHOW PLOTS ========== #

st.subheader("🖼️ Visualizations")

try:
    col1, col2 = st.columns(2)

    with col1:
        st.image('correlation_heatmap.png', caption='Correlation Heatmap', use_column_width=True)

    with col2:
        st.image('actual_vs_predicted.png', caption='Actual vs Predicted', use_column_width=True)

except FileNotFoundError:
    st.warning("⚠️ Plot images not found.")

st.success("✅ All steps completed successfully.")


# ========== STEP 5: CUSTOM USER INPUT PREDICTION ========== #
st.subheader("🎯 Predict House Price from User Input")

# --- Friendly Labels for Features ---
feature_labels = {
    'OverallQual': "Overall Quality (1=Very Poor to 10=Very Excellent)",
    'OverallCond': "Overall Condition (1=Very Poor to 10=Very Excellent)",
    'GrLivArea': "Above Ground Living Area (sq ft)",
    'GarageCars': "Garage Capacity (Cars)",
    'GarageArea': "Garage Area (sq ft)",
    'TotalBsmtSF': "Total Basement Area (sq ft)",
    '1stFlrSF': "1st Floor Area (sq ft)",
    '2ndFlrSF': "2nd Floor Area (sq ft)",
    'YearBuilt': "Year Built",
    'FullBath': "Full Bathrooms",
    'HalfBath': "Half Bathrooms",
    'TotRmsAbvGrd': "Total Rooms Above Ground",
    'Fireplaces': "Number of Fireplaces",
    'MasVnrArea': "Masonry Veneer Area (sq ft)",
    'BsmtFullBath': "Basement Full Bathrooms",
    'BsmtUnfSF': "Unfinished Basement Area (sq ft)",
    'BedroomAbvGr': "Bedrooms Above Ground",
    'KitchenAbvGr': "Kitchens Above Ground",
    'WoodDeckSF': "Wood Deck Area (sq ft)",
    'OpenPorchSF': "Open Porch Area (sq ft)",
    'MoSold': "Month of Sale (1-12)",
    'YrSold': "Year Sold",
    'MSSubClass': "Building Class (e.g. 1-story, 2-story, duplex)",
    'LotArea': "Lot Area (Total area in square feet)",
    'BsmtFinSF1': "Basement Finished Area Type 1 (sq ft)"
}

# --- MSSubClass Category Labels ---
ms_subclass_options = {
    20: "1-STORY 1946 & NEWER",
    30: "1-STORY 1945 & OLDER",
    40: "1-STORY W/FINISHED ATTIC",
    45: "1.5-STORY UNFINISHED",
    50: "1.5-STORY FINISHED",
    60: "2-STORY 1946 & NEWER",
    70: "2-STORY 1945 & OLDER",
    75: "2.5-STORY ALL AGES",
    80: "SPLIT OR MULTI-LEVEL",
    85: "SPLIT FOYER",
    90: "DUPLEX",
    120: "1-STORY PUD 1946 & NEWER",
    150: "1.5-STORY PUD ALL AGES",
    160: "2-STORY PUD 1946 & NEWER",
    180: "PUD - MULTILEVEL",
    190: "2 FAMILY CONVERSION"
}

# --- Friendly name for dummy features ---
def friendly_label_from_dummy(dummy_name):
    if '_' not in dummy_name:
        return dummy_name
    base, level = dummy_name.split('_', 1)
    base_labels = {
        'MSZoning': 'Zoning',
        'KitchenQual': 'Kitchen Quality',
        'GarageFinish': 'Garage Finish',
        'Neighborhood': 'Neighborhood',
        'ExterQual': 'Exterior Quality',
        'ExterCond': 'Exterior Condition',
        'Functional': 'Home Functionality',
        'FireplaceQu': 'Fireplace Quality',
        'BsmtQual': 'Basement Quality',
        'BsmtCond': 'Basement Condition',
        'BsmtExposure': 'Basement Exposure',
        'GarageCond': 'Garage Condition',
        'GarageQual': 'Garage Quality',
        'GarageType': 'Garage Type',
        'SaleCondition': 'Sale Condition',
        'HouseStyle': 'House Style',
        'BldgType': 'Building Type',
        'RoofStyle': 'Roof Style',
    }
    label = base_labels.get(base, base)
    return f"{label}: {level}"

# --- Layout with Expander and Reset ---
with st.expander("🔧 Try Your Own Prediction"):
    # if st.button("🔄 Reset "):
    #     for key in st.session_state.keys():
    #         del st.session_state[key]
    #     st.experimental_rerun()


    user_input = {}
    basic_inputs, advanced_inputs = st.columns(2)

    for feature in selected_columns:
        label = feature_labels.get(feature, friendly_label_from_dummy(feature))

        # 👇 Custom dropdown for MSSubClass
        if feature == "MSSubClass":
            subclass_display = {v: k for k, v in ms_subclass_options.items()}
            selected_desc = st.selectbox(f"{label}", list(subclass_display.keys()))
            user_input[feature] = subclass_display[selected_desc]
            continue

        # 👇 One-hot encoded dummy feature (as checkbox)
        if '_' in feature:
            with advanced_inputs:
                user_input[feature] = st.selectbox(f"{label}", ["No", "Yes"]) == "Yes"

        # 👇 Low cardinality numeric values
        elif final_train[feature].nunique() <= 10 and final_train[feature].dtype in [np.int64, np.int32]:
            min_val = int(final_train[feature].min())
            max_val = int(final_train[feature].max())
            median_val = int(final_train[feature].median())
            with basic_inputs:
                user_input[feature] = st.slider(f"{label}", min_val, max_val, median_val)

        # 👇 General continuous numeric input
        else:
            min_val = float(final_train[feature].min())
            max_val = float(final_train[feature].max())
            default_val = float(final_train[feature].mean())
            with basic_inputs:
                user_input[feature] = st.number_input(f"{label}", min_value=min_val, max_value=max_val, value=default_val)

    # --- Predict and Show Result ---
    if st.button("🚀 Predict"):
        user_df = pd.DataFrame([user_input])
        user_scaled = scaler.transform(user_df)
        user_scaled_df = pd.DataFrame(user_scaled, columns=selected_columns)  # fixes sklearn warning
        user_pred_log = model.predict(user_scaled_df)
        user_pred_price = np.expm1(user_pred_log)[0] * 1e5  # converting lakhs to rupees
        st.success(f"💰 **Predicted House Price:** ₹ {user_pred_price:,.2f}")
