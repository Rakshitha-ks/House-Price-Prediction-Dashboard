# **House Price Prediction using Multiple Linear Regression**

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


# Loading the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Preview data
print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()

train.info()

test_ids = test['Id']  # 👈 save it before dropping


# Check for missing values (top 20)
missing_values = train.isnull().sum().sort_values(ascending=False).head(20)
missing_values

columns_to_remove = missing_values[missing_values > 80].index
columns_to_remove

train.drop(columns=columns_to_remove, inplace=True)
test.drop(columns=columns_to_remove, inplace=True)

train.head()

train.columns

irrelevant_cols = ['Id', 'Utilities', 'Condition2', 'BsmtFinType2', 'RoofMatl', 'PavedDrive', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch']
train.drop(columns=irrelevant_cols, inplace=True)
test.drop(columns=irrelevant_cols, inplace=True)

train.info()

# Select only numerical features
num_cols = train.select_dtypes(include=['number'])

# Compute correlation matrix with SalePrice
correlations = num_cols.corr()['SalePrice'].sort_values()

# Show features with low correlation (e.g., abs(corr) < 0.1)
low_corr = correlations[abs(correlations) < 0.1]
print("🔻 Features weakly correlated with SalePrice:\n")
print(low_corr)

# Plot correlation heatmap of top 15 features
plt.figure(figsize=(12,8))
top_corr = num_cols.corr()['SalePrice'].abs().sort_values(ascending=False).head(15).index
sns.heatmap(num_cols[top_corr].corr(), annot=True, cmap='coolwarm')
plt.title(" Top 15 Correlated Numerical Features with SalePrice")




# Save the plot
plt.savefig("correlation_heatmap.png", bbox_inches='tight')
st.pyplot(plt.gcf())
plt.show()
plt.close()

# Identify features with very low correlation (abs < 0.03)
low_corr_features = correlations[abs(correlations) < 0.03].index.tolist()
print("Weak correlation features being dropped:", low_corr_features)

# Drop weak correlation features
train.drop(columns=low_corr_features, inplace=True, errors='ignore')
test.drop(columns=low_corr_features, inplace=True, errors='ignore')

# Final shape after drops
print("Final shape of train data:", train.shape)

# Combine for consistent processing
train['is_train'] = 1
test['is_train'] = 0
test['SalePrice'] = np.nan  # add dummy target
combined = pd.concat([train, test], axis=0)

# Separate column types
num_cols = combined.select_dtypes(include=['number']).columns.tolist()
cat_cols = combined.select_dtypes(include=['object']).columns.tolist()

# ✅ Fill missing values
# Fill numeric columns with 0 or median
zero_fill = ['GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFullBath']
median_fill = [ 'MasVnrArea', 'TotalBsmtSF']

for col in zero_fill:
    combined[col] = combined[col].fillna(0)

for col in median_fill:
    combined[col] = combined[col].fillna(combined[col].median())

combined[num_cols] = combined[num_cols].fillna(0)

# Fill categorical columns with "None"
combined[cat_cols] = combined[cat_cols].fillna("None")

# Encode categorical variables using one-hot encoding
combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

# Split back into train and test
train_filled = combined_encoded[combined_encoded['is_train'] == 1].drop(columns='is_train')
test_filled = combined_encoded[combined_encoded['is_train'] == 0].drop(columns=['is_train', 'SalePrice'])

# Confirm no missing values remain
print("✅ Train nulls:", train_filled.isnull().sum().sum())
print("✅ Test nulls:", test_filled.isnull().sum().sum())

train.shape

train_filled.shape



# 1. Keep only numeric features (excluding the target)
X = train_filled.select_dtypes(include='number').drop('SalePrice', axis=1)

# 2. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Compute VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
vif_data.sort_values(by="VIF", ascending=False, inplace=True)

print("📌 VIF values before dropping:\n", vif_data)

# 4. Drop features with VIF > 10 (high multicollinearity)
high_vif = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()
print("\n🧹 Dropping high VIF features:", high_vif)

X_reduced = X.drop(columns=high_vif)

# 5. Optional: Recalculate VIF after dropping
X_scaled_reduced = scaler.fit_transform(X_reduced)
vif_data_reduced = pd.DataFrame()
vif_data_reduced["Feature"] = X_reduced.columns
vif_data_reduced["VIF"] = [variance_inflation_factor(X_scaled_reduced, i) for i in range(X_scaled_reduced.shape[1])]
vif_data_reduced.sort_values(by="VIF", ascending=False, inplace=True)

print("\n✅ VIF after dropping:\n", vif_data_reduced)

# 6. Reconstruct cleaned dataset with reduced features and SalePrice
final_train = pd.concat([X_reduced, train['SalePrice']], axis=1)
print("\n✅ Final shape of train data:", final_train.shape)

# Drop the same high-VIF columns from test data
test_reduced = test_filled.drop(columns=high_vif, errors='ignore')


# Histogram of original SalePrice
plt.figure(figsize=(8, 5))
sns.histplot(train_filled['SalePrice'], kde=True)
plt.title("Original SalePrice Distribution")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.show()

# Calculate skewness
original_skew = skew(train_filled['SalePrice'])
print(f"📈 Skewness of SalePrice: {original_skew:.2f}")

# If skew > 0.75, log-transform is generally helpful
print("➡️ The target is skewed. Log transformation is recommended.")

train_filled['SalePrice'] = np.log1p(train_filled['SalePrice'])

# Log-transformed SalePrice plot
plt.figure(figsize=(8, 5))
sns.histplot(train_filled['SalePrice'], kde=True)
plt.title("Log-Transformed SalePrice Distribution")
plt.xlabel("log(SalePrice + 1)")
plt.ylabel("Frequency")
plt.show()


final_train = pd.concat([X_reduced, train_filled['SalePrice']], axis=1)



#  Train-test split

X = final_train.drop('SalePrice', axis=1)
y = final_train['SalePrice']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Training Linear Regression model

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model

# WITHOUT CLIPPING
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_log = model.predict(X_val)

# Check performance in log scale first
rmse_log = np.sqrt(mean_squared_error(y_val, y_pred_log))
r2_log = r2_score(y_val, y_pred_log)
print(f"RMSE (log): {rmse_log:.4f}, R² (log): {r2_log:.4f}")

# Inverse transform
y_pred_original = np.expm1(y_pred_log)
y_val_original = np.expm1(y_val)

# Actual performance
rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_original))
mae = mean_absolute_error(y_val_original, y_pred_original)
r2 = r2_score(y_val_original, y_pred_original)



print(f"\n📊 Model Performance:")
print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")


# 📈 Plot: Actual vs Predicted Prices
plt.figure(figsize=(6,6))
plt.scatter(y_val_original, y_pred_original, alpha=0.5, color='dodgerblue')
plt.plot([min(y_val_original), max(y_val_original)],
         [min(y_val_original), max(y_val_original)], color='red', linestyle='--')
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted Prices")
plt.grid(True)


# Save the plot
plt.savefig("actual_vs_predicted.png", bbox_inches='tight')
st.pyplot(plt.gcf())
plt.show()
plt.close()

# Helper function to evaluate modelsss
def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # Clip and reverse log
    # y_pred = np.clip(y_pred, a_min=None, a_max=10)
    y_pred_original = np.expm1(y_pred)
    y_val_original = np.expm1(y_val)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_original))
    mae = mean_absolute_error(y_val_original, y_pred_original)
    r2 = r2_score(y_val_original, y_pred_original)

    print(f"\n📊 {name} Performance:")
    print(f"✅ RMSE: {rmse:.2f}")
    print(f"✅ MAE: {mae:.2f}")
    print(f"✅ R² Score: {r2:.4f}")

# Linear Regression
evaluate_model(LinearRegression(), "Linear Regression")

# Ridge Regression (L2 regularization)
evaluate_model(Ridge(alpha=10), "Ridge Regression (α=10)")

# Lasso Regression (L1 regularization)
evaluate_model(Lasso(alpha=0.001), "Lasso Regression (α=0.001)")

# Predict on real test set
test_scaled = scaler.transform(test_reduced[X_reduced.columns])
test_preds_log = model.predict(test_scaled)
test_preds = np.expm1(test_preds_log) * 1e5  # Converts log-lakhs to rupees

# Save submission
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_preds
})
submission.to_csv("final_submission.csv", index=False)
print("✅ Submission file 'final_submission.csv' created!")


print(test_preds[:10]) # sample output


import pickle

# Save cleaned train and test data
with open('train_filled.pkl', 'wb') as f:
    pickle.dump(train_filled, f)

with open('test_filled.pkl', 'wb') as f:
    pickle.dump(test_filled, f)

# Save reduced features (after VIF)
with open('X_reduced_columns.pkl', 'wb') as f:
    pickle.dump(X_reduced.columns.tolist(), f)

# Save final train and test datasets used in modeling
with open('final_train.pkl', 'wb') as f:
    pickle.dump(final_train, f)

with open('test_reduced.pkl', 'wb') as f:
    pickle.dump(test_reduced, f)

# Save the StandardScaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save trained model
with open('final_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save VIF table
with open('vif_table.pkl', 'wb') as f:
    pickle.dump(vif_data, f)

# Save model results
model_results = {
    "Linear Regression": {
        "RMSE": rmse,
        "MAE": mae,
        "R² Score": r2
    }
}
with open('model_results.pkl', 'wb') as f:
    pickle.dump(model_results, f)
