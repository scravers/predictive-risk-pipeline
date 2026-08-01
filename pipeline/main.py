import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import resample
from ucimlrepo import fetch_ucirepo 
  
TARGET_COL = 'Bankrupt?'
# Set a strict custom thresholds
custom_threshold = 0.50
# custom_threshold = 0.90


def load_data(target_col):     
    dataset = fetch_ucirepo(id=572) 

    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    return df.drop(target_col, axis=1), df[target_col]




def downsample_training(X_train, y_train):
   # 3. DOWNSAMPLE ONLY THE TRAINING DATA
   print("\nDownsampling the training set...")
   # Combine X_train and y_train temporarily for resampling
   train_data = pd.concat([X_train, y_train], axis=1)


   # Separate the minority and majority classes in the training data
   train_min = train_data[train_data[TARGET_COL] == 1]
   train_maj = train_data[train_data[TARGET_COL] == 0]


   # Downsample the majority class to match the minority class (1:1 ratio)
   train_maj_downsampled = resample(train_maj, replace=False, n_samples=len(train_min), random_state=21)


   # Recombine into a balanced training set
   train_balanced = pd.concat([train_maj_downsampled, train_min])


   X_train_balanced = train_balanced.drop(TARGET_COL, axis=1)
   y_train_balanced = train_balanced[TARGET_COL]


   print(f"Original Training Set: {len(X_train)} rows.")
   print(f"Balanced Training Set: {len(X_train_balanced)} rows.\n")


   return X_train_balanced, y_train_balanced




X, y = load_data(TARGET_COL)


print("--- Input Data (First 5 Rows) ---")
print(X.head())


X_train_unbalance, X_test, y_train_unbalance, y_test = train_test_split(X, y, test_size=0.2, random_state=21)


X_train, y_train = downsample_training(X_train_unbalance, y_train_unbalance)


# Scale the features (Fit ONLY on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# STEP 1: Feature Selection via L1 (Lasso)
# ==========================================
print("Running Step 1: L1 Feature Selection...")


# C controls the penalty strength. Smaller C = stronger penalty (more features dropped)
lasso_selector = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=21, class_weight='balanced')
lasso_selector.fit(X_train_scaled, y_train)


# Filter out all the features that the L1 penalty pushed to 0
selector = SelectFromModel(lasso_selector, prefit=True)
X_train_reduced = selector.transform(X_train_scaled)
X_test_reduced = selector.transform(X_test_scaled)


print(f"Original number of features: {X_train_scaled.shape[1]}")
print(f"Reduced number of features: {X_train_reduced.shape[1]}")
print(f"Features dropped: {X_train_scaled.shape[1] - X_train_reduced.shape[1]}\n")


# Displaying Intermediate Outputs (After Scaling and Feature Selection)
print("--- Intermediate Output: Scaled & Reduced X_train (First 5 Rows) ---")
# Converting back to DataFrame just for a clean visual printout
intermediate_df = pd.DataFrame(X_train_reduced)
print(intermediate_df.head())


# ==========================================
# STEP 2: SVM with Grid Search
# ==========================================
print("Running Step 2: Tuning Hyperparameters for SVM...")


param_grid = {
   'C': [0.01, 0.1, 1, 10],
   'gamma': ['scale', 'auto', 0.1, 1],
   'kernel': ['rbf']
}


# Set up the Grid Search using the REDUCED datasets
grid = GridSearchCV(SVC(probability=True),
                   param_grid,
                   refit=True,
                   verbose=1,
                   cv=5)


grid.fit(X_train_reduced, y_train)


print(f"Best Parameters Found: {grid.best_params_}")


# Use the best model
model = grid.best_estimator_


# y_pred = model.predict(X_test_reduced)


# Get the probabilities for class 1
y_prob = model.predict_proba(X_test_reduced)[:, 1]






# Apply the new threshold: True becomes 1, False becomes 0
y_pred = (y_prob >= custom_threshold).astype(int)


# Displaying Final Results
print("--- Final Results (First 5 Rows) ---")


# Create a DataFrame combining Actuals, Probabilities, and the Custom Prediction
results_df = pd.DataFrame({
   'Actual class': y_test,
   'Probability of being class 1': y_prob,
   'Final prediction (Threshold=' + str(custom_threshold) + ')': y_pred  # Or whatever threshold you used
})


print(results_df.head())


print(classification_report(y_test, y_pred))


# Plot A: Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('images/confusion_matrix_.png')
print("Saved confusion matrix image.")


# Plot B: ROC Curve
y_prob = model.predict_proba(X_test_reduced)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)


plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.legend()
plt.savefig('images/roc_curve.png')
print("Saved ROC curve image.")
