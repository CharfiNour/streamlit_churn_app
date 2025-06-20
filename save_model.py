import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("Expresso_churn_dataset.csv")

# Prepare features and target
X = df[['REGULARITY', 'FREQ_TOP_PACK', 'ORANGE', 'TIGO']]
y = df['CHURN']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

# Train the model
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

model = XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.01,
    random_state=42,
    subsample=0.6,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight 
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)