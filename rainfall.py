import pandas as pd

data = pd.read_csv("weather.csv")

print("Shape:", data.shape)
print(data.head())
print(data.columns)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("weather.csv")

# Normalize column names
data.columns = data.columns.str.strip().str.lower()

# Check target column
if 'raintomorrow' not in data.columns:
    raise ValueError("Column 'RainTomorrow' not found in dataset")

# Fill missing values instead of dropping all rows
# Fill missing values safely
for col in data.columns:
    if data[col].dtype == 'object':
        if data[col].mode().empty:
            data[col] = data[col].fillna("Unknown")
        else:
            data[col] = data[col].fillna(data[col].mode().iloc[0])
    else:
        data[col] = data[col].fillna(data[col].mean())


# Encode categorical columns
le = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = le.fit_transform(data[col])

# Features & target
X = data.drop('raintomorrow', axis=1)
y = data['raintomorrow']

# Safety check
if len(X) == 0:
    raise ValueError("Dataset is empty after preprocessing")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Output
print("🌧️ Rainfall Prediction Model")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Dataset shape:", data.shape)
print(data.isna().sum())

