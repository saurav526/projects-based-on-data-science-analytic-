import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1️⃣ Load Dataset
data = pd.read_csv("customers.csv")
print("Dataset Loaded Successfully")
print(data.head())

# 2️⃣ Select Features
X = data[['Age', 'AnnualIncome', 'SpendingScore']]

# 3️⃣ Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# 5️⃣ Show Clustered Data
print("\nClustered Customers:")
print(data)

# 6️⃣ Visualization
plt.scatter(
    data['AnnualIncome'],
    data['SpendingScore'],
    c=data['Cluster']
)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")
plt.show()
