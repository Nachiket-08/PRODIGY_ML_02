

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import joblib
import random

RSEED = 24
np.random.seed(RSEED)
random.seed(RSEED)

DATA_FILE = (r"C:\Users\Hp\Downloads\Mall_Customers.csv")

def load_data(path):
    assert os.path.exists(path), f"Put dataset at: {path}"
    df = pd.read_csv(path)
    # classic choice: Annual Income and Spending Score
    if "Annual Income (k$)" in df.columns and "Spending Score (1-100)" in df.columns:
        X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
    else:
        # fallback: try columns by index (common variants)
        X = df.iloc[:, 3:5].values
    return df, X

def elbow_and_silhouette(X_scaled, max_k=10):
    wcss = []
    sil = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=RSEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        wcss.append(km.inertia_)
        sil.append(silhouette_score(X_scaled, labels))
    return wcss, sil

def main():
    df, X = load_data(DATA_FILE)
    print("Rows loaded:", len(df))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    wcss, sil = elbow_and_silhouette(X_scaled, max_k=10)

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(range(2, 11), wcss, marker='o')
    plt.title("Elbow plot")
    plt.xlabel("k")
    plt.ylabel("WCSS")

    plt.subplot(1,2,2)
    plt.plot(range(2, 11), sil, marker='o', color='orange')
    plt.title("Silhouette scores")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.savefig("task2_elbow_silhouette.png", dpi=150)
    plt.show()

    # From elbow + silhouette decide on k (common example is 5)
    chosen_k = 5
    print("Using k =", chosen_k)

    final_km = KMeans(n_clusters=chosen_k, random_state=RSEED, n_init=20)
    cluster_labels = final_km.fit_predict(X_scaled)

    df["Cluster"] = cluster_labels

    # save cluster centers in original feature space
    centers_scaled = final_km.cluster_centers_
    centers_orig = scaler.inverse_transform(centers_scaled)

    centers_df = pd.DataFrame(centers_orig, columns=["AnnualIncome", "SpendingScore"])
    print("\nCluster centers (approx):")
    print(centers_df)

    df.to_csv("task2_customers_with_clusters.csv", index=False)
    joblib.dump(final_km, "task2_kmeans_model.joblib")
    joblib.dump(scaler, "task2_scaler.joblib")
    print("\nSaved clustered CSV and model/scaler")

    # Plot clusters nicely
    plt.figure(figsize=(8,6))
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
    for c in range(chosen_k):
        mask = df["Cluster"] == c
        plt.scatter(df.loc[mask, "Annual Income (k$)"] if "Annual Income (k$)" in df.columns else df.iloc[mask,3],
                    df.loc[mask, "Spending Score (1-100)"] if "Spending Score (1-100)" in df.columns else df.iloc[mask,4],
                    s=60, alpha=0.7, label=f"Cluster {c}", color=colors[c % len(colors)])
    plt.scatter(centers_orig[:,0], centers_orig[:,1], s=250, marker='X', c='black', label='Centroids')
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.title("Customer segments (K-Means)")
    plt.tight_layout()
    plt.savefig("task2_clusters.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
