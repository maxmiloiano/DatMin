import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    st.title("GMM Clustering and Classification")

    # --- Upload Dataset --- #
    st.header("Step 1: Upload Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, delimiter=';')
        data = data.head(500)  # Limit to first 500 rows
        st.write(f"Dataset successfully uploaded with {len(data)} rows and {len(data.columns)} columns.")
        st.write(data.head())

        # --- Data Cleaning --- #
        st.header("Step 2: Data Cleaning")
        irrelevant_cols = ['Customer ID', 'Name', 'Surname', 'Birthdate', 'Merchant Name']
        data_cleaned = data.drop(columns=[col for col in irrelevant_cols if col in data.columns], errors='ignore')

        if 'Date' in data_cleaned.columns:
            data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], errors='coerce').astype(int) / 10**9

        for col in data_cleaned.select_dtypes(include=['object']).columns:
            data_cleaned[col].fillna('Unknown', inplace=True)

        for col in data_cleaned.select_dtypes(include=['float64', 'int64']).columns:
            data_cleaned[col].fillna(data_cleaned[col].median(), inplace=True)

        label_encoder = LabelEncoder()
        for col in data_cleaned.select_dtypes(include=['object']).columns:
            data_cleaned[col] = label_encoder.fit_transform(data_cleaned[col])

        st.write("Cleaned Dataset:")
        st.write(data_cleaned.head())

        # --- Exploratory Data Analysis (EDA) --- #
        st.header("Step 3: Exploratory Data Analysis (EDA)")
        st.write("Dataset Description:")
        st.write(data_cleaned.describe())

        if 'Transaction Amount' in data_cleaned.columns:
            st.write("Distribution of Transaction Amount:")
            fig, ax = plt.subplots()
            data_cleaned['Transaction Amount'].hist(bins=30, alpha=0.7, color='blue', ax=ax)
            ax.set_title('Transaction Amount Distribution')
            ax.set_xlabel('Transaction Amount')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

        # --- Scaling Data --- #
        st.header("Step 4: Data Scaling")
        scaler = StandardScaler()
        data_numeric = data_cleaned.select_dtypes(include=['float64', 'int64'])
        data_scaled = scaler.fit_transform(data_numeric)

        # --- Gaussian Mixture Model (GMM) Clustering --- #
        st.header("Step 5: Gaussian Mixture Model (GMM) Clustering")
        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3, step=1)

        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        clusters = gmm.fit_predict(data_scaled)
        data_cleaned['Cluster'] = clusters

        st.write(f"Clustering completed with {n_clusters} clusters.")
        st.write(data_cleaned['Cluster'].value_counts())

        # Silhouette Score
        if len(set(clusters)) > 1:
            silhouette_avg = silhouette_score(data_scaled, clusters)
            st.write(f"Silhouette Score: {silhouette_avg:.2f}")

        # Visualize Clusters
        st.write("Cluster Visualization with PCA:")
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_scaled)
        data_cleaned['PCA1'] = data_pca[:, 0]
        data_cleaned['PCA2'] = data_pca[:, 1]

        fig, ax = plt.subplots()
        sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data_cleaned, palette='Set2', ax=ax)
        ax.set_title('Clusters Visualized in PCA Space')
        st.pyplot(fig)

        # --- Classification --- #
        st.header("Step 6: Classification")
        X = data_cleaned.drop(columns=['Cluster', 'PCA1', 'PCA2'], errors='ignore')
        y = data_cleaned['Cluster']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Classification Accuracy: {accuracy:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)

        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred, target_names=[f'Cluster {i}' for i in range(n_clusters)]))

        # Download Results
        st.header("Download Results")
        data_cleaned['Predicted Cluster'] = clf.predict(X)
        csv = data_cleaned.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='gmm_clustering_results_full.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
