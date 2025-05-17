import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

class CustomerSegmentation:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.numerical_features = ['age', 'quantity', 'price']
        self.categorical_features = ['gender', 'category', 'payment_method', 'shopping_mall']
        
    def preprocess_data(self):
        # Convert date to datetime
        self.data['invoice_date'] = pd.to_datetime(self.data['invoice_date'], format='%d/%m/%Y')
        
        # Add time-based features
        self.data['year'] = self.data['invoice_date'].dt.year
        self.data['month'] = self.data['invoice_date'].dt.month
        self.data['day_of_week'] = self.data['invoice_date'].dt.dayofweek
        
        # Calculate total spending
        self.data['total_spending'] = self.data['quantity'] * self.data['price']
        
        # Create age groups
        self.data['age_group'] = pd.cut(self.data['age'], 
                                      bins=[0, 25, 35, 45, 55, 65, 100],
                                      labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
        
        return self.data
    
    def perform_kmeans_clustering(self, n_clusters=4):
        # Prepare numerical data
        numerical_data = self.data[self.numerical_features].copy()
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numerical_data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.data['kmeans_cluster'] = kmeans.fit_predict(scaled_data)
        
        # Calculate cluster statistics
        cluster_stats = self.data.groupby('kmeans_cluster').agg({
            'age': ['mean', 'std'],
            'quantity': ['mean', 'std'],
            'price': ['mean', 'std'],
            'total_spending': ['mean', 'std']
        }).round(2)
        
        return cluster_stats
    
    def perform_kmodes_clustering(self, n_clusters=4):
        # Prepare categorical data
        categorical_data = self.data[self.categorical_features].copy()
        
        # Perform K-modes clustering
        kmodes = KModes(n_clusters=n_clusters, random_state=42)
        self.data['kmodes_cluster'] = kmodes.fit_predict(categorical_data)
        
        # Calculate cluster characteristics
        cluster_chars = self.data.groupby('kmodes_cluster').agg({
            'gender': lambda x: x.mode()[0],
            'category': lambda x: x.mode()[0],
            'payment_method': lambda x: x.mode()[0],
            'shopping_mall': lambda x: x.mode()[0]
        })
        
        return cluster_chars
    
    def analyze_clusters(self):
        # Combine both clustering results
        self.data['combined_segment'] = self.data['kmeans_cluster'].astype(str) + '_' + self.data['kmodes_cluster'].astype(str)
        
        # Calculate segment statistics
        segment_stats = self.data.groupby('combined_segment').agg({
            'age': 'mean',
            'total_spending': 'mean',
            'quantity': 'mean',
            'price': 'mean'
        }).round(2)
        
        return segment_stats
    
    def plot_cluster_analysis(self):
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Age vs Total Spending by K-means Cluster
        sns.scatterplot(data=self.data, x='age', y='total_spending', 
                       hue='kmeans_cluster', palette='Set1', ax=axes[0,0])
        axes[0,0].set_title('Age vs Total Spending by K-means Cluster')
        
        # Plot 2: Category Distribution by K-modes Cluster
        category_dist = pd.crosstab(self.data['kmodes_cluster'], self.data['category'])
        category_dist.plot(kind='bar', stacked=True, ax=axes[0,1])
        axes[0,1].set_title('Category Distribution by K-modes Cluster')
        
        # Plot 3: Payment Method Distribution
        payment_dist = pd.crosstab(self.data['kmeans_cluster'], self.data['payment_method'])
        payment_dist.plot(kind='bar', stacked=True, ax=axes[1,0])
        axes[1,0].set_title('Payment Method Distribution by K-means Cluster')
        
        # Plot 4: Age Group Distribution
        age_dist = pd.crosstab(self.data['kmodes_cluster'], self.data['age_group'])
        age_dist.plot(kind='bar', stacked=True, ax=axes[1,1])
        axes[1,1].set_title('Age Group Distribution by K-modes Cluster')
        
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    # Initialize the segmentation analysis
    segmentation = CustomerSegmentation('datasets/customer_shopping_data.csv')
    
    # Preprocess the data
    data = segmentation.preprocess_data()
    
    # Perform clustering
    kmeans_stats = segmentation.perform_kmeans_clustering()
    kmodes_stats = segmentation.perform_kmodes_clustering()
    
    # Analyze clusters
    segment_stats = segmentation.analyze_clusters()
    
    # Plot results
    fig = segmentation.plot_cluster_analysis()
    plt.show()
    
    # Print statistics
    print("\nK-means Cluster Statistics:")
    print(kmeans_stats)
    print("\nK-modes Cluster Characteristics:")
    print(kmodes_stats)
    print("\nCombined Segment Statistics:")
    print(segment_stats) 