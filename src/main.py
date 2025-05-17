"""
Main analysis script for Istanbul Shopping and Tourism Analysis.
This script orchestrates the complete analysis pipeline including:
- Data loading and preprocessing
- Customer segmentation
- Purchase prediction
- Visualization generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Import custom modules
from analysis.customer_segmentation import CustomerSegmentation
from models.purchase_prediction import PurchasePrediction
from visualization.plot_generator import (
    plot_demographic_distribution,
    plot_spending_patterns,
    plot_correlation_analysis
)

def load_data(file_path):
    """
    Load and perform initial data inspection.
    
    Args:
        file_path (str): Path to the dataset
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    print("\nDataset Shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    return df

def preprocess_data(df):
    """
    Preprocess the dataset for analysis.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Processed dataset
    """
    print("\nPreprocessing data...")
    
    # Convert date columns to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Extract temporal features
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.day
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    
    # Create total amount feature
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    print("Preprocessing complete!")
    return df

def perform_customer_segmentation(df):
    """
    Perform customer segmentation analysis.
    
    Args:
        df (pd.DataFrame): Processed dataset
        
    Returns:
        pd.DataFrame: Customer features with cluster assignments
    """
    print("\nPerforming customer segmentation...")
    
    segmenter = CustomerSegmentation(n_clusters=4)
    customer_features = segmenter.fit(df)
    cluster_analysis = segmenter.analyze_clusters(customer_features)
    
    print("\nCustomer Segment Analysis:")
    print(cluster_analysis)
    
    return customer_features

def train_purchase_prediction(df):
    """
    Train purchase prediction model.
    
    Args:
        df (pd.DataFrame): Processed dataset
        
    Returns:
        dict: Model performance metrics
    """
    print("\nTraining purchase prediction model...")
    
    predictor = PurchasePrediction()
    metrics = predictor.fit(df)
    
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def generate_visualizations(df):
    """
    Generate all analysis visualizations.
    
    Args:
        df (pd.DataFrame): Processed dataset
    """
    print("\nGenerating visualizations...")
    
    # Set style for better visualizations
    plt.style.use('seaborn')
    sns.set_palette('husl')
    
    print("\nDemographic Distribution:")
    plot_demographic_distribution(df)
    
    print("\nSpending Patterns:")
    plot_spending_patterns(df)
    
    print("\nCorrelation Analysis:")
    plot_correlation_analysis(df)

def main():
    """
    Main execution function.
    """
    # Load and preprocess data
    df = load_data('data/istanbul_shopping_data.csv')
    df_processed = preprocess_data(df)
    
    # Perform customer segmentation
    customer_features = perform_customer_segmentation(df_processed)
    
    # Train purchase prediction model
    metrics = train_purchase_prediction(df_processed)
    
    # Generate visualizations
    generate_visualizations(df_processed)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 