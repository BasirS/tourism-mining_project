import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class PlotGenerator:
    def __init__(self, data):
        self.data = data
        
    def plot_demographic_distribution(self):
        """Plot age and gender distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Age distribution
        sns.histplot(data=self.data, x='age', bins=30, ax=ax1)
        ax1.set_title('Age Distribution')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Count')
        
        # Gender distribution
        gender_counts = self.data['gender'].value_counts()
        gender_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
        ax2.set_title('Gender Distribution')
        
        plt.tight_layout()
        return fig
    
    def plot_spending_patterns(self):
        """Plot spending patterns by category and payment method"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Average spending by category
        category_spending = self.data.groupby('category')['total_spending'].mean().sort_values(ascending=False)
        category_spending.plot(kind='bar', ax=ax1)
        ax1.set_title('Average Spending by Category')
        ax1.set_xlabel('Category')
        ax1.set_ylabel('Average Spending')
        plt.xticks(rotation=45)
        
        # Payment method distribution
        payment_counts = self.data['payment_method'].value_counts()
        payment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
        ax2.set_title('Payment Method Distribution')
        
        plt.tight_layout()
        return fig
    
    def plot_temporal_trends(self):
        """Plot temporal trends in spending"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Monthly spending trends
        monthly_spending = self.data.groupby(['year', 'month'])['total_spending'].mean().unstack()
        monthly_spending.plot(kind='line', marker='o', ax=ax1)
        ax1.set_title('Monthly Spending Trends')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Average Spending')
        ax1.legend(title='Year')
        
        # Day of week spending patterns
        dow_spending = self.data.groupby('day_of_week')['total_spending'].mean()
        dow_spending.plot(kind='bar', ax=ax2)
        ax2.set_title('Average Spending by Day of Week')
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Average Spending')
        ax2.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        plt.tight_layout()
        return fig
    
    def plot_mall_analysis(self):
        """Plot mall-specific analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Average spending by mall
        mall_spending = self.data.groupby('shopping_mall')['total_spending'].mean().sort_values(ascending=False)
        mall_spending.plot(kind='bar', ax=ax1)
        ax1.set_title('Average Spending by Mall')
        ax1.set_xlabel('Mall')
        ax1.set_ylabel('Average Spending')
        plt.xticks(rotation=45)
        
        # Category distribution by mall
        mall_category = pd.crosstab(self.data['shopping_mall'], self.data['category'])
        mall_category.plot(kind='bar', stacked=True, ax=ax2)
        ax2.set_title('Category Distribution by Mall')
        ax2.set_xlabel('Mall')
        ax2.set_ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_analysis(self):
        """Plot correlation analysis between numerical features"""
        numerical_data = self.data[['age', 'quantity', 'price', 'total_spending']]
        correlation_matrix = numerical_data.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        return plt.gcf()
    
    def generate_all_plots(self):
        """Generate all plots and save them"""
        plots = {
            'demographic_distribution': self.plot_demographic_distribution(),
            'spending_patterns': self.plot_spending_patterns(),
            'temporal_trends': self.plot_temporal_trends(),
            'mall_analysis': self.plot_mall_analysis(),
            'correlation_analysis': self.plot_correlation_analysis()
        }
        
        # Save plots
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for name, fig in plots.items():
            fig.savefig(f'plots/{name}_{timestamp}.png')
            plt.close(fig)
        
        return plots

if __name__ == "__main__":
    # Load data
    data = pd.read_csv('datasets/customer_shopping_data.csv')
    
    # Create plot generator
    plotter = PlotGenerator(data)
    
    # Generate all plots
    plots = plotter.generate_all_plots()
    
    # Display plots
    for name, fig in plots.items():
        plt.figure(fig.number)
        plt.show() 