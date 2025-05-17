import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class PurchasePrediction:
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
        
        # Create spending categories
        self.data['spending_category'] = pd.qcut(self.data['total_spending'], 
                                               q=4, 
                                               labels=['Low', 'Medium', 'High', 'Very High'])
        
        return self.data
    
    def prepare_features(self):
        # Create feature matrix
        X = self.data[self.numerical_features + self.categorical_features]
        y = self.data['spending_category']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self):
        # Create preprocessing steps
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        # Create model pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        return model
    
    def train_and_evaluate(self):
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_features()
        
        # Build and train model
        model = self.build_model()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate classification report
        report = classification_report(y_test, y_pred)
        
        return accuracy, report, model
    
    def analyze_feature_importance(self, model):
        # Get feature names
        feature_names = self.numerical_features + list(
            model.named_steps['preprocessor']
            .named_transformers_['cat']
            .named_steps['onehot']
            .get_feature_names_out(self.categorical_features)
        )
        
        # Get feature importance
        importance = model.named_steps['classifier'].feature_importances_
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance
    
    def plot_feature_importance(self, feature_importance):
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
        plt.title('Top 10 Most Important Features')
        plt.tight_layout()
        return plt.gcf()

if __name__ == "__main__":
    # Initialize the prediction model
    predictor = PurchasePrediction('datasets/customer_shopping_data.csv')
    
    # Preprocess the data
    data = predictor.preprocess_data()
    
    # Train and evaluate the model
    accuracy, report, model = predictor.train_and_evaluate()
    
    # Analyze feature importance
    feature_importance = predictor.analyze_feature_importance(model)
    
    # Plot feature importance
    fig = predictor.plot_feature_importance(feature_importance)
    plt.show()
    
    # Print results
    print(f"\nModel Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(report)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10)) 