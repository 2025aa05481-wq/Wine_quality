import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="ML Assignment 2 - Wine Quality Classification",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .best-model {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models and results
@st.cache_resource
def load_models_and_results():
    """Load all trained models and results"""
    models = {}
    results = {}
    scaler = None
    
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'K-Nearest Neighbor': 'k-nearest_neighbor.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    try:
        # Load models
        for name, filename in model_files.items():
            if os.path.exists(f'model/{filename}'):
                with open(f'model/{filename}', 'rb') as f:
                    models[name] = pickle.load(f)
        
        # Load results
        if os.path.exists('model/model_results.pkl'):
            with open('model/model_results.pkl', 'rb') as f:
                results = pickle.load(f)
        
        # Load scaler
        if os.path.exists('model/scaler.pkl'):
            with open('model/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
                
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}, {}, None
    
    return models, results, scaler

# Load dataset
@st.cache_data
def load_dataset():
    """Load the wine quality dataset"""
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df = pd.read_csv(url, sep=';')
        # Create binary classification
        df['quality_binary'] = (df['quality'] > 6).astype(int)
        df['quality_label'] = df['quality_binary'].map({0: 'Bad', 1: 'Good'})
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Main application
def main():
    # Load models and data
    models, results, scaler = load_models_and_results()
    df = load_dataset()
    
    # Sidebar navigation
    st.sidebar.title("🍷 Wine Quality Classification")
    page = st.sidebar.selectbox("Navigate", [
        "Home", 
        "Dataset Overview", 
        "Model Comparison", 
        "Model Prediction", 
        "Upload Dataset"
    ])
    
    if page == "Home":
        st.markdown('<h1 class="main-header">Machine Learning Assignment 2</h1>', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">Wine Quality Classification System</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 📊 Project Overview
            This project implements a comprehensive machine learning classification system using **6 different algorithms** 
            to predict wine quality based on physicochemical properties.
            
            ### 🎯 Objective
            - **Task**: Binary Classification (Good vs Bad Wine)
            - **Dataset**: Wine Quality Dataset (Red Wine)
            - **Features**: 11 physicochemical properties
            - **Samples**: 1,599 instances
            
            ### 🤖 ML Models Implemented
            1. Logistic Regression
            2. Decision Tree Classifier
            3. K-Nearest Neighbor
            4. Naive Bayes
            5. Random Forest
            6. XGBoost
            """)
        
        with col2:
            st.markdown("""
            ### 📈 Evaluation Metrics
            - Accuracy
            - AUC Score
            - Precision
            - Recall
            - F1 Score
            - Matthews Correlation Coefficient
            
            ### 🚀 Features
            - Interactive model comparison
            - Real-time predictions
            - Custom dataset upload
            - Performance visualization
            """)
        
        if results:
            best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
            st.markdown(f"""
            <div class="best-model">
                <h3>🏆 Best Performing Model</h3>
                <p><strong>{best_model}</strong></p>
                <p>Accuracy: {results[best_model]['accuracy']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif page == "Dataset Overview":
        st.markdown('<h1 class="main-header">Dataset Overview</h1>', unsafe_allow_html=True)
        
        if df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Dataset Statistics")
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Features:** {len(df.columns) - 2}")  # Excluding quality and binary
                st.write(f"**Samples:** {df.shape[0]}")
                
                # Quality distribution
                st.subheader("🍷 Quality Distribution")
                quality_counts = df['quality_label'].value_counts()
                fig = px.pie(values=quality_counts.values, names=quality_counts.index, 
                           title="Wine Quality Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📋 Feature Information")
                feature_info = pd.DataFrame({
                    'Feature': df.columns[:11],
                    'Type': ['Numeric'] * 11,
                    'Description': [
                        'Fixed acidity', 'Volatile acidity', 'Citric acid',
                        'Residual sugar', 'Chlorides', 'Free sulfur dioxide',
                        'Total sulfur dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol'
                    ]
                })
                st.dataframe(feature_info, use_container_width=True)
                
                # Correlation heatmap
                st.subheader("🔗 Feature Correlations")
                corr_matrix = df.iloc[:, :11].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                              title="Feature Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
            
            # Data preview
            st.subheader("👁️ Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
    
    elif page == "Model Comparison":
        st.markdown('<h1 class="main-header">Model Performance Comparison</h1>', unsafe_allow_html=True)
        
        if not results:
            st.error("No results found. Please train the models first.")
            return
        
        # Create comparison table
        st.subheader("📊 Performance Metrics Table")
        
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.round(4)
        comparison_df.columns = [col.capitalize() for col in comparison_df.columns]
        
        # Add ranking
        comparison_df['Rank'] = comparison_df['Accuracy'].rank(ascending=False).astype(int)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Accuracy Comparison")
            fig = px.bar(x=comparison_df.index, y=comparison_df['Accuracy'],
                        title="Model Accuracy Comparison",
                        labels={'x': 'Model', 'y': 'Accuracy'})
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 AUC Score Comparison")
            fig = px.bar(x=comparison_df.index, y=comparison_df['Auc'],
                        title="Model AUC Score Comparison",
                        labels={'x': 'Model', 'y': 'AUC Score'})
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics comparison
        st.subheader("📊 All Metrics Comparison")
        metrics = ['Accuracy', 'Auc', 'Precision', 'Recall', 'F1', 'Mcc']
        
        fig = make_subplots(rows=2, cols=3, subplot_titles=metrics)
        
        for i, metric in enumerate(metrics):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            fig.add_trace(
                go.Bar(x=comparison_df.index, y=comparison_df[metric], name=metric),
                row=row, col=col
            )
        
        fig.update_layout(height=600, showlegend=False, title_text="Comprehensive Metrics Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model analysis
        best_model = comparison_df.index[0]
        st.markdown(f"""
        <div class="best-model">
            <h3>🏆 Detailed Analysis - {best_model}</h3>
            <p>This model achieved the best performance with the following metrics:</p>
            <ul>
                <li><strong>Accuracy:</strong> {comparison_df.loc[best_model, 'Accuracy']:.4f}</li>
                <li><strong>AUC:</strong> {comparison_df.loc[best_model, 'Auc']:.4f}</li>
                <li><strong>Precision:</strong> {comparison_df.loc[best_model, 'Precision']:.4f}</li>
                <li><strong>Recall:</strong> {comparison_df.loc[best_model, 'Recall']:.4f}</li>
                <li><strong>F1 Score:</strong> {comparison_df.loc[best_model, 'F1']:.4f}</li>
                <li><strong>MCC:</strong> {comparison_df.loc[best_model, 'Mcc']:.4f}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif page == "Model Prediction":
        st.markdown('<h1 class="main-header">Model Prediction</h1>', unsafe_allow_html=True)
        
        if not models or scaler is None:
            st.error("Models not loaded. Please train models first.")
            return
        
        # Model selection
        selected_model = st.selectbox("Select Model", list(models.keys()))
        
        # Input features
        st.subheader("🔢 Input Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fixed_acidity = st.number_input("Fixed Acidity", min_value=4.0, max_value=16.0, value=8.0, step=0.1)
            volatile_acidity = st.number_input("Volatile Acidity", min_value=0.1, max_value=2.0, value=0.5, step=0.01)
            citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
            residual_sugar = st.number_input("Residual Sugar", min_value=0.5, max_value=15.0, value=2.5, step=0.1)
        
        with col2:
            chlorides = st.number_input("Chlorides", min_value=0.01, max_value=0.6, value=0.08, step=0.01)
            free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=1.0, max_value=72.0, value=15.0, step=1.0)
            total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=6.0, max_value=289.0, value=50.0, step=1.0)
            density = st.number_input("Density", min_value=0.99, max_value=1.01, value=0.997, step=0.001)
        
        with col3:
            ph = st.number_input("pH", min_value=2.7, max_value=4.0, value=3.3, step=0.01)
            sulphates = st.number_input("Sulphates", min_value=0.3, max_value=2.0, value=0.6, step=0.01)
            alcohol = st.number_input("Alcohol", min_value=8.0, max_value=15.0, value=10.0, step=0.1)
        
        # Prediction button
        if st.button("🔮 Predict Wine Quality", type="primary"):
            # Prepare input data
            input_data = np.array([[
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                ph, sulphates, alcohol
            ]])
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            model = models[selected_model]
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None
            
            # Display results
            st.subheader("🎯 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                quality_label = "Good Wine 🍷" if prediction == 1 else "Bad Wine 🍾"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Predicted Quality</h3>
                    <h2>{quality_label}</h2>
                    <p>Model: {selected_model}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if prediction_proba is not None:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Prediction Probability</h3>
                        <p><strong>Good Wine:</strong> {prediction_proba[1]:.4f}</p>
                        <p><strong>Bad Wine:</strong> {prediction_proba[0]:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Probability chart
            if prediction_proba is not None:
                fig = px.bar(x=['Bad Wine', 'Good Wine'], y=prediction_proba,
                           title="Prediction Probability Distribution",
                           labels={'x': 'Wine Quality', 'y': 'Probability'})
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Upload Dataset":
        st.markdown('<h1 class="main-header">Upload Custom Dataset</h1>', unsafe_allow_html=True)
        
        st.subheader("📁 Upload Your Dataset")
        st.write("Upload a CSV file with the same structure as the wine quality dataset.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                custom_df = pd.read_csv(uploaded_file)
                
                st.success("Dataset uploaded successfully!")
                
                # Show dataset info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Dataset Information")
                    st.write(f"**Shape:** {custom_df.shape}")
                    st.write(f"**Columns:** {list(custom_df.columns)}")
                    
                    # Check if quality column exists
                    if 'quality' in custom_df.columns:
                        st.subheader("🍷 Quality Distribution")
                        quality_dist = custom_df['quality'].value_counts().sort_index()
                        fig = px.bar(x=quality_dist.index, y=quality_dist.values,
                                   title="Quality Score Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("👁️ Data Preview")
                    st.dataframe(custom_df.head(), use_container_width=True)
                
                # Data statistics
                st.subheader("📈 Statistical Summary")
                st.dataframe(custom_df.describe(), use_container_width=True)
                
                # Allow model training on custom data
                if st.button("🚀 Train Models on Custom Data", type="secondary"):
                    st.info("This feature would retrain the models with your custom dataset. For now, please use the pre-trained models.")
                
            except Exception as e:
                st.error(f"Error reading the file: {e}")
        
        else:
            st.info("Please upload a CSV file to proceed.")
            st.markdown("""
            ### Expected CSV Format:
            The CSV file should contain the following columns:
            - fixed_acidity
            - volatile_acidity
            - citric_acid
            - residual_sugar
            - chlorides
            - free_sulfur_dioxide
            - total_sulfur_dioxide
            - density
            - pH
            - sulphates
            - alcohol
            - quality (optional, for evaluation)
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Machine Learning Assignment 2 - Wine Quality Classification</p>
    <p>Built with Streamlit, Scikit-learn, and XGBoost</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
