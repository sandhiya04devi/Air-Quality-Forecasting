import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objs as go

class AirQualityAnalyzer:
    def __init__(self, data_path):
        """
        Initialize Air Quality Analyzer
        
        Args:
            data_path (str): Path to preprocessed air quality data
        """
        try:
            # Read data with explicit datetime parsing
            self.df = pd.read_csv(data_path, parse_dates=['date'])
            
            # Validate data
            self._validate_data()
        except FileNotFoundError:
            st.error(f"Error: Data file not found at {data_path}")
            raise
        except pd.errors.EmptyDataError:
            st.error("Error: The data file is empty.")
            raise
        except Exception as e:
            st.error(f"Unexpected error loading data: {e}")
            raise
    
    def _validate_data(self):
        """
        Validate input data integrity
        """
        # Check for required columns
        required_columns = ['date', 'state', 'PM2.5']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Check for non-null values
        if self.df[required_columns].isnull().any().any():
            st.warning("Warning: Dataset contains null values. Consider cleaning the data.")
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            st.warning("Converting date column to datetime.")
            self.df['date'] = pd.to_datetime(self.df['date'])
    
    def get_available_states(self):
        """
        Get list of available states
        
        Returns:
            list: Unique states in the dataset
        """
        try:
            return sorted(self.df['state'].unique().tolist())
        except Exception as e:
            st.error(f"Error retrieving states: {e}")
            return []
    
    def prepare_state_data(self, state, n_lags=10):
        """
        Prepare data for a specific state
        
        Args:
            state (str): State name
            n_lags (int): Number of lag features
        
        Returns:
            tuple: Processed features, target, scaler, dates
        """
        try:
            # Use .loc for safe data selection
            state_data = self.df.loc[self.df['state'].str.lower() == state.lower()].copy()
            
            if state_data.empty:
                st.error(f"No data found for state: {state}")
                return None, None, None, None
            
            state_data = state_data.sort_values('date')
            state_data.set_index('date', inplace=True)
            
            # Prepare time series data
            data = state_data['PM2.5'].values
            
            # Create lagged features
            X, y = self._create_time_series_features(data, n_lags)
            
            # Scale data
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            y_scaled = scaler.fit_transform(y.reshape(-1, 1))
            
            return X_scaled, y_scaled, scaler, state_data.index[n_lags:]
        
        except Exception as e:
            st.error(f"Error preparing state data: {e}")
            return None, None, None, None
    
    def _create_time_series_features(self, data, n_lags):
        """
        Create time series features with lag
        
        Args:
            data (np.array): Time series data
            n_lags (int): Number of lag features
        
        Returns:
            tuple: Features and targets
        """
        features = []
        targets = []
        
        for i in range(len(data) - n_lags):
            lag_sequence = data[i:i+n_lags]
            features.append(lag_sequence)
            targets.append(data[i+n_lags])
        
        return np.array(features), np.array(targets)
    
    def train_model(self, state):
        """
        Train and evaluate model for a specific state
        
        Args:
            state (str): State name
        
        Returns:
            dict: Model training results
        """
        try:
            # Prepare data
            X, y, scaler, dates = self.prepare_state_data(state)
            
            if X is None or y is None:
                st.error("Could not prepare data for modeling.")
                return None, None, None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train.ravel())
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            
            # Performance metrics
            results = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            return model, scaler, results
        
        except Exception as e:
            st.error(f"Error training model: {e}")
            return None, None, None
    
    def forecast(self, state, model, scaler, days_ahead=30):
        """
        Forecast air quality for a state
        
        Args:
            state (str): State name
            model (RandomForestRegressor): Trained model
            scaler (MinMaxScaler): Data scaler
            days_ahead (int): Number of days to forecast
        
        Returns:
            pd.DataFrame: Forecast results
        """
        try:
            # Prepare data
            X, y, _, dates = self.prepare_state_data(state)
            
            if X is None or y is None:
                st.error("Could not prepare data for forecasting.")
                return None
            
            # Forecast
            last_sequence = X[-1]
            forecasts = []
            
            for _ in range(days_ahead):
                # Predict next value (extract scalar value)
                next_value = model.predict(last_sequence.reshape(1, -1))[0]
                forecasts.append(next_value)
                
                # Update sequence
                last_sequence = np.roll(last_sequence, -1)
                last_sequence[-1] = next_value
            
            # Inverse transform
            forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))
            
            # Create forecast dataframe
            forecast_dates = pd.date_range(start=dates[-1], periods=days_ahead+1)[1:]
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'predicted_pm25': forecasts.flatten()
            })
            
            return forecast_df
        
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
            return None
    
    def analyze_seasonal_trends(self, state):
        """
        Analyze seasonal trends in PM2.5 concentrations
        
        Args:
            state (str): State name
        
        Returns:
            pd.DataFrame: Monthly average PM2.5 values
        """
        try:
            # Use .loc for safe data selection
            state_data = self.df.loc[self.df['state'].str.lower() == state.lower()].copy()
            
            if state_data.empty:
                st.error(f"No data found for state: {state}")
                return None
            
            # Extract month and calculate monthly averages
            state_data['month'] = state_data['date'].dt.month
            monthly_avg = state_data.groupby('month')['PM2.5'].mean().reset_index()
            
            # Map month numbers to names
            month_names = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April', 
                5: 'May', 6: 'June', 7: 'July', 8: 'August', 
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }
            monthly_avg['month_name'] = monthly_avg['month'].map(month_names)
            
            return monthly_avg
        
        except Exception as e:
            st.error(f"Error analyzing seasonal trends: {e}")
            return None

def main():
    st.set_page_config(page_title="Air Quality Forecast", page_icon="🌍", layout="wide")
    
    st.title("🌍 Air Quality Forecast and Analysis")
    
    # Load data
    try:
        data_path = 'data/india_air_quality.csv'
        analyzer = AirQualityAnalyzer(data_path)
    except Exception:
        st.error("Failed to initialize the application. Please check the data source.")
        return
    
    # State selection
    states = analyzer.get_available_states()
    if not states:
        st.error("No states available in the dataset.")
        return
    
    selected_state = st.selectbox("Select State", states)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Forecast", "Model Performance", "Seasonal Trends"])
    
    with tab1:
        st.header(f"Air Quality Forecast for {selected_state}")
        
        # Train model
        with st.spinner('Training Model...'):
            model, scaler, performance = analyzer.train_model(selected_state)
        
        if model is None:
            st.error("Failed to train the model. Please check your data.")
        else:
            # Generate forecast
            with st.spinner('Generating Forecast...'):
                forecast_df = analyzer.forecast(selected_state, model, scaler)
            
            if forecast_df is not None:
                # Visualization
                fig = px.line(forecast_df, x='date', y='predicted_pm25', 
                              title=f'PM2.5 Forecast for {selected_state}')
                st.plotly_chart(fig)
                
                # Forecast table
                st.dataframe(forecast_df)
            else:
                st.error("Could not generate forecast.")
    
    with tab2:
        st.header("Model Performance Metrics")
        
        if performance is None:
            st.error("No performance metrics available.")
        else:
            # Display performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Squared Error", f"{performance['mse']:.4f}")
            with col2:
                st.metric("Root Mean Squared Error", f"{performance['rmse']:.4f}")
            with col3:
                st.metric("Mean Absolute Error", f"{performance['mae']:.4f}")
            with col4:
                st.metric("R² Score", f"{performance['r2']:.4f}")
    
    with tab3:
        st.header("Seasonal Pollution Trends")
        
        # Seasonal analysis
        monthly_avg = analyzer.analyze_seasonal_trends(selected_state)
        
        if monthly_avg is not None:
            # Visualization
            fig_seasonal = px.bar(monthly_avg, x='month_name', y='PM2.5', 
                                   title=f'Monthly Average PM2.5 in {selected_state}')
            st.plotly_chart(fig_seasonal)
            
            # Additional insights
            best_month = monthly_avg.loc[monthly_avg['PM2.5'].idxmin(), 'month_name']
            worst_month = monthly_avg.loc[monthly_avg['PM2.5'].idxmax(), 'month_name']
            
            st.markdown(f"""
            **Best Month:** {best_month}
            **Worst Month:** {worst_month}
            """)
        else:
            st.error("Could not analyze seasonal trends.")

if __name__ == "__main__":
    main()