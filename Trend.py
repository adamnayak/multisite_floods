import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def fit_loess_smoothing(data, frac=1, it=0):
    """
    Apply LOESS smoothing to the data.

    :param data: NumPy array of the data to be smoothed.
    :param frac: The fraction of the data used when estimating each y-value.
                 Default is 0.33, meaning that roughly one-third of the data is used for each point's estimation.
    :param it: The number of robustifying iterations. The default is 0, meaning that the function will not perform any robustification iterations.
    :return: Smoothed data as a NumPy array.
    """
    # Ensure data is a 1D array and create an array of x-values to match it
    data = np.asarray(data).flatten()
    x_values = np.arange(len(data))

    # Apply LOESS smoothing
    smoothed = lowess(endog=data, exog=x_values, frac=frac, it=it, return_sorted=False)

    return smoothed


def linear_forecast(smoothed, steps, time, og_data, data_type, plot=True):
    """
    Forecasts future values using linear regression on a time series.

    Parameters:
    - smoothed: Numeric sequence of smoothed data points for fitting (list, NumPy array).
    - steps: Number of future data points to predict.

    Returns:
    - Tuple of (forecasted values, y-intercept of the trendline).
    """
    x_values = np.arange(len(smoothed))
    coefficients = np.polyfit(x_values, smoothed, 1)  # Linear fit
    trendline = np.poly1d(coefficients)
    future_x_values = np.arange(len(smoothed), len(smoothed) + steps)
    historical_forecasts = trendline(list(range(len(smoothed))))
    future_forecasts = trendline(future_x_values)
    intercept = trendline.coefficients[1]

    if plot:
        # Plot detrended ann max time series
        plt.figure(figsize=(10, 6))
        plt.plot(time, og_data, marker='o', linestyle='-', color='b', label = 'Annual Max')
        plt.plot(time, smoothed, linestyle='-', color='r', label = 'Exp Smoothing', linewidth = 2)
        plt.plot(time, historical_forecasts, linestyle='-', color='g', label = 'Polyfit', linewidth = 3)
        plt.title(f'Exponential Smoothing of Annual Maximum {data_type} Over Years')
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        plt.show()

    return future_forecasts, intercept


def perform_lasso_regression(X_sig, y, X_forecast, merged_df, data_type, data_unit, name='', plot_LASSO=False):
    """
    Performs LASSO regression on provided datasets, with an option to plot results.
    
    Parameters:
    - X_sig: Pandas DataFrame of significant variables for regression.
    - y: Series or array-like, target variable.
    - X_forecast: DataFrame or array-like for future forecasts.
    - merged_df: DataFrame containing original data for plotting.
    - data_type: String, type of data for labeling plots.
    - data_unit: String, unit of the data for labeling plots.
    - plot_LASSO: Boolean, controls the plotting of LASSO results.
    
    Returns:
    - A dictionary containing the best alpha, RMSE, and predictions.
    """
    
    def plot_results():
        """
        Plots the original data, LOESS smoothing, and LASSO regression fit.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(merged_df.index, merged_df['Original_Data'], marker='o', linestyle='-', color='#4477AA', label='Annual Max')
        plt.plot(merged_df.index, y, label='LOESS Smoothing', color='#EE6677', linestyle='-', marker='')
        plt.plot(merged_df.index, predictions_all, label='LASSO Regression Fit', color='#228833', linestyle='-')
        
        plt.title(f'{name} Regression Fit on Historical Data', fontsize=14)
        plt.xlabel('Time Periods', fontsize=12)
        plt.ylabel(f'{data_type} ({data_unit})', fontsize=12)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=3, fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.show()
    
    # Splitting dataset
    X_train_sig, X_test_sig, y_train, y_test = train_test_split(X_sig, y, test_size=0.3, random_state=42)
    
    # Initialize and fit LassoCV
    lasso_cv = LassoCV(cv=5, random_state=0)
    lasso_cv.fit(X_train_sig, y_train)
    
    # Outputs
    best_alpha = lasso_cv.alpha_
    print(f"Best alpha found: {best_alpha}")
    
    predictions_sig = lasso_cv.predict(X_test_sig)
    predictions_all = lasso_cv.predict(X_sig)
    forecast_lin = lasso_cv.predict(X_forecast)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions_sig))
    print(f"RMSE: {rmse}")

    print("LASSO weights for each covariate:")
    for i, col in enumerate(X_sig.columns):
        print(f"{col}: {lasso_cv.coef_[i]}")
    
    results = {
        'best_alpha': best_alpha,
        'RMSE': rmse,
        'predictions': forecast_lin
    }
    
    if plot_LASSO:
        plot_results()
    
    return results


def random_forest_analysis(X, y, v_names, trend_max_lags, var_interest, RF_thres=0):
    """
    Performs Random Forest regression, calculates RMSE, and optionally filters significant variables.
    
    Parameters:
    - X: DataFrame or array-like, feature dataset.
    - y: Series or array-like, target variable.
    - v_names: List of strings, names of the variables in X.
    - trend_max_lags: Integer, the maximum number of lags considered for trend analysis.
    - var_interest: String, the variable of interest for the analysis.
    - RF_thres: Float, the threshold for determining significant variables based on their importance.
    - RF_filter: Boolean, if True, filters the variables that are significant based on the specified threshold.
    
    Returns:
    - A dictionary containing the RMSE of the model, feature importances, and optionally a list of significant variables.
    """
    # Splitting dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predictions
    predictions = rf.predict(X_test)
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    # Feature importances
    feature_importances = rf.feature_importances_ * 100
    
    summary_importance = pd.DataFrame({
        "Variable": v_names,
        "Importance (%)": [round(val, 2) for val in feature_importances]})
    
    # Print the results
    print(f"Relative feature importance for up to {trend_max_lags} lags for {var_interest} and CO2 Concentration: \n{summary_importance}")
    
    # Optionally filter significant variables
    significant_vars = summary_importance[summary_importance['Importance (%)'] > RF_thres]['Variable'].tolist()
    print(f"Significant variables (Importance > {RF_thres}%): {significant_vars}")
    
    return {
        'RMSE': rmse,
        'Feature Importances': summary_importance,
        'Significant Variables': significant_vars
    }