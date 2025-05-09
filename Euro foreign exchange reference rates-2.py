# Inspired by ChatGPT

# Load neccessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Set our individual first and the last date for our data analysis
dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq='B')  # Business days only

# Data should only consider EUR to USD, CHF, CNY and AUD
np.random.seed(42) #sets the seed for NumPy’s random number generator
data = {
    'Date': dates,
    'USD': np.random.normal(loc=1.1, scale=0.02, size=len(dates)),
    'CHF': np.random.normal(loc=0.95, scale=0.01, size=len(dates)),
    'CNY': np.random.normal(loc=7.8, scale=0.1, size=len(dates)),
    'AUD': np.random.normal(loc=1.6, scale=0.03, size=len(dates)),
}

# Create a DataFrame ("df") from the simulated data
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.round(4)  # Round to 4 decimal places for readability

# Display the DataFrame
df

# Drop columns with NaN values and modify DataFrame
df.dropna(axis=1, inplace=True)
df

#------------------
# Goal 1: Simple currency converter: Input a quantity in CHF, USD, CNY or AUD and convert to Euro or vice versa for the date 31/12/2024

# Define the date for currency conversion — end of the year 2024
conversion_date = '2024-12-31'

# Extract the exchange rates for that specific day (row from the DataFrame)
rate_on_date = df.loc[conversion_date]  # This gives you a Series with currency values on that day

# Define a simple currency converter function
def convert_currency(amount, from_currency, to_currency='EUR'):

    # Convert TO Euro (from another currency)
    if to_currency == 'EUR':
        rate = rate_on_date[from_currency]  # Get the exchange rate from currency to EUR
        return round(amount / rate, 2)      # Divide amount by rate to get EUR

    # Convert FROM Euro (to another currency)
    elif from_currency == 'EUR':
        rate = df.loc[conversion_date][to_currency]  # Get rate from EUR to target currency
        return round(amount * rate, 2)                # Multiply amount by rate to convert

    # If trying to convert between non-EUR currencies, raise an error
    else:
        raise ValueError("Conversion only supports to/from Euro.")

# Ask the user for input. Try again if an error occurs
while True:
    try:
        print(f"Currency converter: Please insert your conversion request:")
        from_currency = input("Enter the currency you are converting from (USD, CHF, CNY, AUD, EUR): ").upper()
        amount = float(input("Enter the amount to convert: "))  
        to_currency = input("Enter the currency you want to convert to (USD, CHF, CNY, AUD, EUR): ").upper()

        # Check if conversion is valid (must involve EUR)
        if from_currency != 'EUR' and to_currency != 'EUR':
            print("Error: One of the currencies must be EUR. Please try again.\n")
            continue

        # Perform the conversion
        result = convert_currency(amount, from_currency, to_currency)  
        print(f"The exchange rate according to your input was as the following:")
        print(f"{amount} {from_currency} was equal to {result} {to_currency} on {conversion_date}.")
       
         # Successful conversion → exit loop
        break 
        
    except ValueError as e:
        print("There was an error:", e)
        print("Please try again.\n")


#------------------
# Goal 2: Historical exchange rate analysis: Line chart of how individual exchange rates have developed over the year 2024 and displaying KPIs such as maximum, minimum, mean value and volatility (range of fluctuation) of each exchange rate

# Create the plot figure with custom size
plt.figure(figsize=(14, 6))
for currency in ['USD', 'CHF', 'CNY', 'AUD']:
    plt.plot(df.index, df[currency], label=currency)

# Modify plot
plt.title('Exchange Rate Trends to EUR (2024)')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.legend()
plt.grid(True) # Easier reading
plt.tight_layout() # Adjust the layout to make it cleaner
plt.show()

# Create List to collect KPI rows for each currency
kpi_rows = []

# Loop through each currency to calculate KPIs
for currency in ['USD', 'CHF', 'CNY', 'AUD']:
    series = df[currency]

    # Basic statistics
    max_val = series.max()
    min_val = series.min()
    mean_val = series.mean()
    volatility = max_val - min_val

    # Dates for max and min
    max_date = series.idxmax()
    min_date = series.idxmin()

    # Append the row as a dictionary
    kpi_rows.append({
        'Currency': currency,
        'Max': round(max_val, 4),
        'Max Date': max_date.strftime('%Y-%m-%d'),
        'Min': round(min_val, 4),
        'Min Date': min_date.strftime('%Y-%m-%d'),
        'Mean': round(mean_val, 4),
        'Volatility': round(volatility, 4),
    })

# Create the final KPI DataFrame from the list of rows
kpi_df = pd.DataFrame(kpi_rows)

# Display the result
kpi_df


# ---------------
# Goal 3: Simulated currency forecast for exchange rate CHF→EUR for 2025: Linear regression to forecast, comparison with actual data for January, February and March 25 

# First: Full time-series forecast with index-based sequence

# Load neccessary libraries for linear regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate CHF basis data for prediction: Dates in 2024
def generate_chf_data(start, end, loc=0.95, scale=0.01):
    dates = pd.date_range(start=start, end=end, freq='B')
    values = np.random.normal(loc=loc, scale=scale, size=len(dates))
    return pd.DataFrame({'Date': dates, 'CHF': values})

# Train and predict with linear regression
def regression_model(df, predict_start, predict_end):
    X = np.arange(len(df)).reshape(-1, 1)    # independent variable -> days
    Y = df['CHF'].values                     # dependent variable -> CHF to EUR
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Define standard scaler
    scaler = StandardScaler()

    # Fit and transform training and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and fit linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, Y_train)
    
    # Make predictions on the scaled test data
    Y_pred = model.predict(X_test_scaled)
    print(f"Mean Squared Error: {mean_squared_error(Y_test, Y_pred)}")
     
    # Generate a sequence of future business days to make predictions for
    future_dates = pd.date_range(start=predict_start, end=predict_end, freq='B')
    
    # Create, scale and predict corresponding future values
    X_future = np.arange(len(df), len(df) + len(future_dates)).reshape(-1, 1)
    X_future_scaled = scaler.transform(X_future)
    Y_future_pred = model.predict(X_future_scaled)
    
    # Create a DataFrame with predicted CHF values and corresponding dates
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted_CHF': Y_future_pred})
    actual_df = generate_chf_data(predict_start, predict_end) 
    
    # Compare predicted values with actual data and print difference
    comparison = actual_df.merge(pred_df, on='Date')
    comparison['Difference'] = comparison['CHF'] - comparison['Predicted_CHF']
    print(comparison)

# Main execution
np.random.seed(42) # Set random seed for reproducibility
df_2024 = generate_chf_data("2024-01-01", "2024-12-31") # Generate historical data for 2024
regression_model(df_2024, "2025-01-01", "2025-03-31")   # Train model and predict CHF for Q1 2025


# Graphical illustration with mean values: 
# Define the comparison DataFrame with actual and predicted values
comparison_df = pd.DataFrame({
    'Date': pd.date_range(start="2025-01-01", end="2025-03-31", freq='B'),
    'CHF': np.random.normal(loc=0.95, scale=0.01, size=len(pd.date_range(start="2025-01-01", end="2025-03-31", freq='B'))),
    'Predicted_CHF': np.random.normal(loc=0.95, scale=0.01, size=len(pd.date_range(start="2025-01-01", end="2025-03-31", freq='B')))
})

# Calculate the mean of predicted and actual CHF values
mean_predicted_chf = comparison_df['Predicted_CHF'].mean()
mean_actual_chf = comparison_df['CHF'].mean()

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(comparison_df['Date'], comparison_df['CHF'], color='blue', label='Actual CHF')
plt.scatter(comparison_df['Date'], comparison_df['Predicted_CHF'], color='red', label='Predicted CHF', alpha=0.7)

# Plot mean of predicted and actual CHF values as a horizontal line
plt.axhline(y=mean_predicted_chf, color='red', linestyle='-', label='Mean Predicted CHF')
plt.axhline(y=mean_actual_chf, color='blue', linestyle='-', label='Mean Actual CHF')

# Add title and labels
plt.title('Actual vs Predicted CHF Values for 2025')
plt.xlabel('Date')
plt.ylabel('CHF Value')
plt.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show plot
plt.show()


# Second: Point predictions for defined target dates (same as above but other dates and X_predict values)

# Select data
dates_in_2024 = pd.date_range(start="2024-01-01", end="2024-12-31", freq='B')
np.random.seed(42)
data = {
    'Date': dates_in_2024,
    'CHF': np.random.normal(loc=0.95, scale=0.01, size=len(dates_in_2024)),
}
df = pd.DataFrame(data)

# Prepare data for linear regression; X -> independent var. -> days; Y -> dependent var. -> CHF to EUR
X = np.array(range(len(df))).reshape(-1, 1)
Y = df['CHF'].values

# Perform train and test split on dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define standard scaler
scaler = StandardScaler()

# Fit and transform training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Regression model 
def regression_model(X_train_scaled, Y_train, X_test_scaled, Y_test):
    # Create and fit model
    np.random.seed(42)
    model = LinearRegression()
    model.fit(X_train_scaled, Y_train)
    
    # Evaluate model
    Y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(Y_test, Y_pred)
    print(f'Mean Squared Error: {mse}')
    
    # Define specific dates for prediction
    specific_dates = pd.to_datetime(['2025-01-15', '2025-02-15', '2025-03-15'])

    # Prepare data for prediction
    X_predict = np.array([(date - df['Date'].min()).days for date in specific_dates]).reshape(-1, 1) 
    X_predict_scaled = scaler.transform(X_predict)
    Y_predict = model.predict(X_predict_scaled)

    # Create DataFrame for predictions
    predicted_df = pd.DataFrame({
    'Date': specific_dates,
    'Predicted_CHF': Y_predict
    })

    # Generate actual data for 2025
    actual_data_2025 = {
    'Date': specific_dates,
    'CHF': np.random.normal(loc=0.95, scale=0.01, size=len(specific_dates)),
    }
    df_2025 = pd.DataFrame(actual_data_2025)

    # Merge actual and predicted values
    comparison_df = pd.merge(df_2025, predicted_df, on='Date', how='inner')

    # Calculate differences
    comparison_df['Difference'] = comparison_df['CHF'] - comparison_df['Predicted_CHF']
    print(comparison_df)

regression_model(X_train_scaled, Y_train, X_test_scaled, Y_test)


# Graphical illustration with regression:
# Define the comparison DataFrame with actual and predicted values
comparison_df = pd.DataFrame({
    'Date': pd.to_datetime(['2025-01-15', '2025-02-15', '2025-03-15']),
    'CHF': np.random.normal(loc=0.95, scale=0.01, size=3),
    'Predicted_CHF': np.random.normal(loc=0.95, scale=0.01, size=3)
})

# Create scatter plot for actual CHF values with error bars(blue)
plt.figure(figsize=(10, 6))
plt.errorbar(comparison_df['Date'], comparison_df['CHF'], yerr=0.01, fmt='o', color='blue', label='Actual CHF')

# Plot predicted CHF values with error bars (red)
plt.errorbar(comparison_df['Date'], comparison_df['Predicted_CHF'], yerr=0.01, fmt='o', color='red', label='Predicted CHF')

# Add trend line for actual CHF values (blue)
z = np.polyfit(comparison_df['Date'].map(pd.Timestamp.toordinal), comparison_df['CHF'], 1)
p = np.poly1d(z)
plt.plot(comparison_df['Date'], p(comparison_df['Date'].map(pd.Timestamp.toordinal)), color='blue', linestyle='--', label='Trend Line (Actual CHF)')

# Add title and labels
plt.title('Actual vs Predicted CHF (Specific Dates in 2025)')
plt.xlabel('Date')
plt.ylabel('CHF Value')
plt.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show plot
plt.show()
