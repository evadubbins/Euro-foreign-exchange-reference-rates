# Preparation: Load libraries, get the data, set first and last date for our data analysis

# Import neccessary libraries 
import pandas as pd
import zipfile
import io
import datetime
import requests

# Download the ZIP file with the data from the European Central Bank
url = 'https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip'
response = requests.get(url)

# Extract the CSV from the ZIP and load it correctly
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    with z.open('eurofxref-hist.csv') as f:
        # Read with correct delimiter (comma)
        df = pd.read_csv(f)

# Rename first column to 'Date' to make sure rhere is no error occurring 
df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

# Convert Date to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.set_index('Date', inplace=True)

# Keep only the desired currencies: USD, AUD, CHF and CNY
selected_currencies = ['USD', 'AUD', 'CHF', 'CNY']
df = df[selected_currencies]

# Sort by date
df = df.sort_index()

# Show last few rows to confirm it worked
print("The euro exchange reference rates are as follows:")
print(df.tail())

# Drop columns with NaN values and modify DataFrame
df.dropna(axis=0, inplace=True)
df


#------
# Goal 1: Simple currency converter: Input a quantity in CHF, USD, CNY or AUD and convert to Euro or vice versa for the date of yesterday 

# Get yesterday's date
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

# Convert to pandas Timestamp to match DataFrame index
yesterday = pd.Timestamp(yesterday)

# Find the latest available date in the DataFrame that is ≤ yesterday
conversion_date = df.index[df.index <= yesterday].max()

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

    # Error messages for failure
    except ValueError as e:
        print("There was an error:", e)
        print("Please try again.\n")

