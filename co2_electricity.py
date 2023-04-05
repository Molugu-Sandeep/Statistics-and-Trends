import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_and_process_data(filepath):
    '''
    This function accepts path to a file and reads the CSV file to pandas dataframe
    
    Parameters
    ----------
        filepath: str
        
    Returns
    -------
        pandas.DataFrame, pandas.DataFrame
    '''    
    df = pd.read_csv(filepath, na_values = '..')
    df = df.iloc[:, 2:]  # Drop unneeded columns
    df.drop(columns = ['Country Code'], inplace = True)  # Drop Country Code Column
    df = df.set_index('Country Name')  # Set Countries to Index
    df = df.apply(pd.to_numeric, errors = 'coerce')  # Convert to numeric
    df = df[df.index.notna()]  # Drop NaN rows on the index
    df = df.iloc[:217,:] # Select only countries
    
    for col in df.columns:  # Drop all the columns which have all missing values in rows
        if all(df[col].isna()):
            df = df.drop(columns = [col])

    cols = df.columns.tolist() # Process column names
    cols = [year[:4] for year in cols]
    df.columns = cols
    
    return df, df.T


# READING IN THE DATA
co2_year, co2_country = read_and_process_data('CO2.csv')
electricity_year, electricity_country = read_and_process_data('Electricity.csv')


# SUMMARY STATISTICS
summary_co2 = co2_year.describe().T
summary_electricity = electricity_year.describe().T


# plot of the mean value per year
x = summary_co2.index
y1 = summary_co2['mean']
y2 = summary_electricity['mean']
fig, ax1 = plt.subplots(figsize = (8, 6))
ax1.plot(x, y1, label = "CO2 emission (kt)")
ax2 = ax1.twinx()  # create a secondary axis 
ax2.plot(x, y2, label = "Electric Consumption", color = 'red')
plt.xlabel('Year')
ax1.set_ylabel('CO2 Emission (kt)')
ax2.set_ylabel('Electric Consumption (kWh/Capita)')
plt.title("Mean CO2 Emission vs Mean Electric Consumption")
ax1.set_xticks(ax1.get_xticks())
ax1.set_xticklabels(labels = x, rotation = 60)
ax1.legend(loc = 'upper left'), ax2.legend(loc = 'upper right')
plt.grid()
plt.show()


np.random.seed(199)
co2_mean = co2_country.mean().sample(10).sort_values().dropna()  # Mean CO2 emissions/country
countries_comparison = co2_mean.index  # To repeat in other visualisations

# # Bar Plot of mean CO2 emission (Not used in the report)
# co2_mean.plot(kind = 'bar', figsize = (6, 4))
# plt.title("Mean CO2 Emission/Country")
# plt.xlabel("Country Name")
# plt.ylabel("CO2 Emission (kt)")
# plt.show()


# Co2 emission Per Country for the last 25 years
co2_year.loc[countries_comparison, :].T.plot(kind = 'line')
plt.legend(bbox_to_anchor = [1,0.8])
plt.title('CO2 Emission')
plt.xlabel('Time')
plt.ylabel('CO2 Emission (kt)')
plt.show()

# Electricity Consumption Per Country for the last 25 years
electricity_year.loc[countries_comparison, :].T.plot(kind = 'line')
plt.legend(bbox_to_anchor = [1,0.8])
plt.title('Electricity Consumption')
plt.xlabel('Time')
plt.ylabel('Electricity Consumption (kWh)')
plt.show()



# Standard Deviation of CO2 emission Per Country
co2_std = co2_country.std()[countries_comparison].dropna()
co2_std.plot(kind = 'pie', figsize = (6, 4))
plt.title("Standard Deviation of CO2 Emission")
plt.show()


# Grouped Bar Plot; CO2
co2_year.loc[countries_comparison, :].plot(kind = 'bar', figsize = (6, 4))
plt.title("CO2 Emission per country")
plt.xlabel("Country Name")
plt.ylabel("CO2 Emission (kt)")
plt.legend(bbox_to_anchor = [1,1])
plt.show()

# Grouped Bar Plot; Electricity
electricity_year.loc[countries_comparison, :].plot(kind = 'bar', figsize = (6, 4))
plt.title("Electricity Consumption")
plt.xlabel("Country Name")
plt.ylabel("Electricity (kWh)")
plt.legend(bbox_to_anchor = [1,1])
plt.show()


# Investigate for correlation across countries; CO2 and Electricity
countries = []
correlation = []
for i in range(len(countries_comparison)):
    country = countries_comparison[i]
    co2 = co2_year.loc[country, :].dropna()
    ect = electricity_year.loc[country, :].dropna()
    # Filter countries whose NaN's have been dropped
    if len(co2) < len(ect) > 1:
        ix = co2.index
        ect = ect[ix]
        r = np.corrcoef(co2, ect)[0,1]
        countries.append(country)
        correlation.append(r)
    elif len(co2) > len(ect) > 1:
        ix = ect.index
        co2 = co2[ix]
        r = np.corrcoef(co2, ect)[0,1]
        countries.append(country)
        correlation.append(r)
pd.DataFrame({'Country': countries, 'Correlation (CO2 vs Electricty)': correlation})





