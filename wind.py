from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px


def read_data(path, 
              station_code, 
              id_code):

    """
    Reads BoM weather station data from a specified path, station code, and id code. 
    Then, it organises the data into a DataFrame and creates a datetime column from the relevant date and time columns.
    
    Parameters:
        path (str): The path to the directory containing the data files.
        station_code (str): The station code for the weather station.
        id_code (str): The id code for the weather station data.

    Returns:
        df (pd.DataFrame): A DataFrame containing the weather station data.
    """
    

    
    data_path = Path(path)
    pq_path = data_path / f'HM01X_Data_{station_code}_{id_code}.parquet'
    csv_path = data_path / f'HM01X_Data_{station_code}_{id_code}.txt'

    if pq_path.exists():
        df = pd.read_parquet(pq_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=['datetime'])
    else:
        raise FileNotFoundError(f"No data file found for station {station_code} and id {id_code}")

    df = pd.read_csv(csv_path, low_memory=False)
    df["datetime"] = pd.to_datetime(dict(
                                    year=df["Year Month Day Hour Minutes in YYYY.2"],
                                    month=df["MM.2"],
                                    day=df["DD.2"],
                                    hour=df["HH24.2"],
                                    minute=df["MI format in Universal coordinated time"]))

    return df

def plot_wind_speed(data, year, month):

    """
    Plots the wind speed for a given year and month from the provided DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the weather station data.
        year (int): The year for plotting (2010-2025).
        month (int): The month for plotting (1-12).
    """

    df_year_month = {}
    for i in range(2010, 2026):
        for k in range(1, 13):

            df_slice = data[(data['datetime'].dt.year == i) &
                            (data['datetime'].dt.month == k)].copy()

            if df_slice.empty:
                continue

            # Convert to numeric
            df_slice['Wind speed in km/h'] = pd.to_numeric(df_slice['Wind speed in km/h'], 
                                                           errors='coerce')
            wind_speed = df_slice['Wind speed in km/h']
            df_slice['Wind_norm'] = wind_speed / wind_speed.max()

            df_year_month[(i, k)] = df_slice

    
    plt.figure(figsize=(15,6))
    plt.plot(df_year_month[year, month]['datetime'], df_year_month[year, month]['Wind speed in km/h'], 
            color='black', linewidth=0.5)
    ymax = np.max(df_year_month[year, month]['Wind speed in km/h'])
    plt.title(f'Wind Speed at Station:{df_year_month[year, month]["Station Number"].iloc[0]} @ {year}-{month:02d}')
    plt.ylabel('Wind Speed (km/h)')
    plt.xlabel('Year')
    plt.ylim(0, ymax + 10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
