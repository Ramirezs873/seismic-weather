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


def plot_wind_speed(data, year=None, month=None):

    """
    Plots the wind speed for a given year and month from the provided DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the weather station data.
        year (int): The year for plotting (2010-2025 or None for all years).
        month (int or tuple): The month(s) for plotting.
                              None for the entire year,
                              A single month (1-12),
                              or a tuple of (start_month, end_month) for a range of months.
    """

    
    if isinstance(year, (tuple, list)) and len(year) == 2: 
        start_year, end_year = year 
        df_slice = data[(data['datetime'].dt.year >= start_year) & 
                        (data['datetime'].dt.year <= end_year) ].copy()
    elif year is None or year < 2010 or year > 2025:
        df_slice = data.copy()
    else:
        df_slice = data[data['datetime'].dt.year == year].copy()
        

    if isinstance(month, (tuple, list)) and len(month) == 2: 
        start_month, end_month = month 
        df_slice = df_slice[(df_slice['datetime'].dt.month >= start_month) & 
                        (df_slice['datetime'].dt.month <= end_month) ].copy()
    elif isinstance(month, int): 
        df_slice = df_slice[df_slice['datetime'].dt.month == month].copy()
    

    if df_slice.empty: 
        print("No data available for the selected time period.") 
        return

    # Convert to numeric
    df_slice['Wind speed in km/h'] = pd.to_numeric(df_slice['Wind speed in km/h'], 
                                                    errors='coerce')
    wind_speed = df_slice['Wind speed in km/h']
    df_slice['Wind_norm'] = wind_speed / wind_speed.max()

    plt.figure(figsize=(15,6))
    plt.plot(df_slice['datetime'], df_slice['Wind speed in km/h'], 
            color='black', linewidth=0.5)
    ymax = np.max(df_slice['Wind speed in km/h'])
    
    if isinstance(year, (tuple, list)): 
        plt.title(f"Wind Speed at Station {df_slice['Station Number'].iloc[0]} @ {year[0]} to {year[1]}")
    elif year is None:
        plt.title(f'Wind Speed at Station:{df_slice["Station Number"].iloc[0]} @ All Years')
    elif month is not None and isinstance(month, (tuple, list)): 
        plt.title(f'Wind Speed at Station:{df_slice["Station Number"].iloc[0]} @ {year}-{month[0]} to {month[1]}')
    elif month is None:
        plt.title(f'Wind Speed at Station:{df_slice["Station Number"].iloc[0]} @ {year}')
    else:
        plt.title(f'Wind Speed at Station:{df_slice["Station Number"].iloc[0]} @ {year}-{month:02d}')
    plt.ylabel('Wind Speed (km/h)')
    plt.xlabel('Year')
    plt.ylim(0, ymax + 10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_rose_wind(data, year=None, month=None):

    """
    Plots a rose plot for wind speed and direction for a given year and month from the provided DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the weather station data.
        year (int or tuple): The year(s) for plotting.
                             None for all years, 
                             A single year (2010-2025), 
                             or a tuple of (start_year, end_year) for a range of years.
        month (int or tuple): The month(s) for plotting.
                              None for the entire year,
                              A single month (1-12),
                              or a tuple of (start_month, end_month) for a range of months.
    """

    if isinstance(year, (tuple, list)) and len(year) == 2: 
        start_year, end_year = year 
        df_slice = data[(data['datetime'].dt.year >= start_year) & 
                        (data['datetime'].dt.year <= end_year) ].copy()
    elif year is None or year < 2010 or year > 2025:
        df_slice = data.copy()
    else:
        df_slice = data[data['datetime'].dt.year == year].copy()
        

    if isinstance(month, (tuple, list)) and len(month) == 2: 
        start_month, end_month = month 
        df_slice = df_slice[(df_slice['datetime'].dt.month >= start_month) & 
                        (df_slice['datetime'].dt.month <= end_month) ].copy()
    elif isinstance(month, int): 
        df_slice = df_slice[df_slice['datetime'].dt.month == month].copy()
    

    if df_slice.empty: 
        print("No data available for the selected time period.") 
        return

    # Convert to numeric
    df_slice['Wind speed in km/h'] = pd.to_numeric(df_slice['Wind speed in km/h'], 
                                                    errors='coerce')
    df_slice['Wind direction in degrees true'] = pd.to_numeric(df_slice['Wind direction in degrees true'],
                                                               errors='coerce')
    df_slice['Wind direction in degrees true'] %= 360

    wind_speed = df_slice['Wind speed in km/h']
    df_slice['Wind_norm'] = wind_speed / wind_speed.max()

    speed_bins = [0, 10, 38, 60, 90,np.inf]
    labels = ["Light", "Moderate", "Strong", "Severe", "Extreme"]


    df_slice['speed_bin'] = pd.cut(df_slice['Wind speed in km/h'], bins=speed_bins, labels=labels)

    sector_width = 5
    df_slice['dir_bin'] = (df_slice['Wind direction in degrees true'] // sector_width) * sector_width

    freq = df_slice.groupby(['dir_bin', 'speed_bin']).size().reset_index(name='count')
    freq['percentage'] = 100 * freq['count'] / freq['count'].sum()

    fig = px.bar_polar(freq, r="percentage", theta="dir_bin", color="speed_bin", color_continuous_scale=px.colors.sequential.Plasma)
    
    if isinstance(year, (tuple, list)): 
        fig.update_layout(title=f'Wind Rose at Station:{df_slice["Station Number"].iloc[0]}  @ {year[0]} to {year[1]}',
                          polar=dict(radialaxis=dict(tickformat=".0f%", ticksuffix="%",
                                                     angle=90,
                                                     side="counterclockwise")))
    elif year is None:
        fig.update_layout(title=f'Wind Rose at Station:{df_slice["Station Number"].iloc[0]} @ All Years', 
                          polar=dict(radialaxis=dict(tickformat=".0f%", ticksuffix="%",
                                                     angle=90,
                                                     side="counterclockwise")))
    elif isinstance(month, (tuple, list)): 
        fig.update_layout(title=f'Wind Rose at Station:{df_slice["Station Number"].iloc[0]}  @ {year}-{month[0]} to {month[1]}',
                          polar=dict(radialaxis=dict(tickformat=".0f%", ticksuffix="%",
                                                     angle=90,
                                                     side="counterclockwise")))
    elif month is None:
        fig.update_layout(title=f'Wind Rose at Station:{df_slice["Station Number"].iloc[0]} @ {year}', 
                          polar=dict(radialaxis=dict(tickformat=".0f%", ticksuffix="%",
                                                     angle=90,
                                                     side="counterclockwise")))
    else:
        fig.update_layout(title=f'Wind Rose at Station:{df_slice["Station Number"].iloc[0]} @ {year}-{month:02d}', 
                          polar=dict(radialaxis=dict(tickformat=".0f%", ticksuffix="%",
                                                     angle=90,
                                                     side="counterclockwise")))

    fig.show()