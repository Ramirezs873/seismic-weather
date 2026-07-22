from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px
from obspy.signal.util import smooth
from obspy import UTCDateTime as UTC
from scipy.stats import linregress
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from obspy import Trace
from obspy import Stream
from collections import defaultdict
from datetime import timedelta
from obspy import read
from scipy.fft import fft,fftfreq, rfft, rfftfreq
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

def read_data(path, 
              station_code, 
              id_code):

    """
    Reads BoM weather station data from a 
    specified path, station code, and id code. 
    Then, it organises the data into a DataFrame 
    and creates a datetime column 
    from the relevant date and time columns.
    
    Parameters:
        path (str): 
            The path to the directory containing the data files.
        station_code (str): 
            The station code for the weather station.
        id_code (str): 
            The id code for the weather station data.
    Returns:
        df (pd.DataFrame): 
            A DataFrame containing the weather station data.
    """
    
    # Define path
    data_path = Path(path)
    pq_path = data_path / f'HM01X_Data_{station_code}_{id_code}.parquet'
    csv_path = data_path / f'HM01X_Data_{station_code}_{id_code}.txt'
    if pq_path.exists():
        df = pd.read_parquet(pq_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        raise FileNotFoundError(f"No data file found for station {station_code} and id {id_code}")

    # Create efficient UTC datetime columns.
    if "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(dict(
                                        year=df["Year Month Day Hour Minutes in YYYY.2"],
                                        month=df["MM.2"],
                                        day=df["DD.2"],
                                        hour=df["HH24.2"],
                                        minute=df["MI format in Universal coordinated time"]))

    return df

def resample_aws(data, year, month, day, hour, freq="30min"):
    """
    Resample wind data to a regular datetime grid.
    Missing timestamps become NaN rows automatically.
    
    Parameters:
    data (pd.DataFrame):
        The DataFrame containing the weather station data.
    freq (str):
        Time interval for resample.
        e.g "30min"
    """

    # Check if time inputs are single valued or a range.
    # Year
    if isinstance(year, (tuple, list)) and len(year) == 2: 
        start_year, end_year = year 
        df_slice = data[(data['datetime'].dt.year >= start_year) & 
                        (data['datetime'].dt.year <= end_year) ].copy()
    elif year is None or year < 2010 or year > 2025:
        df_slice = data.copy()
    else:
        df_slice = data[data['datetime'].dt.year == year].copy()
    # Month
    if isinstance(month, (tuple, list)) and len(month) == 2: 
        start_month, end_month = month 
        df_slice = df_slice[(df_slice['datetime'].dt.month >= start_month) & 
                        (df_slice['datetime'].dt.month <= end_month) ].copy()
    elif isinstance(month, int): 
        df_slice = df_slice[df_slice['datetime'].dt.month == month].copy()
    # Day
    if isinstance(day, (tuple, list)) and len(day) == 2:
        start_day, end_day = day
        df_slice = df_slice[(df_slice['datetime'].dt.day >= start_day) & 
                            (df_slice['datetime'].dt.day <= end_day)].copy()
    elif isinstance(day, int):
        df_slice = df_slice[df_slice['datetime'].dt.day == day].copy()
    # Hour
    if isinstance(hour, (tuple, list)) and len(hour) == 2:
        start_hour, end_hour = hour
        df_slice = df_slice[(df_slice['datetime'].dt.hour >= start_hour) & 
                            (df_slice['datetime'].dt.hour <= end_hour)].copy()
    elif isinstance(hour, int):
        df_slice = df_slice[df_slice['datetime'].dt.hour == hour].copy()
    
    # Check if there is data
    if df_slice.empty: 
        print("No data available for the selected time period.") 
        return
    
    df_slice = df_slice.set_index("datetime")
    df = df_slice.resample(freq).asfreq() 
    df.index.name = "datetime"
    

    return df.reset_index()

def wind_speed(data, 
                year=None, 
                month=None, 
                day=None,
                hour=None,
                plot = True,
                apply_smooth = False,
                smoothie = 3):

    """
    Plots the wind speed for a given year and month from the provided DataFrame.

    Parameters:
    data (pd.DataFrame): 
        The DataFrame containing the weather station data.
    year (int): 
        The year for plotting (2010-2025 or None for all years).
    month (int or tuple): 
        The month(s) for plotting.
        None for the entire year,
        A single month (1-12),
        or a tuple of (start_month, end_month) for a range of months.
    day (int or tuple): 
        The day(s) for plotting.
        None for the entire month,
        A single day (1-31),
        or a tuple of (start_day, end_day) for a range of days.
    hour (int or tuple):
        The hour(s) for plotting.
        None for entire day,
        A single hour (0-23),
        or a tuple of (start_hour, end_hour) for a range of hours.
    apply_smooth (bool): 
        True/False. Applying ObsPy smooth() function.
    smoothie (int):
        Number of values to calculate moving average for smoothing.
    """
    
    # Check if time inputs are single valued or a range.
    # Year
    if isinstance(year, (tuple, list)) and len(year) == 2: 
        start_year, end_year = year 
        df_slice = data[(data['datetime'].dt.year >= start_year) & 
                        (data['datetime'].dt.year <= end_year) ].copy()
    elif year is None or year < 2010 or year > 2025:
        df_slice = data.copy()
    else:
        df_slice = data[data['datetime'].dt.year == year].copy()
    # Month
    if isinstance(month, (tuple, list)) and len(month) == 2: 
        start_month, end_month = month 
        df_slice = df_slice[(df_slice['datetime'].dt.month >= start_month) & 
                        (df_slice['datetime'].dt.month <= end_month) ].copy()
    elif isinstance(month, int): 
        df_slice = df_slice[df_slice['datetime'].dt.month == month].copy()
    # Day
    if isinstance(day, (tuple, list)) and len(day) == 2:
        start_day, end_day = day
        df_slice = df_slice[(df_slice['datetime'].dt.day >= start_day) & 
                            (df_slice['datetime'].dt.day <= end_day)].copy()
    elif isinstance(day, int):
        df_slice = df_slice[df_slice['datetime'].dt.day == day].copy()
    # Hour
    if isinstance(hour, (tuple, list)) and len(hour) == 2:
        start_hour, end_hour = hour
        df_slice = df_slice[(df_slice['datetime'].dt.hour >= start_hour) & 
                            (df_slice['datetime'].dt.hour <= end_hour)].copy()
    elif isinstance(hour, int):
        df_slice = df_slice[df_slice['datetime'].dt.hour == hour].copy()
    
    # Check if there is data
    if df_slice.empty: 
        print("No data available for the selected time period.") 
        return

    # Convert to numeric
    df_slice['Wind speed in km/h'] = pd.to_numeric(df_slice['Wind speed in km/h'], 
                                                    errors='coerce')
    wind_speed = df_slice['Wind speed in km/h']

    # Normalisation
    # df_slice['Wind_norm'] = wind_speed / wind_speed.max()

    # Smoothing
    if apply_smooth == True:
        wind_speed = smooth(wind_speed.to_numpy(), smoothie)
    else:
        wind_speed = wind_speed.to_numpy()

    # Clean Data
    valid_mask = ~np.isnan(wind_speed)

    wind_speed = wind_speed[valid_mask]
    time = df_slice['datetime'].to_numpy()[valid_mask]

    if plot == True:
        
        # Create Figure 
        plt.figure(figsize=(15,6))

        # Plot
        plt.plot(time, wind_speed, 
                color='black', linewidth=0.5)
        
        # Title construction
        title = f"Wind Speed at Station {df_slice['Station Number'].iloc[0]} for "
        # Year
        if isinstance(year, (tuple, list)):
            title += f"{year[0]} to {year[1]}"
        elif year is None:
            title += "All Years"
        else:
            title += f"{year}"
        # Month
        if isinstance(month, (tuple, list)):
            title += f", Months:{month[0]} to {month[1]}"
        elif isinstance(month, int):
            title += f", Month:{month}"
        # Day
        if isinstance(day, (tuple, list)):
            title += f", Days:{day[0]} to {day[1]}"
        elif isinstance(day, int):
            title += f", Day:{day}"
        # Hour
        if isinstance(hour, (tuple, list)):
            title += f", Hours:{hour[0]} to {hour[1]}"
        elif isinstance(hour, int):
            title += f", Hour:{hour}"
        plt.title(title)

        # Plot Formating
        plt.ylabel('Wind Speed (km/h)')
        plt.xlabel('Time')
        ymax = np.max(wind_speed) # For y axis limit
        plt.ylim(0, ymax + 10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return time, wind_speed

def plot_rose_wind(data, 
                   year=None, 
                   month=None, 
                   day=None,
                   hour=None):

    """
    Plots a rose plot of wind speed and direction 
    for a given year and month from the provided DataFrame.

    Parameters:
        data (pd.DataFrame): 
            The DataFrame containing the weather station data.
        year (int or tuple): 
            The year(s) for plotting.
            None for all years, 
            A single year (2010-2025), 
            or a tuple of (start_year, end_year) for a range of years.
        month (int or tuple): 
            The month(s) for plotting.
            None for the entire year,
            A single month (1-12),
            or a tuple of (start_month, end_month) for a range of months.
        day (int or tuple): 
        The day(s) for plotting.
            None for the entire month,
            A single day (1-31),
            or a tuple of (start_day, end_day) for a range of days.
        hour (int or tuple):
            The hour(s) for plotting.
            None for entire day,
            A single hour (0-23),
            or a tuple of (start_hour, end_hour) for a range of hours.
    """

    # Check if time inputs are single valued or a range.
    # Year
    if isinstance(year, (tuple, list)) and len(year) == 2: 
        start_year, end_year = year 
        df_slice = data[(data['datetime'].dt.year >= start_year) & 
                        (data['datetime'].dt.year <= end_year) ].copy()
    elif year is None or year < 2010 or year > 2025:
        df_slice = data.copy()
    else:
        df_slice = data[data['datetime'].dt.year == year].copy()
    # Month
    if isinstance(month, (tuple, list)) and len(month) == 2: 
        start_month, end_month = month 
        df_slice = df_slice[(df_slice['datetime'].dt.month >= start_month) & 
                        (df_slice['datetime'].dt.month <= end_month) ].copy()
    elif isinstance(month, int): 
        df_slice = df_slice[df_slice['datetime'].dt.month == month].copy()
    # Day
    if isinstance(day, (tuple, list)) and len(day) == 2:
        start_day, end_day = day
        df_slice = df_slice[(df_slice['datetime'].dt.day >= start_day) & 
                            (df_slice['datetime'].dt.day <= end_day)].copy()
    elif isinstance(day, int):
        df_slice = df_slice[df_slice['datetime'].dt.day == day].copy()
    # Hour
    if isinstance(hour, (tuple, list)) and len(hour) == 2:
        start_hour, end_hour = hour
        df_slice = df_slice[(df_slice['datetime'].dt.hour >= start_hour) & 
                            (df_slice['datetime'].dt.hour <= end_hour)].copy()
    elif isinstance(hour, int):
        df_slice = df_slice[df_slice['datetime'].dt.hour == hour].copy()
    
    # Check if there is data.
    if df_slice.empty: 
        print("No data available for the selected time period.") 
        return

    # Convert to numeric
    df_slice['Wind speed in km/h'] = pd.to_numeric(df_slice['Wind speed in km/h'], 
                                                    errors='coerce')
    df_slice['Wind direction in degrees true'] = pd.to_numeric(df_slice['Wind direction in degrees true'],
                                                               errors='coerce')
    df_slice['Wind direction in degrees true'] %= 360

    # Create variables
    wind_speed = df_slice['Wind speed in km/h']
    wind_dir = df_slice['Wind direction in degrees true']
    
    # Normalisation
    # df_slice['Wind_norm'] = wind_speed / wind_speed.max()

    # Define 'bins'. Labels to categorise wind speed.
    speed_bins = [0, 10, 38, 60, 90,np.inf]
    labels = ["Light", "Moderate", "Strong", "Severe", "Extreme"]
    # Create bin variables
    df_slice['speed_bin'] = pd.cut(wind_speed, bins=speed_bins, labels=labels)
    sector_width = 5
    df_slice['dir_bin'] = (wind_dir // sector_width) * sector_width
    # Create frequency variables
    freq = df_slice.groupby(['dir_bin', 'speed_bin']).size().reset_index(name='count')
    freq['percentage'] = 100 * freq['count'] / freq['count'].sum()

    #Title construction
    title = f"Wind Rose at Station:{df_slice['Station Number'].iloc[0]} @ "
    # Year
    if isinstance(year, (tuple, list)):
        title += f"{year[0]} to {year[1]}"
    elif year is None:
        title += "All Years"
    else:
        title += f"{year}"
    # Month
    if isinstance(month, (tuple, list)):
        title += f", Months:{month[0]} to {month[1]}"
    elif isinstance(month, int):
        title += f", Month:{month}"
    # Day
    if isinstance(day, (tuple, list)):
        title += f", Days:{day[0]} to {day[1]}"
    elif isinstance(day, int):
        title += f", Day:{day}"
    # Hour
    if isinstance(hour, (tuple, list)):
        title += f", Hours:{hour[0]} to {hour[1]}"
    elif isinstance(hour, int):
        title += f", Hour:{hour}"

    #Figure 
    fig = px.bar_polar(freq, 
                       r="percentage", 
                       theta="dir_bin", 
                       color="speed_bin", 
                       color_continuous_scale=px.colors.sequential.Plasma)
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                tickformat=".0f%",
                ticksuffix="%",
                angle=90,
                side="counterclockwise")))
    fig.show()

def compare_seismic_wind(seismic_data, 
                         wind_data, 
                         wind_year=None,
                         wind_month=None,
                         wind_day=None,
                         wind_hour=None,
                         apply_wind_smooth = True,
                         wind_smoothie = 3,
                         apply_seismic_smooth = True,
                         seismic_smoothie = 3,
                         rectify_seis = True):
    
    """
    Compares seismic data with wind data by plotting 
    the North-South and East-West components of both datasets.
    Inspection is only useful for identical time periods 
    but the process works for unrelated time periods. 
    For useful comparisons, trim the seismic data to 
    the same time period you specify for the wind data.

    Parameters:
    seismic_data (dict): 
        A dictionary containing seismic data with isolated components for each station.
    wind_data (pd.DataFrame): 
        A DataFrame containing the wind data.
    wind_year (int or tuple): 
        The year(s) for filtering the wind data. 
        None for all years, 
        A single year (2010-2025), 
        or a tuple of (start_year, end_year) for a range of years.
    wind_month (int or tuple): 
        The month(s) for filtering the wind data. 
        None for the entire year, 
        A single month (1-12), 
        or a tuple of (start_month, end_month) for a range of months.
    wind_day (int or tuple): 
        The day(s) for filtering the wind data. 
        None for the entire month,
        A single day (1-31),
        or a tuple of (start_day, end_day) for a range of days.
    apply_wind_smooth (bool): 
        True/False. Applying ObsPy smooth() function to wind data.
    wind_smoothie (int):
        Number of values to calculate moving average for wind smoothing.
    apply_seismic_smooth (bool): 
        True/False. Applying ObsPy smooth() function to seismic data.
    seismic_smoothie (int):
        Number of values to calculate moving average for seismic smoothing.
    rectify_seis (bool):
        True/False. Rectify seismic data.
    """
    
    # Check if time inputs are single valued or a range.
    # Year
    if isinstance(wind_year, (tuple, list)) and len(wind_year) == 2: 
        start_year, end_year = wind_year 
        df_slice = wind_data[(wind_data['datetime'].dt.year >= start_year) & 
                        (wind_data['datetime'].dt.year <= end_year) ].copy()
    elif wind_year is None or wind_year < 2010 or wind_year > 2025:
        df_slice = wind_data.copy()
    else:
        df_slice = wind_data[wind_data['datetime'].dt.year == wind_year].copy()
    # Month
    if isinstance(wind_month, (tuple, list)) and len(wind_month) == 2: 
        start_month, end_month = wind_month 
        df_slice = df_slice[(df_slice['datetime'].dt.month >= start_month) & 
                        (df_slice['datetime'].dt.month <= end_month) ].copy()
    elif isinstance(wind_month, int): 
        df_slice = df_slice[df_slice['datetime'].dt.month == wind_month].copy()
    # Day
    if isinstance(wind_day, (tuple, list)) and len(wind_day) == 2:
        start_day, end_day = wind_day
        df_slice = df_slice[(df_slice['datetime'].dt.day >= start_day) & 
                            (df_slice['datetime'].dt.day <= end_day)].copy()           
    elif isinstance(wind_day, int):
        df_slice = df_slice[df_slice['datetime'].dt.day == wind_day].copy()
    # Hour
    if isinstance(wind_hour, (tuple, list)) and len(wind_hour) == 2:
        start_hour, end_hour = wind_hour
        df_slice = df_slice[(df_slice['datetime'].dt.hour >= start_hour) & 
                            (df_slice['datetime'].dt.hour <= end_hour)].copy()
    elif isinstance(wind_hour, int):
        df_slice = df_slice[df_slice['datetime'].dt.hour == wind_hour].copy()

    # Check if there is any data.
    if df_slice.empty: 
        print("No wind data available for the selected time period.") 
        return

    # Convert to numeric
    df_slice['Wind speed in km/h'] = pd.to_numeric(df_slice['Wind speed in km/h'], 
                                                    errors='coerce')
    df_slice['Wind direction in degrees true'] = pd.to_numeric(df_slice['Wind direction in degrees true'],
                                                               errors='coerce')
    df_slice['Wind direction in degrees true'] %= 360
    
    #Normalisation
    # df_slice['Wind_norm'] = wind_speed / wind_speed.max()

    # Create Variables
    WD = df_slice['Wind direction in degrees true']
    wind_speed = df_slice['Wind speed in km/h']

    # WS_norm = df_slice['Wind_norm']
    WS = abs(wind_speed) # Rectified

    # Isolate North-South and East-West wind components
    # Create storage
    u = []
    v = []
    # Loop through all wind speeds and their directions to isolate orthogonal components.
    for i in range(len(WS)):
    # North-South component
        u.append(np.abs(WS.iloc[i] * np.cos(np.deg2rad(WD.iloc[i]))))
    # East-West component
        v.append(np.abs(WS.iloc[i] * np.sin(np.deg2rad(WD.iloc[i]))))

    # Wind Smoothing
    if apply_wind_smooth == True:
        u = smooth(u, wind_smoothie)
        v = smooth(v, wind_smoothie)
    
    # Set up station list
    stations = list(seismic_data.keys())

    # Plot formatting
    nrows = len(stations) + 1

    # Create figure
    fig, ax = plt.subplots(nrows, 2, figsize=(6*nrows, 5))

    # Plot formatting and plot wind
    ymax = max(max(u), max(v))
    ymin = min(min(u), min(v))
    # North-South Component
    ax[0,0].set_ylim(ymin-(0.1*abs(ymin)), ymax+(0.1*abs(ymax)))
    ax[0,0].plot(df_slice['datetime'], u)
    ax[0,0].set_title('North-South Wind Speed')
    ax[0,0].set_xlabel(r'Time')
    # East-West Component
    ax[0,1].set_ylim(ymin-(0.1*abs(ymin)), ymax+(0.1*abs(ymax)))
    ax[0,1].plot(df_slice['datetime'], v)
    ax[0,1].set_title('East-West Wind Speed')
    ax[0,1].set_xlabel(r'Time')
    
    # Processing waveform
    # Create storage for seismic y data
    seismic_ymax = []
    seismic_ymin = []
    seismic_processed = {}
    # Isolate North-South and East-West seismic data
    for station in stations:
        # Data
        EW = seismic_data[station][0]
        NS = seismic_data[station][1]
        # Rectify
        if rectify_seis == True:
            EW = np.abs(EW)
            NS = np.abs(NS)
        # Seismic Smoothing
        if apply_seismic_smooth == True:
            EW = smooth(EW, seismic_smoothie)
            NS = smooth(NS, seismic_smoothie)
        # Add to storage
        seismic_ymax.append(max(EW.max(), NS.max()))
        seismic_ymin.append(min(EW.min(), NS.min()))
        seismic_processed[station] = (EW, NS)

    # Find maximum and minimum values 
    seismic_ymax = max(seismic_ymax)
    seismic_ymin = min(seismic_ymin)

    # Plot Seismic data
    for i, k in enumerate(stations):
        #t = seismic_data[k][2]
        EW, NS = seismic_processed[k]
        # North-South component
        ax[i+1, 0].set_ylim(seismic_ymin-(0.2*abs(seismic_ymin)), seismic_ymax+(0.2*abs(seismic_ymax)))
        ax[i+1, 0].plot(NS)
        ax[i+1, 0].set_title(f'North-South Seismic {k}')
        ax[i+1, 0].set_xlabel(r'Time')
        # East-West component
        ax[i+1, 1].set_ylim(seismic_ymin-(0.2*abs(seismic_ymin)), seismic_ymax+(0.2*abs(seismic_ymax)))
        ax[i+1, 1].plot(EW)
        ax[i+1, 1].set_title(f'East-West Seismic {k}')
        ax[i+1, 1].set_xlabel(r'Time')

    plt.tight_layout()
    plt.show()

def wind_vs_noise(seismic_data, 
                  wind_data, 
                  wind_year=None,
                  wind_month=None,
                  wind_day=None,
                  wind_hour=None,
                  apply_smooth = True,
                  smoothie = 100):
    """
    Create a wind speed vs seismic magnitude plot
    to compare seismic noise and AWS wind speed.

    Parameters:
    seismic_data (dict): 
        A dictionary containing seismic data with isolated components for each station.
    wind_data (pd.DataFrame): 
        A DataFrame containing the wind data.
    wind_year (int or tuple): 
        The year(s) for filtering the wind data. 
        None for all years, 
        A single year (2010-2025), 
        or a tuple of (start_year, end_year) for a range of years.
    wind_month (int or tuple): 
        The month(s) for filtering the wind data. 
        None for the entire year, 
        A single month (1-12), 
        or a tuple of (start_month, end_month) for a range of months.
    wind_day (int or tuple): 
        The day(s) for filtering the wind data. 
        None for the entire month,
        A single day (1-31),
        or a tuple of (start_day, end_day) for a range of days.
    apply_wind_smooth (bool): 
        True/False. Applying ObsPy smooth() function to wind data.
    wind_smoothie (int):
        Number of values to calculate moving average for wind smoothing.
    """

    # Check if time inputs are single valued or a range.
    # Year
    if isinstance(wind_year, (tuple, list)) and len(wind_year) == 2: 
        start_year, end_year = wind_year 
        df_slice = wind_data[(wind_data['datetime'].dt.year >= start_year) & 
                        (wind_data['datetime'].dt.year <= end_year) ].copy()
    elif wind_year is None or wind_year < 2010 or wind_year > 2025:
        df_slice = wind_data.copy()
    else:
        df_slice = wind_data[wind_data['datetime'].dt.year == wind_year].copy()
    # Month
    if isinstance(wind_month, (tuple, list)) and len(wind_month) == 2: 
        start_month, end_month = wind_month 
        df_slice = df_slice[(df_slice['datetime'].dt.month >= start_month) & 
                        (df_slice['datetime'].dt.month <= end_month) ].copy()
    elif isinstance(wind_month, int): 
        df_slice = df_slice[df_slice['datetime'].dt.month == wind_month].copy()
    # Day
    if isinstance(wind_day, (tuple, list)) and len(wind_day) == 2:
        start_day, end_day = wind_day
        df_slice = df_slice[(df_slice['datetime'].dt.day >= start_day) & 
                            (df_slice['datetime'].dt.day <= end_day)].copy()                    
    elif isinstance(wind_day, int):
        df_slice = df_slice[df_slice['datetime'].dt.day == wind_day].copy()
    # Hour
    if isinstance(wind_hour, (tuple, list)) and len(wind_hour) == 2:
        start_hour, end_hour = wind_hour
        df_slice = df_slice[(df_slice['datetime'].dt.hour >= start_hour) & 
                            (df_slice['datetime'].dt.hour <= end_hour)].copy()
    elif isinstance(wind_hour, int):
        df_slice = df_slice[df_slice['datetime'].dt.hour == wind_hour].copy()

    # Check if there is any data.
    if df_slice.empty: 
        print("No wind data available for the selected time period.") 
        return

    # Convert to numeric
    df_slice['Wind speed in km/h'] = pd.to_numeric(df_slice['Wind speed in km/h'], 
                                                    errors='coerce')
    df_slice['Wind direction in degrees true'] = pd.to_numeric(df_slice['Wind direction in degrees true'],
                                                               errors='coerce')
    df_slice['Wind direction in degrees true'] %= 360

    # Create variables
    wind_speed = df_slice['Wind speed in km/h']
    WS = abs(wind_speed)

    # Set up station list
    stations = list(seismic_data.keys())
    #nrows = len(stations) + 1

    # Plot seismic magnitude vs wind speed
    # Loop through all stations
    for i, k in enumerate(stations):
        # Extract NS and EW seismic components
        EW = seismic_data[k][0]
        NS = seismic_data[k][1]
        # Find the overall magnitude. Pythagorean theorem for isolated components
        H = np.sqrt(NS**2 + EW**2)
        # Seismic smoothing
        if apply_smooth == True:
            H = smooth(H, smoothie)
        # Interpolate wind speed data to match the length of the seismic data
        WS_interp = np.interp(np.arange(len(H)), np.linspace(0, len(H), len(WS)),WS)
        # Create Figure and format
        plt.figure(figsize=(15,6)) 
        plt.title('Wind Speed vs Seismic Noise')
        plt.ylim(0,0.1)
        plt.ylabel('Seismic Magnitude')
        plt.xlabel('Wind Speed Magnitude')

        # Plot
        plt.scatter(WS_interp, H)

def seismic_energy(seismic_data, 
                   wind_data, 
                   wind_year=None,
                   wind_month=None,
                   wind_day=None,
                   wind_hour=None,
                   energy_plot = False,
                   wind_vs_energy_plot = False):

    # Check if time inputs are single valued or a range.
    # Year
    if isinstance(wind_year, (tuple, list)) and len(wind_year) == 2: 
        start_year, end_year = wind_year 
        df_slice = wind_data[(wind_data['datetime'].dt.year >= start_year) & 
                        (wind_data['datetime'].dt.year <= end_year) ].copy()
    elif wind_year is None or wind_year < 2010 or wind_year > 2025:
        df_slice = wind_data.copy()
    else:
        df_slice = wind_data[wind_data['datetime'].dt.year == wind_year].copy()
    # Month
    if isinstance(wind_month, (tuple, list)) and len(wind_month) == 2: 
        start_month, end_month = wind_month 
        df_slice = df_slice[(df_slice['datetime'].dt.month >= start_month) & 
                        (df_slice['datetime'].dt.month <= end_month) ].copy()
    elif isinstance(wind_month, int): 
        df_slice = df_slice[df_slice['datetime'].dt.month == wind_month].copy()
    # Day
    if isinstance(wind_day, (tuple, list)) and len(wind_day) == 2:
        start_day, end_day = wind_day
        df_slice = df_slice[(df_slice['datetime'].dt.day >= start_day) & 
                            (df_slice['datetime'].dt.day <= end_day)].copy()                    
    elif isinstance(wind_day, int):
        df_slice = df_slice[df_slice['datetime'].dt.day == wind_day].copy()
    # Hour
    if isinstance(wind_hour, (tuple, list)) and len(wind_hour) == 2:
        start_hour, end_hour = wind_hour
        df_slice = df_slice[(df_slice['datetime'].dt.hour >= start_hour) & 
                            (df_slice['datetime'].dt.hour <= end_hour)].copy()
    elif isinstance(wind_hour, int):
        df_slice = df_slice[df_slice['datetime'].dt.hour == wind_hour].copy()

    # Check if there is any data.
    if df_slice.empty: 
        print("No wind data available for the selected time period.") 
        return
    
    aws_times = pd.to_datetime(df_slice["datetime"]).sort_values().reset_index(drop=True)
    aws_times = [UTC(ts) for ts in aws_times]
    
    station_list = list(seismic_data.keys())

    energy = {}

    for station in station_list:
        EW, NS, fs, t_start = seismic_data[station]
        
        energies = []

        for i in range(1, len(aws_times)):

            t0 = UTC(aws_times[i-1])
            t1 = UTC(aws_times[i])

            i0 = int((t0 - t_start) * fs)
            i1 = int((t1 - t_start) * fs)

            i0 = max(i0, 0)
            i1 = min(i1, len(EW))

            if i1 <= i0:
                energies.append(np.nan)
                continue

            seg_EW = EW[i0:i1]
            seg_NS = NS[i0:i1]

            if np.any(~np.isfinite(seg_EW)) or np.any(~np.isfinite(seg_NS)):
                energies.append(np.nan)
                continue

            energy_sum = np.sum(seg_EW**2 + seg_NS**2)

            energies.append(energy_sum)

        energy[station] = np.array(energies)

    times = aws_times[1:]
    times_dt = [t.datetime for t in times]

    if energy_plot == True:
        for station, energies in energy.items():
            plt.plot(times_dt, energies, label=station, alpha=0.8)
        plt.title("Seismic Energy Over Time")
        plt.xlabel("Time")
        plt.ylabel(r"Energy ($EW^{2} + NS^{2}$)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    if wind_vs_energy_plot == True:
            
        # Convert to numeric
        df_slice['Wind speed in km/h'] = pd.to_numeric(df_slice['Wind speed in km/h'], 
                                                        errors='coerce')
        df_slice['Wind direction in degrees true'] = pd.to_numeric(df_slice['Wind direction in degrees true'],
                                                                errors='coerce')
        df_slice['Wind direction in degrees true'] %= 360

        # Create variables
        wind_speed = df_slice['Wind speed in km/h']
        WS = abs(wind_speed)
        
        # Plot
        fig, ax1 = plt.subplots(figsize=(12,6))


        # plot AWS
        ax1.plot(df_slice["datetime"], WS,
                "k-", linewidth=0.8, label="Wind Speed")
        ax1.set_ylabel(r"Wind Speed ($km/hr$)", color="k")
        ax1.tick_params(axis="y", labelcolor="k")
        ax1.set_ylim(bottom=0)

        # Plot Seismic

        ax2 = ax1.twinx()
        for station, energies in energy.items():
            ax2.plot(times_dt, energies, label=station, alpha=0.8)
        plt.title("Seismic Energy Over Time")
        plt.xlabel("Time")
        plt.ylabel(r"Energy ($EW^{2} + NS^{2}$)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        plt.tight_layout()
        plt.show()


    return times, energy    

def wind_seis_energy_scatter(seismic_energy,
                             wind_data,
                             wind_thresh,
                             wind_year = None,
                             wind_month = None,
                             wind_day = None,
                             wind_hour = None):
    
    # Check if time inputs are single valued or a range.
    # Year
    if isinstance(wind_year, (tuple, list)) and len(wind_year) == 2: 
        start_year, end_year = wind_year 
        df_slice = wind_data[(wind_data['datetime'].dt.year >= start_year) & 
                        (wind_data['datetime'].dt.year <= end_year) ].copy()
    elif wind_year is None or wind_year < 2010 or wind_year > 2025:
        df_slice = wind_data.copy()
    else:
        df_slice = wind_data[wind_data['datetime'].dt.year == wind_year].copy()
    # Month
    if isinstance(wind_month, (tuple, list)) and len(wind_month) == 2: 
        start_month, end_month = wind_month 
        df_slice = df_slice[(df_slice['datetime'].dt.month >= start_month) & 
                        (df_slice['datetime'].dt.month <= end_month) ].copy()
    elif isinstance(wind_month, int): 
        df_slice = df_slice[df_slice['datetime'].dt.month == wind_month].copy()
    # Day
    if isinstance(wind_day, (tuple, list)) and len(wind_day) == 2:
        start_day, end_day = wind_day
        df_slice = df_slice[(df_slice['datetime'].dt.day >= start_day) & 
                            (df_slice['datetime'].dt.day <= end_day)].copy()                    
    elif isinstance(wind_day, int):
        df_slice = df_slice[df_slice['datetime'].dt.day == wind_day].copy()
    # Hour
    if isinstance(wind_hour, (tuple, list)) and len(wind_hour) == 2:
        start_hour, end_hour = wind_hour
        df_slice = df_slice[(df_slice['datetime'].dt.hour >= start_hour) & 
                            (df_slice['datetime'].dt.hour <= end_hour)].copy()
    elif isinstance(wind_hour, int):
        df_slice = df_slice[df_slice['datetime'].dt.hour == wind_hour].copy()

    # Check if there is any data.
    if df_slice.empty: 
        print("No wind data available for the selected time period.") 
        return
    
     # Convert to numeric
    df_slice['Wind speed in km/h'] = pd.to_numeric(df_slice['Wind speed in km/h'], 
                                                    errors='coerce')
    df_slice['Wind direction in degrees true'] = pd.to_numeric(df_slice['Wind direction in degrees true'],
                                                            errors='coerce')
    df_slice['Wind direction in degrees true'] %= 360

    # Create variables
    wind_speed = df_slice['Wind speed in km/h']
    WS = abs(wind_speed)
    
    
    seis_time = seismic_energy[0] 
    seis_time = [t.datetime for t in seis_time]

    energy = seismic_energy[1][list(seismic_energy[1].keys())[0]]

    energy = pd.to_numeric(energy, errors="coerce")
    energy_series = pd.Series(np.where(np.isfinite(energy), 
                                       np.log(energy), 
                                       np.nan),
                              index=pd.to_datetime(seis_time))
    
    aws_series = df_slice.set_index("datetime")['Wind speed in km/h']
    combined = pd.concat([aws_series, energy_series.rename("energy")], axis=1).dropna()
    

    mask = combined['Wind speed in km/h'].values > wind_thresh

    x = combined['Wind speed in km/h'].values
    y = combined["energy"].values

    # linear regression 
    slope, intercept, r, p, se = linregress(x[mask], y[mask])
    x_fit = np.linspace(x[mask].min(), x[mask].max(), 200)
    y_fit = slope * x_fit + intercept

    r2 = r**2
    p_str = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
    fit_label = f"fit: y = {slope:.3f}x + {intercept:.2f}\nR² = {r2:.3f}, p = {p_str} for wind over {wind_thresh} (km/hr)."

    plt.scatter(x, y, c="blue", s=10, alpha=0.5, linewidths=0, label=f"n = {len(x)}")
    plt.plot(x_fit, y_fit, "k-", linewidth=1.5, label=fit_label)

    plt.xlabel("Wind Speed")
    plt.ylabel("Seismic Energy")
    plt.title("Wind Speed vs Seismic Energy")
    plt.legend()
    plt.ylim(np.percentile(y, 2.5), np.percentile(y, 100 - 2.5))
    plt.tight_layout()
    
def apply_filter(wave_dict = None, 
                 filter_type = None, 
                 freqmin=None, 
                 freqmax=None, 
                 freq=None, 
                 corners=4, 
                 zerophase=True,
                 save_mseed=False,
                 config=None,
                 read_file=True,
                 filename='default_filter'):

    """
    Apply a filter to seismic waveform data stored in a dictionary without altering the original.

    Parameters:
    wave_dict (dict):
        Dictionary containing seismic waveform data.
    filter_type (str):
        Type of filter to apply. 
        Options include 'bandpass', 'bandstop', 'lowpass', 'highpass'.
        'lowpass_cheby_2', 'lowpass_fir', 'remez_fir' currently unsupported.
    freqmin (float):
        Minimum frequency for bandpass/bandstop filters.
    freqmax (float):
        Maximum frequency for bandpass/bandstop filters.
    freq (float):
        Cutoff frequency for lowpass/highpass filters.
    corners (int):
        Number of corners for the filter.
    zerophase (bool):
        True/False. If True, apply a zero-phase filter.
    save_mseed (bool):
        True/False. True to save as mseed file.
    config (dict):
        Information from a config file containing the local "seismic_data_path".
    read_file (bool):
        True/False. True to switch on file checking.
    filename (str):
        Title of saved mseed file.

    Returns:
    filtered_dict (dict):
        Dictionary containing filtered seismic waveform data.
    """

    # Path 
    base_path = Path(config["seismic_data_path"]) if config else Path(".")
    base_path.mkdir(parents=True, exist_ok=True)
    file_path = (base_path / filename).with_suffix(".mseed")

    # Read file if it exists
    if file_path.exists():
        if read_file == True:
            print(f"Reading existing file: {file_path}")
            stream = read(str(file_path))
            filtered_dict = defaultdict(list)
            for tr in stream:
                filtered_dict[tr.stats.station].append(tr)

    else:
            
        # Setup dictionary
        filtered_dict = defaultdict(list)
        streams = []
        # Loop through and apply filter to the traces
        for station_name, traces in wave_dict.items():
            st = Stream([tr.copy() for tr in traces]) # Copy to avoid overwriting data

            if filter_type in ('bandpass', 'bandstop'):
                st.filter(type=filter_type, 
                        freqmin=freqmin, 
                        freqmax=freqmax, 
                        corners=corners, 
                        zerophase=zerophase)
                streams.append(st)

            elif filter_type in ('lowpass', 'highpass'): 
                st.filter(type=filter_type, 
                        freq=freq, 
                        corners=corners, 
                        zerophase=zerophase)
                streams.append(st)

            else:
                raise ValueError(f"Unsupported filter type: {filter_type}") #'lowpass_cheby_2', 'lowpass_fir', 'remez_fir' currently not setup

            filtered_dict[station_name] = st.traces # Write to a dictionary
        
        # Save as mseed file
        if save_mseed == True:
            merged_stream = streams[0].copy() # Copy to avoid overwriting data
            for st in streams[1:]:
                merged_stream += st
            merged_stream.merge()

            merged_stream.write(f'{str(file_path)}', format="MSEED")
            print(f'Saved as {file_path}')

    return filtered_dict

def select_time(wave_dict=None, 
                t_start=None, 
                duration=None,
                save_mseed=False,
                config=None,
                read_file=True,
                filename='default_trim'):
    
    """
    Select a specific time window from seismic waveform data stored in a dictionary without altering the original.

    Parameters:
    wave_dict (dict):
        Dictionary containing seismic waveform data.
    t_start (UTCDateTime):
        Start time for the time window.
    duration (float):
        Duration of the time window in seconds.
    save_mseed (bool):
        True/False. True to save as mseed file.
    config (dict):
        Information from a config file containing the local "seismic_data_path".
    read_file (bool):
        True/False. True to switch on file checking.
    filename (str):
        Title of saved mseed file.
    Returns:
    new_dict (dict):
        Dictionary containing selected seismic waveform data.
    """

    # Path 
    base_path = Path(config["seismic_data_path"]) if config else Path(".")
    base_path.mkdir(parents=True, exist_ok=True)
    file_path = (base_path / filename).with_suffix(".mseed")

    # Read file if it exists
    if file_path.exists():
        if read_file == True:
            print(f"Reading existing file: {file_path}")
            stream = read(str(file_path))
            new_dict = defaultdict(list)
            for tr in stream:
                new_dict[tr.stats.station].append(tr)
            
            return new_dict
        
    else:
        # Establish timespan
        t_end = t_start + duration
        streams = []
        # Trim to desired timespan and write to a direction
        new_dict = defaultdict(list)
        for station_name in wave_dict:
            st = Stream(wave_dict[station_name]).copy() # Copy to avoid overwriting data
            st.trim(starttime=t_start, endtime=t_end, pad=False)
            streams.append(st)
            new_dict[station_name].extend(st.traces)

        # Save as mseed file
        if save_mseed == True:
            merged_stream = streams[0].copy() # Copy to avoid overwriting data
            for st in streams[1:]:
                merged_stream += st
            merged_stream.merge()

            merged_stream.write(f'{str(file_path)}', format="MSEED")
            print(f'Saved as {file_path}')

    return new_dict


def signal_to_noise(wave_dict, 
                    filtered_dict,
                    NS_channel,
                    EW_channel,
                    Z_channel):
    """
    Calculates the noise to signal ratio for the 
    Z, NS and EW components of seismic data and 
    stores the results as a dictionary.
    Parameters:
    wave_dict (dict):
        A wave dictionary containing seismic waveform data.
    filtered_dict (dict):
        A dictionary containing filtered seismic waveform data.
    NS_channel (list of str):
            List of channel codes for the NS component.
    EW_channel (list of str):
        List of channel codes for the EW component.
    Z_channel (list of str):
        List of channel codes for the Z component.
    """
    
    # Set up seismic waveform data from wave_dict and the filtered data from filtered_dict 
    # and calculate the noise to signal ratio.
    # Setup Storage
    ratio_dict = {}
    # Loop through wave_dict, find the channels, and store the data in a new dictionary
    for i, (station, stream) in enumerate(wave_dict.items(), start=2):
        st = Stream(stream)
        st.sort(['channel'])
        NS = np.array(find_channel(st, NS_channel)) # Try to find NS channel from function input
        EW = np.array(find_channel(st, EW_channel)) # Try to find EW channel from function input
        Z = np.array(find_channel(st, Z_channel))

        # Warnings for missing channels.
        if Z is None:
            print(f"Warning: Missing Z channel for {station}. Skipping."
                    "This may cause issues if the Z channel is not missing in wave_dict.")
        if NS is None:
            print(f"Warning: Missing NS channel for {station}. Skipping."
                    "This may cause issues if the NS channel is not missing in wave_dict.")
        if EW is None:
            print(f"Warning: Missing EW channel for {station}. Skipping."
                    "This may cause issues if the EW channel is not missing in wave_dict.")

        # Filtered Waveform Data
        for i, (station, stream) in enumerate(filtered_dict.items(), start=2):
            st = Stream(stream)
            st.sort(['channel'])
            NS_filt = np.array(find_channel(st, NS_channel)) # Try to find NS channel from function input
            EW_filt = np.array(find_channel(st, EW_channel)) # Try to find EW channel from function input
            Z_filt = np.array(find_channel(st, Z_channel))

                # Warnings for missing channels.
            if Z_filt is None:
                print(f"Warning: Missing Z channel for filtered {station}. Skipping."
                        "This may cause issues if the Z channel is not missing in filtered_dict.")
            if NS_filt is None:
                print(f"Warning: Missing NS channel for filtered {station}. Skipping."
                        "This may cause issues if the NS channel is not missing in filtered_dict.")
            if EW_filt is None:
                print(f"Warning: Missing EW channel for filtered {station}. Skipping."
                        "This may cause issues if the EW channel is not missing in filtered_dict.")


            # Calculate noise to signal ratio for each station and component, and store in a new dictionary.
            if Z is not None and Z_filt is not None:
                noise_Z = Z - Z_filt
                signal_P_Z = np.mean(Z_filt ** 2)
                noise_P_Z = np.mean(noise_Z ** 2)
                ratio_Z = 10 * np.log10(signal_P_Z / noise_P_Z)
            else:
                ratio_Z = None
                print(f"No Data for Z component in {station}. Skipping.")
            # NS component
            if NS is not None and NS_filt is not None:
                noise_NS = NS - NS_filt
                signal_P_NS = np.mean(NS_filt ** 2)
                noise_P_NS = np.mean(noise_NS ** 2)
                ratio_NS = 10 * np.log10(signal_P_NS / noise_P_NS)
            else:
                ratio_NS = None
                print(f"No Data for NS component in {station}. Skipping.")
            # EW component
            if EW is not None and EW_filt is not None:
                noise_EW = EW - EW_filt
                signal_P_EW = np.mean(EW_filt ** 2)
                noise_P_EW = np.mean(noise_EW ** 2)
                ratio_EW = 10 * np.log10(signal_P_EW / noise_P_EW)
            else:
                ratio_EW = None
                print(f"No Data for EW component in {station}. Skipping.")
            # Store the ratios in a new dictionary
            ratio_dict[station] = {"Z": float(ratio_Z) if ratio_Z is not None else None,
                                "NS": float(ratio_NS) if ratio_NS is not None else None,
                                "EW": float(ratio_EW) if ratio_EW is not None else None}

    return ratio_dict

def montecarlo_optimal_snr(wind_speed,
                           wave_dict,
                           freq_range,
                           n_iterations,
                           NS_channel,
                           EW_channel,
                           Z_channel):
    """
    Compute a montecarlo test optimising the bandpass filter
    to find the best correlation between Wind Speed and Seismic Signal.

    Parameters:
        wind_speed (array):
            An array of wind speed data in format 
            wind_speed[0] (time), wind_speed[1] (speed array).
        wave_dict (dict):
             A wave dictionary containing seismic waveform data.
        freq_range (tuple):
            Two values indicating the minimum and maximum frequency (Hz) to test.
            e.g (5, 45).
        n_iterations (int):
            Number of trials for the montecarlo test.
        NS_channel (list of str):
            List of channel codes for the NS component.
        EW_channel (list of str):
            List of channel codes for the EW component.
        Z_channel (list of str):
            List of channel codes for the Z component.

    Returns:
        The statistics for the iteration with the highest avg R squared value.
    """

    # Store results
    results = []
    station_list = list(wave_dict.keys())

    for station in station_list:
        print(f"Processing {station}...")
        station_results = []
        for i in range(n_iterations):
            # Randomly select frequency range
            fmin = random.randint(freq_range[0], freq_range[1] - 1)
            fmax = random.randint(fmin + 1, freq_range[1])
            
            # Apply bandpass filter
            CWA_test = apply_filter({station: wave_dict[station]}, 'bandpass', freqmin=fmin, freqmax=fmax)
            
            # Calculate SNR for each 30-min window
            
            snr_test = []
            aws_times = wind_speed[0]   # timestamps
            wind_speed_values = wind_speed[1]   # wind speeds

            for time in aws_times:
                t_test = UTC(str(time))
                trim_test = select_time(CWA_test, t_test - timedelta(minutes=30), 1800) # Minus 30 minutes as each AWS Timestamp is a past 30 min average 
                trim_wave_dict = select_time({station: wave_dict[station]}, t_test- timedelta(minutes=30), 1800)
                snr_test.append(signal_to_noise(trim_wave_dict, trim_test,  NS_channel, EW_channel, Z_channel))
                
            # Extract Z, NS, EW components
            Z_test = [snr_test[k][station]['Z'] for k in range(len(snr_test))]
            NS_test = [snr_test[k][station]['NS'] for k in range(len(snr_test))]
            EW_test = [snr_test[k][station]['EW'] for k in range(len(snr_test))]
            
            
            # Calculate R² for each component
            windarray = np.array(wind_speed_values.reshape(-1, 1))
            
            Z_model = LinearRegression().fit(windarray, Z_test)
            Z_r_sq = Z_model.score(windarray, Z_test)
            
            NS_model = LinearRegression().fit(windarray, NS_test)
            NS_r_sq = NS_model.score(windarray, NS_test)
            
            EW_model = LinearRegression().fit(windarray, EW_test)
            EW_r_sq = EW_model.score(windarray, EW_test)
            
            # Store results
            station_results.append({
                'fmin': fmin,
                'fmax': fmax,
                'Z_r2': Z_r_sq,
                'NS_r2': NS_r_sq,
                'EW_r2': EW_r_sq,
                'avg_r2': (Z_r_sq + NS_r_sq + EW_r_sq) / 3,
                'Z': Z_test,
                'NS': NS_test,
                'EW': EW_test,
                'Z_model': Z_model,
                'NS_model': NS_model,
                'EW_model': EW_model,
                'windarray': windarray
            })
        
            print(f"Completed {i+1}/{n_iterations} iterations")

        # Find best result
        best_result = max(station_results, key=lambda x: x['avg_r2'])
        print(f"\nBest frequency range for {station}: {best_result['fmin']} - {best_result['fmax']} Hz")
        print(f"Z R²: {best_result['Z_r2']:.4f}")
        print(f"NS R²: {best_result['NS_r2']:.4f}")
        print(f"EW R²: {best_result['EW_r2']:.4f}")
        print(f"Average R²: {best_result['avg_r2']:.4f}")
        results.append(station_results)

    return best_result, results

def find_channel(stream, options):
    """
    Find appropriate NS and EW Channels from a chosen stream.
    Copied from align.py.
    
    Parameters:
    stream (obspy.core.stream.Stream):
        An ObsPy stream object.
    options (list of str):
        A list of channel codes.
    """
    # Loop through streams and find the associated channel code
    traces = []
    for ch in options:
        traces.extend(stream.select(channel=ch))
        if len(traces) > 0:
            return traces # Return first channel code
        
    # If none are found
    return None 

def plot_statistics(monte_result,
                    ylabel = 'Power'):

    """
    Using the best result from seis_aws_fft_cor(),
    plot wind speed against each channel's power and fit
    the trendline.

    Parameters:
        monte_result (dict):
            The output of seis_aws_fft_cor().
        ylabel (str):
            Y label and adjusts suptitle. 
            Change to power if using power_wind_monte() output.
    """
    
    # Gather variables
    NS_model = monte_result['NS_model']
    EW_model = monte_result['EW_model']
    Z_model = monte_result['Z_model']

    windarray = monte_result['aws_array']
    WS = np.array(monte_result['aws_array'].reshape(-1))
    NS = monte_result['NS']
    EW = monte_result['EW']
    Z = monte_result['Z']
    fmin = monte_result['fmin']
    fmax = monte_result['fmax']

    # Sort variables for simple plotting
    idx = np.argsort(WS)

    WS_sorted = WS[idx]
    EW_pred = EW_model.predict(windarray)[idx]
    NS_pred = NS_model.predict(windarray)[idx]
    Z_pred = Z_model.predict(windarray)[idx]

    # Plot
    fig, ax = plt.subplots(3,1,figsize = (10,8))

    # EW
    ax[0].scatter(WS, EW)
    ax[0].plot(WS_sorted, EW_pred, 'r')
    ax[0].set_title(f"EW (R² = {monte_result['EW_r2']:.3f})")

    # NS
    ax[1].scatter(WS, NS)
    ax[1].plot(WS_sorted, NS_pred, 'r')
    ax[1].set_title(f"NS (R² = {monte_result['NS_r2']:.3f})")

    # Z
    ax[2].scatter(WS, Z)
    ax[2].plot(WS_sorted, Z_pred, 'r')
    ax[2].set_title(f"Z (R² = {monte_result['Z_r2']:.3f})")

    # Plot Formatting
    plt.xlabel('Wind Speed (km/hr)')
    for a in ax:
        a.set_ylabel(ylabel + ' $m^2s^{-2}$')
    title = 'Wind Speed vs' + ' ' + ylabel +f' for {fmin}-{fmax}Hz Band'
    plt.suptitle(title, fontsize = 20)
    plt.tight_layout()
    
def power_wind_freq_filter(wind_speed,
                     wave_dict = None,
                     use_file = False,
                     fmin = 1,
                     fmax = 49,
                     seismic_mseed_name=None,
                     config=None,
                     csv = False,
                     csv_title = 'monte_results'):

    # Old method
    # Very Computationally heavy
    if use_file == True:
        # Path 
        base_path = Path(config["seismic_data_path"]) if config else Path(".")
        base_path.mkdir(parents=True, exist_ok=True)
        file_path = (base_path / seismic_mseed_name).with_suffix(".mseed")

        # Read file if it exists
        if file_path.exists():
            print(f"Reading existing file: {file_path}")
            stream = read(str(file_path))
            wave_dict = defaultdict(list)
            for tr in stream:
                wave_dict[tr.stats.station].append(tr)
        else:
            print("No File Found")

    else:
        if wave_dict == None:
            print('wave_dict not selected. Please input a file or a dictionary. (default: use_file = False )')
            return None
    
    station_list = list(wave_dict.keys())
    aws_times = wind_speed[0]   # timestamps
    wind_speed_values = wind_speed[1]   # wind speeds
    results = []
    all_results = []
    best_results = []

    for station in station_list:
        
        station_results = []

        # Or spectrogram?
        # fft?
        for f1 in range(fmin, fmax):
                for f2 in range(f1+1, fmax+1):
                    if f1<f2:
                        # Window Function dependent on fmin?
                        filt = apply_filter({station: wave_dict[station]}, 
                                            filter_type = 'bandpass',
                                            freqmin= f1,
                                            freqmax =f2)
                        
                        Z_power = []
                        NS_power = []
                        EW_power = []

                        # Take outside frequency loop / remove loop - time index array ?
                        for time in aws_times:
                            t_test = UTC(str(time))
                            trim_wave_dict = select_time(filt, 
                                                        t_test- timedelta(minutes=30), 
                                                        1800) # Minus 30 minutes as each AWS Timestamp is a past 30 min average 
                            st = Stream(trim_wave_dict[station])

                            # Puts in alphabetical order. 
                            # E, N, Z
                            st.sort(['channel'])   

                            EW = st.select(channel="*E")[0].data
                            NS = st.select(channel="*N")[0].data
                            Z  = st.select(channel="*Z")[0].data

                            EW_power.append(np.mean(EW**2))
                            NS_power.append(np.mean(NS**2))
                            Z_power.append(np.mean(Z**2))
                            
                        WS = np.array(wind_speed_values)
                        Z_power = np.array(Z_power)
                        NS_power = np.array(NS_power)
                        EW_power = np.array(EW_power)

                        # Remove outliers
                        data = np.column_stack([WS, Z_power, NS_power, EW_power])

                        mean = np.nanmean(data, axis=0)
                        std = np.nanstd(data, axis=0)

                        # Avoid divide-by-zero
                        std[std == 0] = 1

                        z_scores = (data - mean) / std
                        mask = np.all(np.abs(z_scores) < 3, axis=1)

                        WS_filt = WS[mask]
                        Z_filt = Z_power[mask]
                        NS_filt = NS_power[mask]
                        EW_filt = EW_power[mask]
                        
                        windarray = np.array(WS_filt).reshape(-1, 1)
                        
                        # Calculate R² for each component
                        Z_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(windarray, Z_filt)
                        Z_r_sq = Z_model.score(windarray, Z_filt)
                        
                        NS_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(windarray, NS_filt)
                        NS_r_sq = NS_model.score(windarray, NS_filt)
                        
                        EW_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(windarray, EW_filt)
                        EW_r_sq = EW_model.score(windarray, EW_filt)

                        # Store results
                        station_results.append({
                            'fmin': f1,
                            'fmax': f2,
                            'Z_r2': Z_r_sq,
                            'NS_r2': NS_r_sq,
                            'EW_r2': EW_r_sq,
                            'avg_r2': (Z_r_sq + NS_r_sq + EW_r_sq) / 3,
                            'Z': Z_filt,
                            'NS': NS_filt,
                            'EW': EW_filt,
                            'Z_model': Z_model,
                            'NS_model': NS_model,
                            'EW_model': EW_model,
                            'windarray': windarray
                        })

                        print(f"Completed Iteration {f1}Hz-{f2}Hz. AVG R Sqaured = {(Z_r_sq + NS_r_sq + EW_r_sq) / 3}")
        # Find best result
        best_result = max(station_results, key=lambda x: x['avg_r2'])
        best_results.append({'station': station, **best_result})
        print(f"\nBest frequency range for {station}: {best_result['fmin']} - {best_result['fmax']} Hz")
        print(f"Z R²: {best_result['Z_r2']:.4f}")
        print(f"NS R²: {best_result['NS_r2']:.4f}")
        print(f"EW R²: {best_result['EW_r2']:.4f}")
        print(f"Average R²: {best_result['avg_r2']:.4f}")
        results.append(station_results)

        if csv == True:
            for result in station_results:
                all_results.append({
                    'station': station,
                    'fmin': result['fmin'],
                    'fmax': result['fmax'],
                    'Z_r2': result['Z_r2'],
                    'NS_r2': result['NS_r2'],
                    'EW_r2': result['EW_r2'],
                    'avg_r2': result['avg_r2']
                    })

    if csv == True:
        df = pd.DataFrame(all_results)
        csv_name = f"{csv_title}.csv"
        df.to_csv(csv_name, index=False)
        print(f"Results saved to {csv_name}")


    return best_results, results

def load_seis_data(config = None, filename = None, path = None):

    if config:
        # Path 
        base_path = Path(config["seismic_data_path"]) if config else Path(".")
        base_path.mkdir(parents=True, exist_ok=True)
        if filename:
            file_path = (base_path / filename).with_suffix(".mseed")
        else:
            print("Please provide a filename that exists in the seismic data path.")
            return None
    elif path:
            file_path = Path(path)
    else:
        print("Please provide either a filename in the config seismic data path or a path to the seismic data.")
        return None

    print(f"Reading existing file: {file_path}")
    stream = read(str(file_path))
    new_dict = defaultdict(list)
    for tr in stream:
        new_dict[tr.stats.station].append(tr)
    return new_dict

def power_wind_fft(wind_speed,
                     wave_dict = None,
                     fmin = 1,
                     fmax = 49,
                     config=None,
                     use_file = False,
                     seismic_mseed_name=None,
                     csv = False,
                     csv_title = 'results'):
    
    """
    Finds the correlation between seismic data and AWS wind speed.
    Seismic data is trimmed to 30 minute segments which match AWS dataset.
    The real fast fourier transform is compute for each seismic component and 
    the power is calculated for each time step. For each frequency band the power
    is calculated and correlated with AWS data through the mutual time series.

    Parameters:
        wind_speed (array):
            An array of wind speed data in format 
            wind_speed[0] (time), wind_speed[1] (speed array).
        wave_dict (dict):
             A wave dictionary containing seismic waveform data.
        fmin (int):
            Minimum frequency value for calculating bandwidths.
        fmax (int):
            Maximum frequency value for calculating bandwidths.
        config (dict):
            Information from a config file containing the local "seismic_data_path".
        use_file (bool):
            True/False. True to switch on file checking for mseed file.
        seismic_mseed_file (str):
            Title of saved mseed file.
        csv (bool):
            True/False. True to save returns as a csv file.
        csv_title (str):
            Title of csv file.
    
    Returns:
        best_results (dict):
            A dictionary containing information about the best correlation result for each station.
        results (dict):
            A dictionary containing information about the all the correlaation result for each station.
    """

    # File and Storage
    if use_file == True:
        # Path 
        base_path = Path(config["seismic_data_path"]) if config else Path(".")
        base_path.mkdir(parents=True, exist_ok=True)
        file_path = (base_path / seismic_mseed_name).with_suffix(".mseed")

        # Read file if it exists
        if file_path.exists():
            print(f"Reading existing file: {file_path}")
            stream = read(str(file_path))
            wave_dict = defaultdict(list)
            for tr in stream:
                wave_dict[tr.stats.station].append(tr)
        else:
            print("No File Found")

    else:
        if wave_dict == None:
            print('wave_dict not selected. Please input a file or a dictionary. (default: use_file = False )')
            return None
    
    # Setup 
    station_list = list(wave_dict.keys())
    aws_times = wind_speed[0]   # timestamps
    wind_speed_values = wind_speed[1]   # wind speeds
    WS = np.array(wind_speed_values)
    results = []
    all_results = []
    best_results = []

    # Loop through all stations
    for station in station_list:
        
        # Final Results for each station loop  
        station_results = []
        # EW, NS, Z components following real fast fourier transform (RFFT)
        spectra = []

        # Loop through all time periods
        for time in aws_times:

            # Matchup AWS & Seismic series. Important for missed AWS measurements.
            t_test = UTC(str(time))
            trim_wave_dict = select_time({station: wave_dict[station]}, 
                                        t_test - timedelta(minutes=30), 
                                        1800) # Minus 30 minutes as each AWS Timestamp is a past 30 min average 
            # Organise Streams
            st = Stream(trim_wave_dict[station])

            # Puts in alphabetical order. 
            # E, N, Z
            st.sort(['channel'])   

            EW = st.select(channel="*E")[0].data
            NS = st.select(channel="*N")[0].data
            Z  = st.select(channel="*Z")[0].data

            # Compute rfft for each component
            y_EW = rfft(EW)
            y_NS = rfft(NS)
            y_Z = rfft(Z)

            # EW, NS, Z all same length and freq
            fs = st[0].stats.sampling_rate
            # Compute rfft frequency
            freq = rfftfreq(len(EW), 1/fs)

            # Create a mask to remove negative values
            mask = freq >= 0
            freq = freq[mask]

            # Calculate power for each component
            EW_p = np.abs(y_EW[mask])**2
            NS_p = np.abs(y_NS[mask])**2
            Z_p = np.abs(y_Z[mask])**2

            # Save spectra as a dictionary
            spectra.append({'freq': freq,
                            'EW': EW_p,
                            'NS': NS_p,
                            'Z': Z_p,
                            'len': len(EW)})
            
        # Loop through frequency bands for each time index
        for f1 in range(fmin, fmax):
                    for f2 in range(f1+1, fmax+1):
                        if f1<f2:

                            # Setup power 
                            EW_power = []
                            NS_power = []
                            Z_power = []

                            # Loop through spectra at each time index
                            for spec in spectra:
                                
                                # Create bandwidth (f1,f2)
                                bandwidth = ((spec['freq'] >= f1) & (spec['freq'] < f2))

                                # Calculate power of the bandwidth. Used typical method for fft bands. 
                                EW_power.append(2*np.sum(spec['EW'][bandwidth]) / spec['len'])
                                NS_power.append(2*np.sum(spec['NS'][bandwidth]) / spec['len'])
                                Z_power.append(2*np.sum(spec['Z'][bandwidth]) / spec['len'])
                            
                            # Convert to arrays
                            EW_power = np.array(EW_power)
                            NS_power = np.array(NS_power)
                            Z_power = np.array(Z_power)

                            # Remove outliers setup
                            data = np.column_stack([WS, Z_power, NS_power, EW_power])
                            mean = np.nanmean(data, axis=0)
                            std = np.nanstd(data, axis=0)

                            # Avoid dividing by zero
                            std[std == 0] = 1

                            # Calculate z scores
                            z_scores = (data - mean) / std

                            # Remove outliers
                            mask = np.all(np.abs(z_scores) < 3, axis=1)
                            WS_filt = WS[mask]
                            Z_filt = Z_power[mask]
                            NS_filt = NS_power[mask]
                            EW_filt = EW_power[mask]
                            
                            # Reshape for R² analysis
                            windarray = np.array(WS_filt).reshape(-1, 1)
                            
                            # Calculate R² for each component
                            Z_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(windarray, Z_filt)
                            Z_r_sq = Z_model.score(windarray, Z_filt)
                            
                            NS_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(windarray, NS_filt)
                            NS_r_sq = NS_model.score(windarray, NS_filt)
                            
                            EW_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(windarray, EW_filt)
                            EW_r_sq = EW_model.score(windarray, EW_filt)

                            # Store results
                            station_results.append({
                                'fmin': f1,
                                'fmax': f2,
                                'Z_r2': Z_r_sq,
                                'NS_r2': NS_r_sq,
                                'EW_r2': EW_r_sq,
                                'avg_r2': (Z_r_sq + NS_r_sq + EW_r_sq) / 3,
                                'Z': Z_filt,
                                'NS': NS_filt,
                                'EW': EW_filt,
                                'Z_model': Z_model,
                                'NS_model': NS_model,
                                'EW_model': EW_model,
                                'windarray': windarray
                            })

                            print(f"Completed Iteration {f1}Hz-{f2}Hz. AVG R Sqaured = {(Z_r_sq + NS_r_sq + EW_r_sq) / 3}")
        
        # Find best result
        best_result = max(station_results, key=lambda x: x['avg_r2'])
        best_results.append({'station': station, **best_result})

        # Print best result
        print(f"\nBest frequency range for {station}: {best_result['fmin']} - {best_result['fmax']} Hz")
        print(f"Z R²: {best_result['Z_r2']:.4f}")
        print(f"NS R²: {best_result['NS_r2']:.4f}")
        print(f"EW R²: {best_result['EW_r2']:.4f}")
        print(f"Average R²: {best_result['avg_r2']:.4f}")
        results.append(station_results)

        # Save to csv file setup
        if csv == True:
            for result in station_results:
                all_results.append({
                    'station': station,
                    'fmin': result['fmin'],
                    'fmax': result['fmax'],
                    'Z_r2': result['Z_r2'],
                    'NS_r2': result['NS_r2'],
                    'EW_r2': result['EW_r2'],
                    'avg_r2': result['avg_r2']
                    })
                
    # Save all results to csv file
    if csv == True:
        df = pd.DataFrame(all_results)
        csv_name = f"{csv_title}.csv"
        df.to_csv(csv_name, index=False)
        print(f"Results saved to {csv_name}")


    return best_results, results

def AWS_variable(data, 
                variable='Wind speed in km/h',
                year=None, 
                month=None, 
                day=None,
                hour=None,
                plot = True,
                apply_smooth = False,
                smoothie = 3):

    """
    Plots the a chosen variable for a given year and month from the provided DataFrame.

    Parameters:
    data (pd.DataFrame): 
        The DataFrame containing the weather station data.
    variable (str): 
        The variable to plot. Must be a column heading in the DataFrame.
    year (int): 
        The year for plotting (2010-2025 or None for all years).
    month (int or tuple): 
        The month(s) for plotting.
        None for the entire year,
        A single month (1-12),
        or a tuple of (start_month, end_month) for a range of months.
    day (int or tuple): 
        The day(s) for plotting.
        None for the entire month,
        A single day (1-31),
        or a tuple of (start_day, end_day) for a range of days.
    hour (int or tuple):
        The hour(s) for plotting.
        None for entire day,
        A single hour (0-23),
        or a tuple of (start_hour, end_hour) for a range of hours.
    apply_smooth (bool): 
        True/False. Applying ObsPy smooth() function.
    smoothie (int):
        Number of values to calculate moving average for smoothing.
    """
    
    # Check if time inputs are single valued or a range.
    # Year
    if isinstance(year, (tuple, list)) and len(year) == 2: 
        start_year, end_year = year 
        df_slice = data[(data['datetime'].dt.year >= start_year) & 
                        (data['datetime'].dt.year <= end_year) ].copy()
    elif year is None or year < 2010 or year > 2025:
        df_slice = data.copy()
    else:
        df_slice = data[data['datetime'].dt.year == year].copy()
    # Month
    if isinstance(month, (tuple, list)) and len(month) == 2: 
        start_month, end_month = month 
        df_slice = df_slice[(df_slice['datetime'].dt.month >= start_month) & 
                        (df_slice['datetime'].dt.month <= end_month) ].copy()
    elif isinstance(month, int): 
        df_slice = df_slice[df_slice['datetime'].dt.month == month].copy()
    # Day
    if isinstance(day, (tuple, list)) and len(day) == 2:
        start_day, end_day = day
        df_slice = df_slice[(df_slice['datetime'].dt.day >= start_day) & 
                            (df_slice['datetime'].dt.day <= end_day)].copy()
    elif isinstance(day, int):
        df_slice = df_slice[df_slice['datetime'].dt.day == day].copy()
    # Hour
    if isinstance(hour, (tuple, list)) and len(hour) == 2:
        start_hour, end_hour = hour
        df_slice = df_slice[(df_slice['datetime'].dt.hour >= start_hour) & 
                            (df_slice['datetime'].dt.hour <= end_hour)].copy()
    elif isinstance(hour, int):
        df_slice = df_slice[df_slice['datetime'].dt.hour == hour].copy()
    
    # Check if there is data
    if df_slice.empty: 
        print("No data available for the selected time period.") 
        return

    # Check if variable exists in the DataFrame
    if variable not in df_slice.columns:
        print(f"Variable '{variable}' not found in the DataFrame.")
        return
    
    # Convert to numeric
    df_slice[variable] = pd.to_numeric(df_slice[variable], errors='coerce')
    var = df_slice[variable]

    # Normalisation
    # df_slice['Wind_norm'] = wind_speed / wind_speed.max()

    # Smoothing
    if apply_smooth == True:
        var = smooth(var.to_numpy(), smoothie)
    else:
        var = var.to_numpy()

    # Clean Data
    valid_mask = ~np.isnan(var)

    var = var[valid_mask]
    time = df_slice['datetime'].to_numpy()[valid_mask]

    if plot == True:
        
        # Create Figure 
        plt.figure(figsize=(15,6))

        # Plot
        plt.plot(time, var, 
                color='black', linewidth=0.5)
        
        # Title construction
        title = f"{variable} at Station {df_slice['Station Number'].iloc[0]} for "
        # Year
        if isinstance(year, (tuple, list)):
            title += f"{year[0]} to {year[1]}"
        elif year is None:
            title += "All Years"
        else:
            title += f"{year}"
        # Month
        if isinstance(month, (tuple, list)):
            title += f", Months:{month[0]} to {month[1]}"
        elif isinstance(month, int):
            title += f", Month:{month}"
        # Day
        if isinstance(day, (tuple, list)):
            title += f", Days:{day[0]} to {day[1]}"
        elif isinstance(day, int):
            title += f", Day:{day}"
        # Hour
        if isinstance(hour, (tuple, list)):
            title += f", Hours:{hour[0]} to {hour[1]}"
        elif isinstance(hour, int):
            title += f", Hour:{hour}"
        plt.title(title)

        # Plot Formating
        plt.ylabel(f'{variable}')
        plt.xlabel('Time')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return time, var

def spectra_fft(aws_variable,
                wave_dict = None,
                config=None,
                use_file = False,
                seismic_mseed_name=None, 
                pad_value=np.nan):

    """
    Finds the correlation between seismic data and an AWS variable.
    Seismic data is trimmed to 30 minute segments which match AWS dataset.
    The real fast fourier transform is compute for each seismic component and 
    the power is calculated for each time step. For each frequency band the power
    is calculated and correlated with AWS data through the mutual time series.

    Parameters:
        aws_variable (array):
            An array of AWS data in format 
            aws_variable[0] (time), aws_variable[1] (variable array).
        wave_dict (dict):
             A wave dictionary containing seismic waveform data.
        config (dict):
            Information from a config file containing the local "seismic_data_path".
        use_file (bool):
            True/False. True to switch on file checking for mseed file.
        seismic_mseed_file (str):
            Title of saved mseed file.
            Seismic data must begin atleast 30 minutes before the first AWS timestamp
            and end anytime after the final AWS timestamp.
        pad_value (float):
            Value to use for padding.

    Returns:
       all_spectra (list):
            A list of dictionaries containing the frequency and power 
            for each component (EW, NS, Z) for each station and time period.
    """

    # File and Storage
    if use_file == True:
        # Path 
        base_path = Path(config["seismic_data_path"]) if config else Path(".")
        base_path.mkdir(parents=True, exist_ok=True)
        file_path = (base_path / seismic_mseed_name).with_suffix(".mseed")

        # Read file if it exists
        if file_path.exists():
            print(f"Reading existing file: {file_path}")
            stream = read(str(file_path))
            wave_dict = defaultdict(list)
            for tr in stream:
                wave_dict[tr.stats.station].append(tr)
        else:
            print("No File Found")

    else:
        if wave_dict == None:
            print('wave_dict not selected. Please input a file or a dictionary. (default: use_file = False )')
            return None
    
    # Setup 
    station_list = list(wave_dict.keys())
    aws_times = aws_variable[0]   # timestamps
    # EW, NS, Z components following real fast fourier transform (RFFT)
    all_spectra = []

    # Loop through all stations
    for station in station_list:
        
        spectra = []

        # Organise Streams
        st = Stream(wave_dict[station])

        # Puts in alphabetical order. 
        # E, N, Z
        st.sort(['channel'])   

        EW = st.select(channel="*E")[0].data
        NS = st.select(channel="*N")[0].data
        Z  = st.select(channel="*Z")[0].data
        
        # EW, NS, Z all same length and freq
        fs = st[0].stats.sampling_rate
        start_time = st[0].stats.starttime

        # Define Window Length Dependent on AWS
        # 30 Min (times 60 sec) AWS Measurement window
        # times seismic sampling rate
        window_length = int(30 * 60 * fs)

        # Setup time index list
        start_i = []

        # Loop through all time periods
        for time in aws_times:
            
            # 30 min collection time before AWS measurement
            t0 = UTC(str(time)) - timedelta(minutes=30)
            # Define index
            idx = int((t0 - start_time) * fs)
            start_i.append(idx)
        
        # Convert to np array
        start_i = np.array(start_i)
        # Define n
        n_starts = start_i.shape[0]

        # Setup Output Arrays
        EW_out = np.full((n_starts, window_length), pad_value, dtype=EW.dtype) 
        NS_out = np.full((n_starts, window_length), pad_value, dtype=NS.dtype) 
        Z_out = np.full((n_starts, window_length), pad_value, dtype=Z.dtype) 

        # Define the offsets
        offsets = np.arange(window_length)

        # Define the seismic index based on the offsets
        index = start_i[:, None] + offsets[None, :]

        # Check if data exists within the index bounds
        EW_in_bounds = (index >= 0) & (index < EW.shape[0])
        NS_in_bounds = (index >= 0) & (index < NS.shape[0])
        Z_in_bounds = (index >= 0) & (index < Z.shape[0])

        # Define the valid indicies
        EW_flat_idx = index[EW_in_bounds]
        NS_flat_idx = index[NS_in_bounds]
        Z_flat_idx = index[Z_in_bounds]

        # Place the balid indicies into the output arrays
        EW_out[EW_in_bounds] = EW[EW_flat_idx]
        NS_out[NS_in_bounds] = NS[NS_flat_idx]
        Z_out[Z_in_bounds] = Z[Z_flat_idx]

        # Stop if seismic data doesn't line up
        if (np.isnan(EW_out).any() or np.isnan(NS_out).any() or np.isnan(Z_out).any()):
            print('Error: Seismic data needs to span at least 30 minutes before AWS start time up until the final AWS time stamp.')
            return None

        # Apply Hann Window 
        taper_length = int(0.02 * window_length) # 2% taper
        window = np.ones(window_length) # Establish uniform window
        hann = np.hanning(2 * taper_length) # Create hann window for both (2) sides of data
        window[:taper_length] = hann[:taper_length] # Apply 2% to first half
        window[-taper_length:] = hann[taper_length:] # Apply 2% to second half
        # Apply
        EW_win = EW_out * window[None, :]
        NS_win = NS_out * window[None, :]
        Z_win = Z_out * window[None, :]

        # Compute rfft for each component
        y_EW = rfft(EW_win, axis = 1)
        y_NS = rfft(NS_win, axis = 1)
        y_Z = rfft(Z_win, axis = 1)
    
        # Compute rfft frequency
        freq = rfftfreq(window_length, 1/fs)

        # Calculate power for each component
        EW_p = np.abs(y_EW)**2
        NS_p = np.abs(y_NS)**2
        Z_p = np.abs(y_Z)**2

        # Save spectra as a dictionary
        spectra.append({'freq': freq,
                        'EW': EW_p,
                        'NS': NS_p,
                        'Z': Z_p,
                        'time' : aws_times,
                        'aws_values' : aws_variable[1]})

        all_spectra.append({station: spectra})
       
    return all_spectra
        
def seis_aws_fft_cor(spectra,
                    fmin = 1,
                    fmax = 49,
                    csv = False,
                    csv_title = 'results'):
    """
    Finds the correlation between seismic data and AWS wind speed.

    Parameters:
        spectra (list):
            A list of dictionaries containing the frequency and power 
            for each seismic component (EW, NS, Z) for each station and time period.
        fmin (int):
            Minimum frequency value for calculating the power of each bandwidths.
        fmax (int):
            Maximum frequency value for calculating the power of each  bandwidths.
        csv (bool):
            True/False. True to save returns as a csv file.
        csv_title (str):
            Title for the output csv file.

    Outputs:
        best_results (list):
            A list of dictionaries containing information about the best correlation result for each station.
        results (list):
            A list of dictionaries containing information about all the correlation results for each station.
    """
    
        
    results = []
    all_results = []
    best_results = []

    for station_dict in spectra:
        station = list(station_dict.keys())[0]

        EW_power = station_dict[station][0]['EW']
        NS_power = station_dict[station][0]['NS']
        Z_power = station_dict[station][0]['Z']
        freq = station_dict[station][0]['freq']
        aws_values = station_dict[station][0]['aws_values']

        valid = ~np.any(np.isnan(EW_power), axis=1)
        EW_valid = EW_power[valid]
        NS_valid = NS_power[valid]
        Z_valid = Z_power[valid]
        aws_valid = np.asarray(aws_values)[valid]

        station_results = []


        for f1 in range(fmin, fmax):
            for f2 in range(f1+1, fmax+1):

                band_width = (freq >= f1) & (freq <= f2)

                EW_band_power = np.mean(EW_valid[:, band_width], axis=1)
                NS_band_power = np.mean(NS_valid[:, band_width], axis=1)
                Z_band_power = np.mean(Z_valid[:, band_width], axis=1)

                # Remove outliers setup
                data = np.column_stack([aws_valid, Z_band_power, NS_band_power, EW_band_power])
                mean = np.nanmean(data, axis=0)
                std = np.nanstd(data, axis=0)

                # Avoid dividing by zero
                std[std == 0] = 1

                # Calculate z scores
                z_scores = (data - mean) / std

                # Remove outliers
                mask = np.all(np.abs(z_scores) < 3, axis=1)
                aws_mask = aws_valid[mask]
                Z_mask = Z_band_power[mask]
                NS_mask = NS_band_power[mask]
                EW_mask = EW_band_power[mask]
                
                # Reshape for R² analysis
                aws_array = np.array(aws_mask).reshape(-1, 1)
                
                # Calculate R² for each component
                Z_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(aws_array, Z_mask)
                Z_r_sq = Z_model.score(aws_array, Z_mask)
                
                NS_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(aws_array, NS_mask)
                NS_r_sq = NS_model.score(aws_array, NS_mask)
                
                EW_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(aws_array, EW_mask)
                EW_r_sq = EW_model.score(aws_array, EW_mask)

                # Store results
                station_results.append({
                    'fmin': f1,
                    'fmax': f2,
                    'Z_r2': Z_r_sq,
                    'NS_r2': NS_r_sq,
                    'EW_r2': EW_r_sq,
                    'avg_r2': (Z_r_sq + NS_r_sq + EW_r_sq) / 3,
                    'Z': Z_mask,
                    'NS': NS_mask,
                    'EW': EW_mask,
                    'Z_model': Z_model,
                    'NS_model': NS_model,
                    'EW_model': EW_model,
                    'aws_array': aws_array
                })

                print(f"Completed Iteration {f1}Hz-{f2}Hz. AVG R Sqaured = {(Z_r_sq + NS_r_sq + EW_r_sq) / 3}")

    
        # Find best result
        best_result = max(station_results, key=lambda x: x['avg_r2'])
        best_results.append({'station': station, **best_result})

        # Print best result
        print(f"\nBest frequency range for {station}: {best_result['fmin']} - {best_result['fmax']} Hz")
        print(f"Z R²: {best_result['Z_r2']:.4f}")
        print(f"NS R²: {best_result['NS_r2']:.4f}")
        print(f"EW R²: {best_result['EW_r2']:.4f}")
        print(f"Average R²: {best_result['avg_r2']:.4f}")
        results.append(station_results)

        # Save to csv file setup
        if csv == True:
            for result in station_results:
                all_results.append({
                    'station': station,
                    'fmin': result['fmin'],
                    'fmax': result['fmax'],
                    'Z_r2': result['Z_r2'],
                    'NS_r2': result['NS_r2'],
                    'EW_r2': result['EW_r2'],
                    'avg_r2': result['avg_r2']
                    })
                
    # Save all results to csv file
    if csv == True:
        df = pd.DataFrame(all_results)
        csv_name = f"{csv_title}.csv"
        df.to_csv(csv_name, index=False)
        print(f"Results saved to {csv_name}")


    return best_results, results
                
               
def single_rf(spectra,
                fmin = 1,
                fmax = 49,
                f_band_width = 1,
                step_size = 1,
                csv = False,
                csv_title = 'results',
                plot_best = True,
                plot_title = None,
                variable_units = None,
                plot_stat_results = True):
    
    """
    Finds the correlation between seismic data and AWS wind speed 
    via a Random Forest Regression.
    Only individual frequency bands inspect.

    Parameters:
        spectra (list):
            A list of dictionaries containing the frequency, power, and aws data 
            for each seismic component (EW, NS, Z) for each station and time period.
        fmin (int):
            Minimum frequency value for calculating the power of each bandwidths.
        fmax (int):
            Maximum frequency value for calculating the power of each  bandwidths.
        f_band_width (int):
            Bandwidth size. e.g. f_band_width = 1 for (f1,f2)=(1,2), 2 for (1,3), 3 for (1,4). 
        step_size (int):
            Frequency band step size. Set to less than f_band_width for overlapping bands. 
        csv (bool):
            True/False. True to save returns as a csv file.
        csv_title (str):
            Title for the output csv file.
        plot_best (bool):
            True/False. Plots the best R² and rmse for each station channel.
        plot_title (str):
            Title for plot_best plots.
        variable_units (str):
            AWS variable label, e.g 'wind speed (m/s)'
        plot_stat_results (bool):
            Plots all the R² and rmse values against frequency bandwidth centres.

    Outputs:
        best_r_results (list):
            A list of dictionaries containing information about the best (max) r2 correlation result for each station.
        best_rmse_results (list):
            A list of dictionaries containing information about the best (min) rmse correlation result for each station.
        results (list):
            A list of dictionaries containing information about all the correlation results for each station.
    """
    
    # Setup Result Lists
    results = []
    all_results = []
    best_r_results = []
    best_rmse_results = []
    best_cv_r_results = []

    # Create Bandwidths
    bands = []
    for f1 in range(fmin, fmax - f_band_width + 1, step_size):
        f2 = f1 + f_band_width
        band = (f1, f2)
        bands.append(band)

    # Loop through stations
    for station_dict in spectra:

        # Setup Variables
        station = list(station_dict.keys())[0] 
        EW_power = station_dict[station][0]['EW']
        NS_power = station_dict[station][0]['NS']
        Z_power = station_dict[station][0]['Z']
        freq = station_dict[station][0]['freq']
        aws_values = station_dict[station][0]['aws_values']

        # Setup results
        station_results = []

        # Apply bandwidths to data
        for i, (f1,f2) in enumerate(bands):
            band_width = (freq >= f1) & (freq < f2)

            # Convert to log to better inspect power scales and apply bandwidth
            # [:, band_width], select frequencies and slice unwanted freq data from the row
            # .mean(axis=1), mean for the selected frequency row. Reshape for model input.
            Z_log = np.log10(Z_power[:, band_width].mean(axis=1) + 1e-20).reshape(-1,1)
            NS_log = np.log10(NS_power[:, band_width].mean(axis=1) + 1e-20).reshape(-1,1)
            EW_log = np.log10(EW_power[:, band_width].mean(axis=1) + 1e-20).reshape(-1,1)
     
            # Train Model
            Z_X_train, Z_X_test, Z_y_train, Z_y_test = train_test_split(Z_log, aws_values, test_size=0.2,random_state = 42)
            NS_X_train, NS_X_test, NS_y_train, NS_y_test = train_test_split(NS_log, aws_values, test_size=0.2,random_state = 42)
            EW_X_train, EW_X_test, EW_y_train, EW_y_test = train_test_split(EW_log, aws_values, test_size=0.2,random_state = 42)
            
            # Define Random Forest Model
            Z_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
            NS_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
            EW_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
            
            # Cross Validation
            scores_Z = cross_val_score(Z_model, Z_log, aws_values, cv=5, scoring='r2')
            scores_NS = cross_val_score(NS_model, NS_log, aws_values, cv=5, scoring='r2')
            scores_EW = cross_val_score(EW_model, EW_log, aws_values, cv=5, scoring='r2')

            # Fit Model
            Z_model.fit(Z_X_train, Z_y_train)
            NS_model.fit(NS_X_train, NS_y_train)
            EW_model.fit(EW_X_train, EW_y_train)

            # Predictions
            Z_pred = Z_model.predict(Z_X_test)
            NS_pred = NS_model.predict(NS_X_test)
            EW_pred = EW_model.predict(EW_X_test)

            # rmse
            Z_rmse = np.sqrt(mean_squared_error(Z_y_test,Z_pred))
            NS_rmse = np.sqrt(mean_squared_error(NS_y_test,NS_pred))
            EW_rmse = np.sqrt(mean_squared_error(EW_y_test,EW_pred))

            # R²
            Z_r = r2_score(Z_y_test, Z_pred)
            NS_r = r2_score(NS_y_test, NS_pred)
            EW_r = r2_score(EW_y_test, EW_pred)
            
            # Store results
            station_results.append({
                'fmin': f1,
                'fmax': f2,
                'Z_r2': Z_r,
                'NS_r2': NS_r,
                'EW_r2': EW_r,
                'avg_r2': (Z_r + NS_r + EW_r) / 3,
                'Z_rmse': Z_rmse,
                'NS_rmse': NS_rmse,
                'EW_rmse': EW_rmse,
                'avg_rmse': (Z_rmse + NS_rmse + EW_rmse) / 3,
                'Z_y_test': Z_y_test,
                'NS_y_test': NS_y_test,
                'EW_y_test': EW_y_test,
                'Z_pred': Z_pred,
                'NS_pred': NS_pred,
                'EW_pred': EW_pred,
                'Z_cv_r2': scores_Z.mean(),
                'NS_cv_r2': scores_NS.mean(),
                'EW_cv_r2': scores_EW.mean(),
                'avg_cv_r2': (scores_Z.mean() + scores_NS.mean() + scores_EW.mean()) / 3,
                'Z_cv_std': scores_Z.std(),
                'NS_cv_std': scores_NS.std(),
                'EW_cv_std': scores_EW.std(),
                'avg_cv_std': (scores_Z.std() + scores_NS.std() + scores_EW.std()) / 3 })
        
        # Find best result
        # R²
        best_r_result = max(station_results, key=lambda x: x['avg_r2'])
        best_r_results.append({'station': station, **best_r_result})
        # rmse
        best_rmse_result = min(station_results, key=lambda x: x['avg_rmse'])
        best_rmse_results.append({'station': station, **best_rmse_result})
        # Cross Validation R²
        best_cv_r_result = max(station_results, key=lambda x: x['avg_cv_r2'])
        best_cv_r_results.append({'station': station, **best_cv_r_result})
    
        # Print best result
        # R²
        print(f"\nBest R² value frequency range for {station}: {best_r_result['fmin']} - {best_r_result['fmax']} Hz")
        print(f"Z R²: {best_r_result['Z_r2']:.4f}")
        print(f"NS R²: {best_r_result['NS_r2']:.4f}")
        print(f"EW R²: {best_r_result['EW_r2']:.4f}")
        print(f"Average R²: {best_r_result['avg_r2']:.4f}")
        # rmse
        print(f"\nBest rmse value frequency range for {station}: {best_rmse_result['fmin']} - {best_rmse_result['fmax']} Hz")
        print(f"Z rmse: {best_rmse_result['Z_rmse']:.4f}")
        print(f"NS rmse: {best_rmse_result['NS_rmse']:.4f}")
        print(f"EW rmse: {best_rmse_result['EW_rmse']:.4f}")
        print(f"Average rmse: {best_rmse_result['avg_rmse']:.4f}")
        # Cross Validation R²
        print(f"\nBest cv R² value frequency range for {station}: {best_cv_r_result['fmin']} - {best_cv_r_result['fmax']} Hz")
        print(f"Z cv R²: {best_cv_r_result['Z_cv_r2']:.4f} + {best_cv_r_result['Z_cv_std']:.4f}")
        print(f"NS cv R²: {best_cv_r_result['NS_cv_r2']:.4f} + {best_cv_r_result['NS_cv_std']:.4f}")
        print(f"EW cv R²: {best_cv_r_result['EW_cv_r2']:.4f} + {best_cv_r_result['EW_cv_std']:.4f}")
        print(f"Average cv R²: {best_cv_r_result['avg_cv_r2']:.4f}")
        
        results.append(station_results)

        # Plot all average R² and rmse results against centre frequency
        if plot_stat_results ==True:
            band_centres = []
            r2_list = []
            rmse_list = []
            for r in station_results:
                band_centre = [(r['fmin'] + r['fmax'])/2]
                r2 = r['avg_r2']
                rmse = r['avg_rmse']
                band_centres.append(band_centre)
                r2_list.append(r2)
                rmse_list.append(rmse)

            fig, axs = plt.subplots(1,2, figsize=(8,4))
            # R²
            axs[0].scatter(band_centres, r2_list)
            axs[0].set_title(f'{station}: Band Centre Frequency vs Average R²')
            axs[0].set_xlabel('Band Centre Frequency (Hz)')
            axs[0].set_ylabel('Average R²')
            # rmse
            axs[1].scatter(band_centres, rmse_list)
            axs[1].set_title(f'{station}: Band Centre Frequency vs Average rmse')
            axs[1].set_xlabel('Band Centre Frequency (Hz)')
            axs[1].set_ylabel('rmse')

        # Plot the best R² and rmse results for each station channel
        if plot_best == True:
            
            fig, axs = plt.subplots(3, 2, figsize=(10, 10))
            # R²
            axs[0,0].scatter(best_r_result['Z_y_test'],best_r_result['Z_pred'], marker='x')
            lims = [min(axs[0,0].get_xlim()[0], axs[0,0].get_ylim()[0]), max(axs[0,0].get_xlim()[1], axs[0,0].get_ylim()[1])]
            axs[0,0].plot(lims, lims, 'r--')
            axs[0,0].set_title(f"Best R² Z: {best_r_result['Z_r2']:.4f}")

            axs[1,0].scatter(best_r_result['NS_y_test'],best_r_result['NS_pred'], marker='x')
            lims = [min(axs[1,0].get_xlim()[0], axs[1,0].get_ylim()[0]), max(axs[1,0].get_xlim()[1], axs[1,0].get_ylim()[1])]
            axs[1,0].plot(lims, lims, 'r--')
            axs[1,0].set_title(f"Best R² NS: {best_r_result['NS_r2']:.4f}")

            axs[2,0].scatter(best_r_result['EW_y_test'],best_r_result['EW_pred'], marker='x')
            lims = [min(axs[2,0].get_xlim()[0], axs[2,0].get_ylim()[0]), max(axs[2,0].get_xlim()[1], axs[2,0].get_ylim()[1])]
            axs[2,0].plot(lims, lims, 'r--')           
            axs[2,0].set_title(f"Best R² EW: {best_r_result['EW_r2']:.4f}")

            # rmse
            axs[0,1].scatter(best_rmse_result['Z_y_test'],best_rmse_result['Z_pred'], marker='x')
            lims = [min(axs[0,1].get_xlim()[0], axs[0,1].get_ylim()[0]), max(axs[0,1].get_xlim()[1], axs[0,1].get_ylim()[1])]
            axs[0,1].plot(lims, lims, 'r--') 
            axs[0,1].set_title(f"Best rmse Z: {best_rmse_result['Z_rmse']:.4f}")

            axs[1,1].scatter(best_rmse_result['NS_y_test'],best_rmse_result['NS_pred'], marker='x')
            lims = [min(axs[1,1].get_xlim()[0], axs[1,1].get_ylim()[0]), max(axs[1,1].get_xlim()[1], axs[1,1].get_ylim()[1])]
            axs[1,1].plot(lims, lims, 'r--') 
            axs[1,1].set_title(f"Best rmse Z: {best_rmse_result['NS_rmse']:.4f}")

            axs[2,1].scatter(best_rmse_result['EW_y_test'],best_rmse_result['EW_pred'], marker='x')
            lims = [min(axs[2,1].get_xlim()[0], axs[2,1].get_ylim()[0]), max(axs[2,1].get_xlim()[1], axs[2,1].get_ylim()[1])]
            axs[2,1].plot(lims, lims, 'r--') 
            axs[2,1].set_title(f"Best rmse Z: {best_rmse_result['avg_rmse']:.4f}")
            
            # Figure Labels
            if variable_units is None: 
                fig.supxlabel('Observed AWS Measurement')
                fig.supylabel('Predicted AWS Measurement')
            else:
                fig.supxlabel(f'Observed {variable_units}')
                fig.supylabel(f'Predicted {variable_units}')
            if plot_title is None:
                title = f'{station}: Observed AWS Measurement vs Predicted AWS Measured'
            else:
                title = plot_title 

            plt.suptitle(title)

        # Save to csv file setup
        if csv == True:
            for result in station_results:
                all_results.append({
                    'station': station,
                    'fmin': result['fmin'],
                    'fmax': result['fmax'],
                    'Z_r2': result['Z_r2'],
                    'NS_r2': result['NS_r2'],
                    'EW_r2': result['EW_r2'],
                    'avg_r2': result['avg_r2'],
                    'Z_rmse': result['Z_rmse'],
                    'NS_rmse': result['NS_rmse'],
                    'EW_rmse': result['EW_rmse'],
                    'avg_rmse': result['avg_rmse']})

    # Save all results to csv file
    if csv == True:
        df = pd.DataFrame(all_results)
        csv_name = f"{csv_title}.csv"
        df.to_csv(csv_name, index=False)
        print(f"Results saved to {csv_name}")
                                
    return best_r_results, best_rmse_results, results, best_cv_r_results
        
def single_ridge(spectra,
                   fmin = 1,
                   fmax = 49,
                   f_band_width = 1,
                   step_size = 1,
                   csv = False,
                   csv_title = 'results',
                   plot_best = True,
                   plot_title = None,
                   variable_units = None,
                   plot_stat_results = True):
    
    """
    Finds the correlation between seismic data and AWS wind speed 
    via a Ridge Regression.
    Only individual frequency bands inspect.

    Parameters:
        spectra (list):
            A list of dictionaries containing the frequency, power, and aws data 
            for each seismic component (EW, NS, Z) for each station and time period.
        fmin (int):
            Minimum frequency value for calculating the power of each bandwidths.
        fmax (int):
            Maximum frequency value for calculating the power of each  bandwidths.
        f_band_width (int):
            Bandwidth size. e.g. f_band_width = 1 for (f1,f2)=(1,2), 2 for (1,3), 3 for (1,4). 
        step_size (int):
            Frequency band step size. Set to less than f_band_width for overlapping bands. 
        csv (bool):
            True/False. True to save returns as a csv file.
        csv_title (str):
            Title for the output csv file.
        plot_best (bool):
            True/False. Plots the best R² and rmse for each station channel.
        plot_title (str):
            Title for plot_best plots.
        variable_units (str):
            AWS variable label, e.g 'wind speed (m/s)'
        plot_stat_results (bool):
            Plots all the R² and rmse values against frequency bandwidth centres.

    Outputs:
        best_r_results (list):
            A list of dictionaries containing information about the best (max) r2 correlation result for each station.
        best_rmse_results (list):
            A list of dictionaries containing information about the best (min) rmse correlation result for each station.
        results (list):
            A list of dictionaries containing information about all the correlation results for each station.
    """
    
    
    # Setup Result Lists
    results = []
    all_results = []
    best_r_results = []
    best_cv_r_results = []

    # Create Bandwidths
    bands = []
    for f1 in range(fmin, fmax - f_band_width + 1, step_size):
        f2 = f1 + f_band_width
        band = (f1, f2)
        bands.append(band)

    # Loop through stations
    for station_dict in spectra:

        # Setup Variables
        station = list(station_dict.keys())[0] 
        EW_power = station_dict[station][0]['EW']
        NS_power = station_dict[station][0]['NS']
        Z_power = station_dict[station][0]['Z']
        freq = station_dict[station][0]['freq']
        aws_values = station_dict[station][0]['aws_values']

        # Setup results
        station_results = []

        # Apply bandwidths to data
        for i, (f1,f2) in enumerate(bands):
            band_width = (freq >= f1) & (freq < f2)

            # Convert to log to better inspect power scales and apply bandwidth
            # [:, band_width], select frequencies and slice unwanted freq data from the row
            # .mean(axis=1), mean for the selected frequency row. Reshape for model input.
            Z_log = np.log10(Z_power[:, band_width].mean(axis=1) + 1e-20).reshape(-1,1)
            NS_log = np.log10(NS_power[:, band_width].mean(axis=1) + 1e-20).reshape(-1,1)
            EW_log = np.log10(EW_power[:, band_width].mean(axis=1) + 1e-20).reshape(-1,1)
     
            # Train Model
            Z_X_train, Z_X_test, Z_y_train, Z_y_test = train_test_split(Z_log, aws_values, test_size=0.2,random_state = 42)
            NS_X_train, NS_X_test, NS_y_train, NS_y_test = train_test_split(NS_log, aws_values, test_size=0.2,random_state = 42)
            EW_X_train, EW_X_test, EW_y_train, EW_y_test = train_test_split(EW_log, aws_values, test_size=0.2,random_state = 42)

            # Scale 
            Z_scaler = StandardScaler()
            NS_scaler = StandardScaler()
            EW_scaler = StandardScaler()
            Z_X_train_scaled = Z_scaler.fit_transform(Z_X_train)
            Z_X_test_scaled = Z_scaler.transform(Z_X_test)
            NS_X_train_scaled = NS_scaler.fit_transform(NS_X_train)
            NS_X_test_scaled = NS_scaler.transform(NS_X_test)
            EW_X_train_scaled = EW_scaler.fit_transform(EW_X_train)
            EW_X_test_scaled = EW_scaler.transform(EW_X_test)

            # Ridge Model
            Z_model = Ridge(alpha=1.0, solver='auto')
            NS_model = Ridge(alpha=1.0, solver='auto')
            EW_model = Ridge(alpha=1.0, solver='auto')
   
            # Cross Validation
            scores_Z = cross_val_score(Z_model, Z_log, aws_values, cv=5, scoring='r2')
            scores_NS = cross_val_score(NS_model, NS_log, aws_values, cv=5, scoring='r2')
            scores_EW = cross_val_score(EW_model, EW_log, aws_values, cv=5, scoring='r2')

            # Fit Model
            Z_model.fit(Z_X_train_scaled, Z_y_train)
            NS_model.fit(NS_X_train_scaled, NS_y_train)
            EW_model.fit(EW_X_train_scaled, EW_y_train)

            # Predictions
            Z_pred = Z_model.predict(Z_X_test_scaled)
            NS_pred = NS_model.predict(NS_X_test_scaled)
            EW_pred = EW_model.predict(EW_X_test_scaled)

            # R²
            Z_r = Z_model.score(Z_X_test_scaled, Z_y_test)
            NS_r = NS_model.score(NS_X_test_scaled, NS_y_test)
            EW_r = EW_model.score(EW_X_test_scaled, EW_y_test)
            
            # Store results
            station_results.append({
                'fmin': f1,
                'fmax': f2,
                'Z_r2': Z_r,
                'NS_r2': NS_r,
                'EW_r2': EW_r,
                'avg_r2': (Z_r + NS_r + EW_r) / 3,
                'Z_y_test': Z_y_test,
                'NS_y_test': NS_y_test,
                'EW_y_test': EW_y_test,
                'Z_pred': Z_pred,
                'NS_pred': NS_pred,
                'EW_pred': EW_pred,
                'Z_cv_r2': scores_Z.mean(),
                'NS_cv_r2': scores_NS.mean(),
                'EW_cv_r2': scores_EW.mean(),
                'avg_cv_r2': (scores_Z.mean() + scores_NS.mean() + scores_EW.mean()) / 3,
                'Z_cv_std': scores_Z.std(),
                'NS_cv_std': scores_NS.std(),
                'EW_cv_std': scores_EW.std(),
                'avg_cv_std': (scores_Z.std() + scores_NS.std() + scores_EW.std()) / 3 })
        
        # Find best result
        # R²
        best_r_result = max(station_results, key=lambda x: x['avg_r2'])
        best_r_results.append({'station': station, **best_r_result})
        # Cross Validation R²
        best_cv_r_result = max(station_results, key=lambda x: x['avg_cv_r2'])
        best_cv_r_results.append({'station': station, **best_cv_r_result})
    
        # Print best result
        # R²
        print(f"\nBest R² value frequency range for {station}: {best_r_result['fmin']} - {best_r_result['fmax']} Hz")
        print(f"Z R²: {best_r_result['Z_r2']:.4f}")
        print(f"NS R²: {best_r_result['NS_r2']:.4f}")
        print(f"EW R²: {best_r_result['EW_r2']:.4f}")
        print(f"Average R²: {best_r_result['avg_r2']:.4f}")
        # Cross Validation R²
        print(f"\nBest cv R² value frequency range for {station}: {best_cv_r_result['fmin']} - {best_cv_r_result['fmax']} Hz")
        print(f"Z cv R²: {best_cv_r_result['Z_cv_r2']:.4f} + {best_cv_r_result['Z_cv_std']:.4f}")
        print(f"NS cv R²: {best_cv_r_result['NS_cv_r2']:.4f} + {best_cv_r_result['NS_cv_std']:.4f}")
        print(f"EW cv R²: {best_cv_r_result['EW_cv_r2']:.4f} + {best_cv_r_result['EW_cv_std']:.4f}")
        print(f"Average cv R²: {best_cv_r_result['avg_cv_r2']:.4f}")
        
        results.append(station_results)

        # Plot all average R² and rmse results against centre frequency
        if plot_stat_results == True:
            band_centres = []
            r2_list = []
            for r in station_results:
                band_centre = [(r['fmin'] + r['fmax'])/2]
                r2 = r['avg_r2']
                band_centres.append(band_centre)
                r2_list.append(r2)
            
            # R²
            plt.scatter(band_centres, r2_list)
            plt.title(f'{station}: Band Centre Frequency vs Average R²')
            plt.xlabel('Band Centre Frequency (Hz)')
            plt.ylabel('Average R²')
            
        # Plot the best R² and rmse results for each station channel
        if plot_best == True:
            
            fig, axs = plt.subplots(3, 1, figsize=(10, 10))
            # R²
            axs[0].scatter(best_r_result['Z_y_test'],best_r_result['Z_pred'], marker='x')
            lims = [min(axs[0].get_xlim()[0], axs[0].get_ylim()[0]), max(axs[0].get_xlim()[1], axs[0].get_ylim()[1])]
            axs[0].plot(lims, lims, 'r--')
            axs[0].set_title(f"Best R² Z: {best_r_result['Z_r2']:.4f}")

            axs[1].scatter(best_r_result['NS_y_test'],best_r_result['NS_pred'], marker='x')
            lims = [min(axs[1].get_xlim()[0], axs[1].get_ylim()[0]), max(axs[1].get_xlim()[1], axs[1].get_ylim()[1])]
            axs[1].plot(lims, lims, 'r--')
            axs[1].set_title(f"Best R² NS: {best_r_result['NS_r2']:.4f}")

            axs[2].scatter(best_r_result['EW_y_test'],best_r_result['EW_pred'], marker='x')
            lims = [min(axs[2].get_xlim()[0], axs[2].get_ylim()[0]), max(axs[2].get_xlim()[1], axs[2].get_ylim()[1])]
            axs[2].plot(lims, lims, 'r--')           
            axs[2].set_title(f"Best R² EW: {best_r_result['EW_r2']:.4f}")

            # Figure Labels
            if variable_units is None: 
                fig.supxlabel('Observed AWS Measurement')
                fig.supylabel('Predicted AWS Measurement')
            else:
                fig.supxlabel(f'Observed {variable_units}')
                fig.supylabel(f'Predicted {variable_units}')
            if plot_title is None:
                title = f'{station}: Observed AWS Measurement vs Predicted AWS Measured'
            else:
                title = plot_title 

            plt.suptitle(title)

        # Save to csv file setup
        if csv == True:
            for result in station_results:
                all_results.append({
                    'station': station,
                    'fmin': result['fmin'],
                    'fmax': result['fmax'],
                    'Z_r2': result['Z_r2'],
                    'NS_r2': result['NS_r2'],
                    'EW_r2': result['EW_r2'],
                    'avg_r2': result['avg_r2']})

    # Save all results to csv file
    if csv == True:
        df = pd.DataFrame(all_results)
        csv_name = f"{csv_title}.csv"
        df.to_csv(csv_name, index=False)
        print(f"Results saved to {csv_name}")
                                
    return best_r_results, results


def single_elasticnet(spectra,
                        fmin = 1,
                        fmax = 49,
                        f_band_width = 1,
                        step_size = 1,
                        csv = False,
                        csv_title = 'results',
                        plot_best = True,
                        plot_title = None,
                        variable_units = None,
                        plot_stat_results = True):
    
    """
    Finds the correlation between seismic data and AWS wind speed 
    via a ElasticNet Regression.
    Only individual frequency bands inspect.

    Parameters:
        spectra (list):
            A list of dictionaries containing the frequency, power, and aws data 
            for each seismic component (EW, NS, Z) for each station and time period.
        fmin (int):
            Minimum frequency value for calculating the power of each bandwidths.
        fmax (int):
            Maximum frequency value for calculating the power of each  bandwidths.
        f_band_width (int):
            Bandwidth size. e.g. f_band_width = 1 for (f1,f2)=(1,2), 2 for (1,3), 3 for (1,4). 
        step_size (int):
            Frequency band step size. Set to less than f_band_width for overlapping bands. 
        csv (bool):
            True/False. True to save returns as a csv file.
        csv_title (str):
            Title for the output csv file.
        plot_best (bool):
            True/False. Plots the best R² and rmse for each station channel.
        plot_title (str):
            Title for plot_best plots.
        variable_units (str):
            AWS variable label, e.g 'wind speed (m/s)'
        plot_stat_results (bool):
            Plots all the R² and rmse values against frequency bandwidth centres.

    Outputs:
        best_r_results (list):
            A list of dictionaries containing information about the best (max) r2 correlation result for each station.
        best_rmse_results (list):
            A list of dictionaries containing information about the best (min) rmse correlation result for each station.
        results (list):
            A list of dictionaries containing information about all the correlation results for each station.
    """
    
    
    # Setup Result Lists
    results = []
    all_results = []
    best_r_results = []
    best_cv_r_results = []

    # Create Bandwidths
    bands = []
    for f1 in range(fmin, fmax - f_band_width + 1, step_size):
        f2 = f1 + f_band_width
        band = (f1, f2)
        bands.append(band)

    # Loop through stations
    for station_dict in spectra:

        # Setup Variables
        station = list(station_dict.keys())[0] 
        EW_power = station_dict[station][0]['EW']
        NS_power = station_dict[station][0]['NS']
        Z_power = station_dict[station][0]['Z']
        freq = station_dict[station][0]['freq']
        aws_values = station_dict[station][0]['aws_values']

        # Setup results
        station_results = []

        # Apply bandwidths to data
        for i, (f1,f2) in enumerate(bands):
            band_width = (freq >= f1) & (freq < f2)
    
            # Convert to log to better inspect power scales and apply bandwidth
            # [:, band_width], select frequencies and slice unwanted freq data from the row
            # .mean(axis=1), mean for the selected frequency row. Reshape for model input.
            Z_log = np.log10(Z_power[:, band_width].mean(axis=1) + 1e-20).reshape(-1,1)
            NS_log = np.log10(NS_power[:, band_width].mean(axis=1) + 1e-20).reshape(-1,1)
            EW_log = np.log10(EW_power[:, band_width].mean(axis=1) + 1e-20).reshape(-1,1)

            # Train Model
            Z_X_train, Z_X_test, Z_y_train, Z_y_test = train_test_split(Z_log, aws_values, test_size=0.2,random_state = 42)
            NS_X_train, NS_X_test, NS_y_train, NS_y_test = train_test_split(NS_log, aws_values, test_size=0.2,random_state = 42)
            EW_X_train, EW_X_test, EW_y_train, EW_y_test = train_test_split(EW_log, aws_values, test_size=0.2,random_state = 42)
            
            # Scale 
            Z_scaler = StandardScaler()
            NS_scaler = StandardScaler()
            EW_scaler = StandardScaler()
            Z_X_train_scaled = Z_scaler.fit_transform(Z_X_train)
            Z_X_test_scaled = Z_scaler.transform(Z_X_test)
            NS_X_train_scaled = NS_scaler.fit_transform(NS_X_train)
            NS_X_test_scaled = NS_scaler.transform(NS_X_test)
            EW_X_train_scaled = EW_scaler.fit_transform(EW_X_train)
            EW_X_test_scaled = EW_scaler.transform(EW_X_test)

            # Elastic Net Model (Need to find optimal l1 still)
            Z_model = ElasticNet(alpha=0.08, l1_ratio=0.5).fit(Z_X_train_scaled, Z_y_train)
            NS_model = ElasticNet(alpha=0.08, l1_ratio=0.5).fit(NS_X_train_scaled, NS_y_train)
            EW_model = ElasticNet(alpha=0.08, l1_ratio=0.5).fit(EW_X_train_scaled, EW_y_train)

            # Cross Validation
            scores_Z = cross_val_score(Z_model, Z_log, aws_values, cv=5, scoring='r2')
            scores_NS = cross_val_score(NS_model, NS_log, aws_values, cv=5, scoring='r2')
            scores_EW = cross_val_score(EW_model, EW_log, aws_values, cv=5, scoring='r2')

            # Predictions
            Z_pred = Z_model.predict(Z_X_test_scaled)
            NS_pred = NS_model.predict(NS_X_test_scaled)
            EW_pred = EW_model.predict(EW_X_test_scaled)

            # R²
            Z_r = Z_model.score(Z_X_test_scaled, Z_y_test)
            NS_r = NS_model.score(NS_X_test_scaled, NS_y_test)
            EW_r = EW_model.score(EW_X_test_scaled, EW_y_test)
            
            # Store results
            station_results.append({
                'fmin': f1,
                'fmax': f2,
                'Z_r2': Z_r,
                'NS_r2': NS_r,
                'EW_r2': EW_r,
                'avg_r2': (Z_r + NS_r + EW_r) / 3,
                'Z_y_test': Z_y_test,
                'NS_y_test': NS_y_test,
                'EW_y_test': EW_y_test,
                'Z_pred': Z_pred,
                'NS_pred': NS_pred,
                'EW_pred': EW_pred,
                'Z_cv_r2': scores_Z.mean(),
                'NS_cv_r2': scores_NS.mean(),
                'EW_cv_r2': scores_EW.mean(),
                'avg_cv_r2': (scores_Z.mean() + scores_NS.mean() + scores_EW.mean()) / 3,
                'Z_cv_std': scores_Z.std(),
                'NS_cv_std': scores_NS.std(),
                'EW_cv_std': scores_EW.std(),
                'avg_cv_std': (scores_Z.std() + scores_NS.std() + scores_EW.std()) / 3 })
        
        # Find best result
        # R²
        best_r_result = max(station_results, key=lambda x: x['avg_r2'])
        best_r_results.append({'station': station, **best_r_result})
        # Cross Validation R²
        best_cv_r_result = max(station_results, key=lambda x: x['avg_cv_r2'])
        best_cv_r_results.append({'station': station, **best_cv_r_result})
    
        # Print best result
        # R²
        print(f"\nBest R² value frequency range for {station}: {best_r_result['fmin']} - {best_r_result['fmax']} Hz")
        print(f"Z R²: {best_r_result['Z_r2']:.4f}")
        print(f"NS R²: {best_r_result['NS_r2']:.4f}")
        print(f"EW R²: {best_r_result['EW_r2']:.4f}")
        print(f"Average R²: {best_r_result['avg_r2']:.4f}")
        # Cross Validation R²
        print(f"\nBest cv R² value frequency range for {station}: {best_cv_r_result['fmin']} - {best_cv_r_result['fmax']} Hz")
        print(f"Z cv R²: {best_cv_r_result['Z_cv_r2']:.4f} + {best_cv_r_result['Z_cv_std']:.4f}")
        print(f"NS cv R²: {best_cv_r_result['NS_cv_r2']:.4f} + {best_cv_r_result['NS_cv_std']:.4f}")
        print(f"EW cv R²: {best_cv_r_result['EW_cv_r2']:.4f} + {best_cv_r_result['EW_cv_std']:.4f}")
        print(f"Average cv R²: {best_cv_r_result['avg_cv_r2']:.4f}")
        
        results.append(station_results)

        # Plot all average R² and rmse results against centre frequency
        if plot_stat_results == True:
            band_centres = []
            r2_list = []
            for r in station_results:
                band_centre = [(r['fmin'] + r['fmax'])/2]
                r2 = r['avg_r2']
                band_centres.append(band_centre)
                r2_list.append(r2)
            
            # R²
            plt.scatter(band_centres, r2_list)
            plt.title(f'{station}: Band Centre Frequency vs Average R²')
            plt.xlabel('Band Centre Frequency (Hz)')
            plt.ylabel('Average R²')
            
        # Plot the best R² and rmse results for each station channel
        if plot_best == True:
            
            fig, axs = plt.subplots(3, 1, figsize=(10, 10))
            # R²
            axs[0].scatter(best_r_result['Z_y_test'],best_r_result['Z_pred'], marker='x')
            lims = [min(axs[0].get_xlim()[0], axs[0].get_ylim()[0]), max(axs[0].get_xlim()[1], axs[0].get_ylim()[1])]
            axs[0].plot(lims, lims, 'r--')
            axs[0].set_title(f"Best R² Z: {best_r_result['Z_r2']:.4f}")

            axs[1].scatter(best_r_result['NS_y_test'],best_r_result['NS_pred'], marker='x')
            lims = [min(axs[1].get_xlim()[0], axs[1].get_ylim()[0]), max(axs[1].get_xlim()[1], axs[1].get_ylim()[1])]
            axs[1].plot(lims, lims, 'r--')
            axs[1].set_title(f"Best R² NS: {best_r_result['NS_r2']:.4f}")

            axs[2].scatter(best_r_result['EW_y_test'],best_r_result['EW_pred'], marker='x')
            lims = [min(axs[2].get_xlim()[0], axs[2].get_ylim()[0]), max(axs[2].get_xlim()[1], axs[2].get_ylim()[1])]
            axs[2].plot(lims, lims, 'r--')           
            axs[2].set_title(f"Best R² EW: {best_r_result['EW_r2']:.4f}")

            # Figure Labels
            if variable_units is None: 
                fig.supxlabel('Observed AWS Measurement')
                fig.supylabel('Predicted AWS Measurement')
            else:
                fig.supxlabel(f'Observed {variable_units}')
                fig.supylabel(f'Predicted {variable_units}')
            if plot_title is None:
                title = f'{station}: Observed AWS Measurement vs Predicted AWS Measured'
            else:
                title = plot_title 

            plt.suptitle(title)

        # Save to csv file setup
        if csv == True:
            for result in station_results:
                all_results.append({
                    'station': station,
                    'fmin': result['fmin'],
                    'fmax': result['fmax'],
                    'Z_r2': result['Z_r2'],
                    'NS_r2': result['NS_r2'],
                    'EW_r2': result['EW_r2'],
                    'avg_r2': result['avg_r2']})

    # Save all results to csv file
    if csv == True:
        df = pd.DataFrame(all_results)
        csv_name = f"{csv_title}.csv"
        df.to_csv(csv_name, index=False)
        print(f"Results saved to {csv_name}")
                                
    return best_r_results, results

def single_svr(spectra,
                 fmin = 1,
                 fmax = 49,
                 f_band_width = 1,
                 step_size = 1,
                 csv = False,
                 csv_title = 'results',
                 plot_best = True,
                 plot_title = None,
                 variable_units = None,
                 plot_stat_results = True):
    
    """
    Finds the correlation between seismic data and AWS wind speed 
    via a SVR Regression.
    Only individual frequency bands inspect.

    Parameters:
        spectra (list):
            A list of dictionaries containing the frequency, power, and aws data 
            for each seismic component (EW, NS, Z) for each station and time period.
        fmin (int):
            Minimum frequency value for calculating the power of each bandwidths.
        fmax (int):
            Maximum frequency value for calculating the power of each  bandwidths.
        f_band_width (int):
            Bandwidth size. e.g. f_band_width = 1 for (f1,f2)=(1,2), 2 for (1,3), 3 for (1,4). 
        step_size (int):
            Frequency band step size. Set to less than f_band_width for overlapping bands. 
        csv (bool):
            True/False. True to save returns as a csv file.
        csv_title (str):
            Title for the output csv file.
        plot_best (bool):
            True/False. Plots the best R² and rmse for each station channel.
        plot_title (str):
            Title for plot_best plots.
        variable_units (str):
            AWS variable label, e.g 'wind speed (m/s)'
        plot_stat_results (bool):
            Plots all the R² and rmse values against frequency bandwidth centres.

    Outputs:
        best_r_results (list):
            A list of dictionaries containing information about the best (max) r2 correlation result for each station.
        best_rmse_results (list):
            A list of dictionaries containing information about the best (min) rmse correlation result for each station.
        results (list):
            A list of dictionaries containing information about all the correlation results for each station.
    """
    
    
    # Setup Result Lists
    results = []
    all_results = []
    best_r_results = []
    best_cv_r_results = []

    # Create Bandwidths
    bands = []
    for f1 in range(fmin, fmax - f_band_width + 1, step_size):
        f2 = f1 + f_band_width
        band = (f1, f2)
        bands.append(band)

    # Loop through stations
    for station_dict in spectra:

        # Setup Variables
        station = list(station_dict.keys())[0] 
        EW_power = station_dict[station][0]['EW']
        NS_power = station_dict[station][0]['NS']
        Z_power = station_dict[station][0]['Z']
        freq = station_dict[station][0]['freq']
        aws_values = station_dict[station][0]['aws_values']

        # Setup results
        station_results = []

        # Apply bandwidths to data
        for i, (f1,f2) in enumerate(bands):
            band_width = (freq >= f1) & (freq < f2)
    
            # Convert to log to better inspect power scales and apply bandwidth
            # [:, band_width], select frequencies and slice unwanted freq data from the row
            # .mean(axis=1), mean for the selected frequency row. Reshape for model input.
            Z_log = np.log10(Z_power[:, band_width].mean(axis=1) + 1e-20).reshape(-1,1)
            NS_log = np.log10(NS_power[:, band_width].mean(axis=1) + 1e-20).reshape(-1,1)
            EW_log = np.log10(EW_power[:, band_width].mean(axis=1) + 1e-20).reshape(-1,1)

            # Train Model
            Z_X_train, Z_X_test, Z_y_train, Z_y_test = train_test_split(Z_log, aws_values, test_size=0.2,random_state = 42)
            NS_X_train, NS_X_test, NS_y_train, NS_y_test = train_test_split(NS_log, aws_values, test_size=0.2,random_state = 42)
            EW_X_train, EW_X_test, EW_y_train, EW_y_test = train_test_split(EW_log, aws_values, test_size=0.2,random_state = 42)
            
            # Scale 
            Z_scaler = StandardScaler()
            NS_scaler = StandardScaler()
            EW_scaler = StandardScaler()
            Z_X_train_scaled = Z_scaler.fit_transform(Z_X_train)
            Z_X_test_scaled = Z_scaler.transform(Z_X_test)
            NS_X_train_scaled = NS_scaler.fit_transform(NS_X_train)
            NS_X_test_scaled = NS_scaler.transform(NS_X_test)
            EW_X_train_scaled = EW_scaler.fit_transform(EW_X_train)
            EW_X_test_scaled = EW_scaler.transform(EW_X_test)

            # SVR (Need to find optimal values)
            Z_model = SVR(C=1.0, epsilon=0.2).fit(Z_X_train_scaled, Z_y_train)
            NS_model = SVR(C=1.0, epsilon=0.2).fit(NS_X_train_scaled, NS_y_train)
            EW_model = SVR(C=1.0, epsilon=0.2).fit(EW_X_train_scaled, EW_y_train)

            # Cross Validation
            scores_Z = cross_val_score(Z_model, Z_log, aws_values, cv=5, scoring='r2')
            scores_NS = cross_val_score(NS_model, NS_log, aws_values, cv=5, scoring='r2')
            scores_EW = cross_val_score(EW_model, EW_log, aws_values, cv=5, scoring='r2')

            # Predictions
            Z_pred = Z_model.predict(Z_X_test_scaled)
            NS_pred = NS_model.predict(NS_X_test_scaled)
            EW_pred = EW_model.predict(EW_X_test_scaled)

            # R²
            Z_r = Z_model.score(Z_X_test_scaled, Z_y_test)
            NS_r = NS_model.score(NS_X_test_scaled, NS_y_test)
            EW_r = EW_model.score(EW_X_test_scaled, EW_y_test)
            
            # Store results
            station_results.append({
                'fmin': f1,
                'fmax': f2,
                'Z_r2': Z_r,
                'NS_r2': NS_r,
                'EW_r2': EW_r,
                'avg_r2': (Z_r + NS_r + EW_r) / 3,
                'Z_y_test': Z_y_test,
                'NS_y_test': NS_y_test,
                'EW_y_test': EW_y_test,
                'Z_pred': Z_pred,
                'NS_pred': NS_pred,
                'EW_pred': EW_pred,
                'Z_cv_r2': scores_Z.mean(),
                'NS_cv_r2': scores_NS.mean(),
                'EW_cv_r2': scores_EW.mean(),
                'avg_cv_r2': (scores_Z.mean() + scores_NS.mean() + scores_EW.mean()) / 3,
                'Z_cv_std': scores_Z.std(),
                'NS_cv_std': scores_NS.std(),
                'EW_cv_std': scores_EW.std(),
                'avg_cv_std': (scores_Z.std() + scores_NS.std() + scores_EW.std()) / 3 })
        
        
        # Find best result
        # R²
        best_r_result = max(station_results, key=lambda x: x['avg_r2'])
        best_r_results.append({'station': station, **best_r_result})
        # Cross Validation R²
        best_cv_r_result = max(station_results, key=lambda x: x['avg_cv_r2'])
        best_cv_r_results.append({'station': station, **best_cv_r_result})
    
        # Print best result
        # R²
        print(f"\nBest R² value frequency range for {station}: {best_r_result['fmin']} - {best_r_result['fmax']} Hz")
        print(f"Z R²: {best_r_result['Z_r2']:.4f}")
        print(f"NS R²: {best_r_result['NS_r2']:.4f}")
        print(f"EW R²: {best_r_result['EW_r2']:.4f}")
        print(f"Average R²: {best_r_result['avg_r2']:.4f}")
        # Cross Validation R²
        print(f"\nBest cv R² value frequency range for {station}: {best_cv_r_result['fmin']} - {best_cv_r_result['fmax']} Hz")
        print(f"Z cv R²: {best_cv_r_result['Z_cv_r2']:.4f} + {best_cv_r_result['Z_cv_std']:.4f}")
        print(f"NS cv R²: {best_cv_r_result['NS_cv_r2']:.4f} + {best_cv_r_result['NS_cv_std']:.4f}")
        print(f"EW cv R²: {best_cv_r_result['EW_cv_r2']:.4f} + {best_cv_r_result['EW_cv_std']:.4f}")
        print(f"Average cv R²: {best_cv_r_result['avg_cv_r2']:.4f}")
    
        results.append(station_results)

        # Plot all average R² and rmse results against centre frequency
        if plot_stat_results == True:
            band_centres = []
            r2_list = []
            for r in station_results:
                band_centre = [(r['fmin'] + r['fmax'])/2]
                r2 = r['avg_r2']
                band_centres.append(band_centre)
                r2_list.append(r2)
            
            # R²
            plt.scatter(band_centres, r2_list)
            plt.title(f'{station}: Band Centre Frequency vs Average R²')
            plt.xlabel('Band Centre Frequency (Hz)')
            plt.ylabel('Average R²')
            
        # Plot the best R² and rmse results for each station channel
        if plot_best == True:
            
            fig, axs = plt.subplots(3, 1, figsize=(10, 10))
            # R²
            axs[0].scatter(best_r_result['Z_y_test'],best_r_result['Z_pred'], marker='x')
            lims = [min(axs[0].get_xlim()[0], axs[0].get_ylim()[0]), max(axs[0].get_xlim()[1], axs[0].get_ylim()[1])]
            axs[0].plot(lims, lims, 'r--')
            axs[0].set_title(f"Best R² Z: {best_r_result['Z_r2']:.4f}")

            axs[1].scatter(best_r_result['NS_y_test'],best_r_result['NS_pred'], marker='x')
            lims = [min(axs[1].get_xlim()[0], axs[1].get_ylim()[0]), max(axs[1].get_xlim()[1], axs[1].get_ylim()[1])]
            axs[1].plot(lims, lims, 'r--')
            axs[1].set_title(f"Best R² NS: {best_r_result['NS_r2']:.4f}")

            axs[2].scatter(best_r_result['EW_y_test'],best_r_result['EW_pred'], marker='x')
            lims = [min(axs[2].get_xlim()[0], axs[2].get_ylim()[0]), max(axs[2].get_xlim()[1], axs[2].get_ylim()[1])]
            axs[2].plot(lims, lims, 'r--')           
            axs[2].set_title(f"Best R² EW: {best_r_result['EW_r2']:.4f}")

            # Figure Labels
            if variable_units is None: 
                fig.supxlabel('Observed AWS Measurement')
                fig.supylabel('Predicted AWS Measurement')
            else:
                fig.supxlabel(f'Observed {variable_units}')
                fig.supylabel(f'Predicted {variable_units}')
            if plot_title is None:
                title = f'{station}: Observed AWS Measurement vs Predicted AWS Measured'
            else:
                title = plot_title 

            plt.suptitle(title)

        # Save to csv file setup
        if csv == True:
            for result in station_results:
                all_results.append({
                    'station': station,
                    'fmin': result['fmin'],
                    'fmax': result['fmax'],
                    'Z_r2': result['Z_r2'],
                    'NS_r2': result['NS_r2'],
                    'EW_r2': result['EW_r2'],
                    'avg_r2': result['avg_r2']})

    # Save all results to csv file
    if csv == True:
        df = pd.DataFrame(all_results)
        csv_name = f"{csv_title}.csv"
        df.to_csv(csv_name, index=False)
        print(f"Results saved to {csv_name}")
                                
    return best_r_results, results


def single_pca_ridge(spectra,
                   fmin = 1,
                   fmax = 49,
                   f_band_width = 1,
                   step_size = 1,
                   csv = False,
                   csv_title = 'results',
                   plot_best = True,
                   plot_title = None,
                   variable_units = None,
                   plot_stat_results = True):
    
    """
    Finds the correlation between seismic data and AWS wind speed 
    via a PCA and Ridge Regression. 
    Only individual frequency bands inspect.

    Parameters:
        spectra (list):
            A list of dictionaries containing the frequency, power, and aws data 
            for each seismic component (EW, NS, Z) for each station and time period.
        fmin (int):
            Minimum frequency value for calculating the power of each bandwidths.
        fmax (int):
            Maximum frequency value for calculating the power of each  bandwidths.
        f_band_width (int):
            Bandwidth size. e.g. f_band_width = 1 for (f1,f2)=(1,2), 2 for (1,3), 3 for (1,4). 
        step_size (int):
            Frequency band step size. Set to less than f_band_width for overlapping bands. 
        csv (bool):
            True/False. True to save returns as a csv file.
        csv_title (str):
            Title for the output csv file.
        plot_best (bool):
            True/False. Plots the best R² and rmse for each station channel.
        plot_title (str):
            Title for plot_best plots.
        variable_units (str):
            AWS variable label, e.g 'wind speed (m/s)'
        plot_stat_results (bool):
            Plots all the R² and rmse values against frequency bandwidth centres.

    Outputs:
        best_r_results (list):
            A list of dictionaries containing information about the best (max) r2 correlation result for each station.
        best_rmse_results (list):
            A list of dictionaries containing information about the best (min) rmse correlation result for each station.
        results (list):
            A list of dictionaries containing information about all the correlation results for each station.
    """
    
    
    # Setup Result Lists
    results = []
    all_results = []
    best_r_results = []
    best_cv_r_results = []

    # Create Bandwidths
    bands = []
    for f1 in range(fmin, fmax - f_band_width + 1, step_size):
        f2 = f1 + f_band_width
        band = (f1, f2)
        bands.append(band)

    # Loop through stations
    for station_dict in spectra:

        # Setup Variables
        station = list(station_dict.keys())[0] 
        EW_power = station_dict[station][0]['EW']
        NS_power = station_dict[station][0]['NS']
        Z_power = station_dict[station][0]['Z']
        freq = station_dict[station][0]['freq']
        aws_values = station_dict[station][0]['aws_values']

        # Setup results
        station_results = []

        # Apply bandwidths to data
        for i, (f1,f2) in enumerate(bands):
            band_width = (freq >= f1) & (freq < f2)

            # Convert to log to better inspect power scales and apply bandwidth
            # [:, band_width], select frequencies and slice unwanted freq data from the row
            # .mean(axis=1), mean for the selected frequency row. Reshape for model input.
            Z_log = np.log10(Z_power[:, band_width].mean(axis=1) + 1e-20).reshape(-1,1)
            NS_log = np.log10(NS_power[:, band_width].mean(axis=1) + 1e-20).reshape(-1,1)
            EW_log = np.log10(EW_power[:, band_width].mean(axis=1) + 1e-20).reshape(-1,1)
     
            # Train Model
            Z_X_train, Z_X_test, Z_y_train, Z_y_test = train_test_split(Z_log, aws_values, test_size=0.2,random_state = 42)
            NS_X_train, NS_X_test, NS_y_train, NS_y_test = train_test_split(NS_log, aws_values, test_size=0.2,random_state = 42)
            EW_X_train, EW_X_test, EW_y_train, EW_y_test = train_test_split(EW_log, aws_values, test_size=0.2,random_state = 42)
            
            # Scale 
            Z_scaler = StandardScaler()
            NS_scaler = StandardScaler()
            EW_scaler = StandardScaler()
            Z_X_train_scaled = Z_scaler.fit_transform(Z_X_train)
            Z_X_test_scaled = Z_scaler.transform(Z_X_test)
            NS_X_train_scaled = NS_scaler.fit_transform(NS_X_train)
            NS_X_test_scaled = NS_scaler.transform(NS_X_test)
            EW_X_train_scaled = EW_scaler.fit_transform(EW_X_train)
            EW_X_test_scaled = EW_scaler.transform(EW_X_test)

            # PCA
            pca_Z = PCA(n_components=0.95)
            pca_NS = PCA(n_components=0.95)
            pca_EW = PCA(n_components=0.95)
            Z_X_train_pca = pca_Z.fit_transform(Z_X_train_scaled)
            Z_X_test_pca = pca_Z.transform(Z_X_test_scaled)
            NS_X_train_pca = pca_NS.fit_transform(NS_X_train_scaled)
            NS_X_test_pca = pca_NS.transform(NS_X_test_scaled)
            EW_X_train_pca = pca_EW.fit_transform(EW_X_train_scaled)
            EW_X_test_pca = pca_EW.transform(EW_X_test_scaled)
            
            # Ridge Model
            Z_model = Ridge(alpha=1.0, solver='auto')
            NS_model = Ridge(alpha=1.0, solver='auto')
            EW_model = Ridge(alpha=1.0, solver='auto')
            
            # Cross Validation
            scores_Z = cross_val_score(Z_model, Z_log, aws_values, cv=5, scoring='r2')
            scores_NS = cross_val_score(NS_model, NS_log, aws_values, cv=5, scoring='r2')
            scores_EW = cross_val_score(EW_model, EW_log, aws_values, cv=5, scoring='r2')

            # Fit Model
            Z_model.fit(Z_X_train_pca, Z_y_train)
            NS_model.fit(NS_X_train_pca, NS_y_train)
            EW_model.fit(EW_X_train_pca, EW_y_train)

            # Predictions
            Z_pred = Z_model.predict(Z_X_test_pca)
            NS_pred = NS_model.predict(NS_X_test_pca)
            EW_pred = EW_model.predict(EW_X_test_pca)

            # R²
            Z_r = Z_model.score(Z_X_test_scaled, Z_y_test)
            NS_r = NS_model.score(NS_X_test_scaled, NS_y_test)
            EW_r = EW_model.score(EW_X_test_scaled, EW_y_test)
            
            # Store results
            station_results.append({
                'fmin': f1,
                'fmax': f2,
                'Z_r2': Z_r,
                'NS_r2': NS_r,
                'EW_r2': EW_r,
                'avg_r2': (Z_r + NS_r + EW_r) / 3,
                'Z_y_test': Z_y_test,
                'NS_y_test': NS_y_test,
                'EW_y_test': EW_y_test,
                'Z_pred': Z_pred,
                'NS_pred': NS_pred,
                'EW_pred': EW_pred,
                'Z_cv_r2': scores_Z.mean(),
                'NS_cv_r2': scores_NS.mean(),
                'EW_cv_r2': scores_EW.mean(),
                'avg_cv_r2': (scores_Z.mean() + scores_NS.mean() + scores_EW.mean()) / 3,
                'Z_cv_std': scores_Z.std(),
                'NS_cv_std': scores_NS.std(),
                'EW_cv_std': scores_EW.std(),
                'avg_cv_std': (scores_Z.std() + scores_NS.std() + scores_EW.std()) / 3 })
        
        # Find best result
        # R²
        best_r_result = max(station_results, key=lambda x: x['avg_r2'])
        best_r_results.append({'station': station, **best_r_result})
        # Cross Validation R²
        best_cv_r_result = max(station_results, key=lambda x: x['avg_cv_r2'])
        best_cv_r_results.append({'station': station, **best_cv_r_result})
    
        # Print best result
        # R²
        print(f"\nBest R² value frequency range for {station}: {best_r_result['fmin']} - {best_r_result['fmax']} Hz")
        print(f"Z R²: {best_r_result['Z_r2']:.4f}")
        print(f"NS R²: {best_r_result['NS_r2']:.4f}")
        print(f"EW R²: {best_r_result['EW_r2']:.4f}")
        print(f"Average R²: {best_r_result['avg_r2']:.4f}")
        # Cross Validation R²
        print(f"\nBest cv R² value frequency range for {station}: {best_cv_r_result['fmin']} - {best_cv_r_result['fmax']} Hz")
        print(f"Z cv R²: {best_cv_r_result['Z_cv_r2']:.4f} + {best_cv_r_result['Z_cv_std']:.4f}")
        print(f"NS cv R²: {best_cv_r_result['NS_cv_r2']:.4f} + {best_cv_r_result['NS_cv_std']:.4f}")
        print(f"EW cv R²: {best_cv_r_result['EW_cv_r2']:.4f} + {best_cv_r_result['EW_cv_std']:.4f}")
        print(f"Average cv R²: {best_cv_r_result['avg_cv_r2']:.4f}")
        
        results.append(station_results)

        # Plot all average R² and rmse results against centre frequency
        if plot_stat_results == True:
            band_centres = []
            r2_list = []
            for r in station_results:
                band_centre = [(r['fmin'] + r['fmax'])/2]
                r2 = r['avg_r2']
                band_centres.append(band_centre)
                r2_list.append(r2)
            
            # R²
            plt.scatter(band_centres, r2_list)
            plt.title(f'{station}: Band Centre Frequency vs Average R²')
            plt.xlabel('Band Centre Frequency (Hz)')
            plt.ylabel('Average R²')
            
        # Plot the best R² and rmse results for each station channel
        if plot_best == True:
            
            fig, axs = plt.subplots(3, 1, figsize=(10, 10))
            # R²
            axs[0].scatter(best_r_result['Z_y_test'],best_r_result['Z_pred'], marker='x')
            lims = [min(axs[0].get_xlim()[0], axs[0].get_ylim()[0]), max(axs[0].get_xlim()[1], axs[0].get_ylim()[1])]
            axs[0].plot(lims, lims, 'r--')
            axs[0].set_title(f"Best R² Z: {best_r_result['Z_r2']:.4f}")

            axs[1].scatter(best_r_result['NS_y_test'],best_r_result['NS_pred'], marker='x')
            lims = [min(axs[1].get_xlim()[0], axs[1].get_ylim()[0]), max(axs[1].get_xlim()[1], axs[1].get_ylim()[1])]
            axs[1].plot(lims, lims, 'r--')
            axs[1].set_title(f"Best R² NS: {best_r_result['NS_r2']:.4f}")

            axs[2].scatter(best_r_result['EW_y_test'],best_r_result['EW_pred'], marker='x')
            lims = [min(axs[2].get_xlim()[0], axs[2].get_ylim()[0]), max(axs[2].get_xlim()[1], axs[2].get_ylim()[1])]
            axs[2].plot(lims, lims, 'r--')           
            axs[2].set_title(f"Best R² EW: {best_r_result['EW_r2']:.4f}")

            # Figure Labels
            if variable_units is None: 
                fig.supxlabel('Observed AWS Measurement')
                fig.supylabel('Predicted AWS Measurement')
            else:
                fig.supxlabel(f'Observed {variable_units}')
                fig.supylabel(f'Predicted {variable_units}')
            if plot_title is None:
                title = f'{station}: Observed AWS Measurement vs Predicted AWS Measured'
            else:
                title = plot_title 

            plt.suptitle(title)

        # Save to csv file setup
        if csv == True:
            for result in station_results:
                all_results.append({
                    'station': station,
                    'fmin': result['fmin'],
                    'fmax': result['fmax'],
                    'Z_r2': result['Z_r2'],
                    'NS_r2': result['NS_r2'],
                    'EW_r2': result['EW_r2'],
                    'avg_r2': result['avg_r2']})

    # Save all results to csv file
    if csv == True:
        df = pd.DataFrame(all_results)
        csv_name = f"{csv_title}.csv"
        df.to_csv(csv_name, index=False)
        print(f"Results saved to {csv_name}")
                                
    return best_r_results, results

def full_pca_ridge(spectra,
                   fmin = 1,
                   fmax = 49,
                   csv = False,
                   csv_title = 'results',
                   plot_stat_results = True):
    
    """
    Finds the correlation between seismic data and AWS wind speed 
    via a PCA and Ridge Regression. Full inspection.

    Parameters:
        spectra (list):
            A list of dictionaries containing the frequency, power, and aws data 
            for each seismic component (EW, NS, Z) for each station and time period.
        fmin (int):
            Minimum frequency value for calculating the power of each bandwidths.
        fmax (int):
            Maximum frequency value for calculating the power of each  bandwidths.
        plot_stat_results (bool):
            Plots all the R² and rmse values against frequency bandwidth centres.

    Outputs:
        results (list):
            A list of dictionaries containing information about all the correlation results for each station.
    """
    
    
    # Setup Result Lists
    results = []
    all_results = []

    # Loop through stations
    for station_dict in spectra:

        # Setup Variables
        station = list(station_dict.keys())[0] 
        EW_power = station_dict[station][0]['EW']
        NS_power = station_dict[station][0]['NS']
        Z_power = station_dict[station][0]['Z']
        freq = station_dict[station][0]['freq']
        aws_values = station_dict[station][0]['aws_values']

        # Setup results
        station_results = []

        # Apply bandwidths to data
        #for i, (f1,f2) in enumerate(bands):
        band_width = (freq >= fmin) & (freq < fmax)

        # Convert to log to better inspect power scales and apply bandwidth
        # [:, band_width], select frequencies and slice unwanted freq data from the row
        # .mean(axis=1), mean for the selected frequency row. Reshape for model input.
        Z_log = np.log10(Z_power[:, band_width] + 1e-20)
        NS_log = np.log10(NS_power[:, band_width] + 1e-20)
        EW_log = np.log10(EW_power[:, band_width] + 1e-20)
    
        # Train Model
        Z_X_train, Z_X_test, Z_y_train, Z_y_test = train_test_split(Z_log, aws_values, test_size=0.2,random_state = 42)
        NS_X_train, NS_X_test, NS_y_train, NS_y_test = train_test_split(NS_log, aws_values, test_size=0.2,random_state = 42)
        EW_X_train, EW_X_test, EW_y_train, EW_y_test = train_test_split(EW_log, aws_values, test_size=0.2,random_state = 42)
        
        # Pipeline
        # Scale, PCA, Ridge
        Z_model = Pipeline([('scaler', StandardScaler()),('pca', PCA(n_components=0.95)),('ridge', Ridge(alpha=1.0))])
        NS_model = Pipeline([('scaler', StandardScaler()),('pca', PCA(n_components=0.95)),('ridge', Ridge(alpha=1.0))])
        EW_model = Pipeline([('scaler', StandardScaler()),('pca', PCA(n_components=0.95)),('ridge', Ridge(alpha=1.0))])

        # Cross Validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores_Z = cross_val_score(Z_model, Z_log, aws_values, cv=cv, scoring='r2')
        scores_NS = cross_val_score(NS_model, NS_log, aws_values, cv=cv, scoring='r2')
        scores_EW = cross_val_score(EW_model, EW_log, aws_values, cv=cv, scoring='r2')

        # fit model
        Z_model.fit(Z_X_train, Z_y_train)
        NS_model.fit(NS_X_train, NS_y_train)
        EW_model.fit(EW_X_train, EW_y_train)

        # Predictions
        Z_pred = Z_model.predict(Z_X_test)
        NS_pred = NS_model.predict(NS_X_test)
        EW_pred = EW_model.predict(EW_X_test)

        # coeffs
        pca_Z = Z_model.named_steps['pca']
        ridge_Z = Z_model.named_steps['ridge']
        pca_NS = NS_model.named_steps['pca']
        ridge_NS = NS_model.named_steps['ridge']
        pca_EW = EW_model.named_steps['pca']
        ridge_EW = EW_model.named_steps['ridge']

        # R²
        Z_r = Z_model.score(Z_X_test, Z_y_test)
        NS_r = NS_model.score(NS_X_test, NS_y_test)
        EW_r = EW_model.score(EW_X_test, EW_y_test)
        # Features
        Z_frequency_importance = np.abs(pca_Z.components_.T @ ridge_Z.coef_)
        NS_frequency_importance = np.abs(pca_NS.components_.T @ ridge_NS.coef_)
        EW_frequency_importance = np.abs(pca_EW.components_.T @ ridge_EW.coef_)

        # Store results
        station_results.append({
            'fmin': fmin,
            'fmax': fmax,
            'Z_r2': Z_r,
            'NS_r2': NS_r,
            'EW_r2': EW_r,
            'avg_r2': (Z_r + NS_r + EW_r) / 3,
            'Z_y_test': Z_y_test,
            'NS_y_test': NS_y_test,
            'EW_y_test': EW_y_test,
            'Z_pred': Z_pred,
            'NS_pred': NS_pred,
            'EW_pred': EW_pred,
            'Z_cv_r2': scores_Z.mean(),
            'NS_cv_r2': scores_NS.mean(),
            'EW_cv_r2': scores_EW.mean(),
            'avg_cv_r2': (scores_Z.mean() + scores_NS.mean() + scores_EW.mean()) / 3,
            'Z_cv_std': scores_Z.std(),
            'NS_cv_std': scores_NS.std(),
            'EW_cv_std': scores_EW.std(),
            'avg_cv_std': (scores_Z.std() + scores_NS.std() + scores_EW.std()) / 3 })
    
        # Print best result
        # R²
        print(f"Z R²: {Z_r:.4f}")
        print(f"NS R²: {NS_r:.4f}")
        print(f"EW R²: {EW_r:.4f}")
        print(f"Average R²: {(Z_r + NS_r + EW_r) / 3:.4f}")
        # Cross Validation R²
        print(f"Z cv R²: {scores_Z.mean():.4f} + {scores_Z.std():.4f}")
        print(f"NS cv R²: {scores_NS.mean():.4f} + {scores_NS.std():.4f}")
        print(f"EW cv R²: {scores_EW.mean():.4f} + {scores_EW.std():.4f}")
        print(f"Average cv R²: {(scores_Z.mean() + scores_NS.mean() + scores_EW.mean()) / 3:.4f} +/- {(scores_Z.std() + scores_NS.std() + scores_EW.std()) / 3:.4f}")
        
        results.append(station_results)

        # Plot all average R² and rmse results against centre frequency
        if plot_stat_results == True:
            
            # R²
            fig, axs = plt.subplots(3, 1, figsize=(10, 10))
            axs[0].scatter(freq[band_width], Z_frequency_importance)
            axs[0].set_title('Z')
            axs[1].scatter(freq[band_width], NS_frequency_importance)
            axs[1].set_title('NS')
            axs[2].scatter(freq[band_width], EW_frequency_importance)
            axs[2].set_title('EW')

            plt.suptitle(f'{station}:')
            fig.supxlabel('Frequency (Hz)')
            fig.supylabel('PCA + Ridge Contribution')
            
        # Save to csv file setup
        if csv == True:
            for result in station_results:
                all_results.append({
                    'station': station,
                    'fmin': result['fmin'],
                    'fmax': result['fmax'],
                    'Z_r2': result['Z_r2'],
                    'NS_r2': result['NS_r2'],
                    'EW_r2': result['EW_r2'],
                    'avg_r2': result['avg_r2']})

    # Save all results to csv file
    if csv == True:
        df = pd.DataFrame(all_results)
        csv_name = f"{csv_title}.csv"
        df.to_csv(csv_name, index=False)
        print(f"Results saved to {csv_name}")
                                
    return results


def full_svr(spectra,
             fmin = 1,
             fmax = 49,
             f_band_width = 1,
             step_size = 1,
             csv = False,
             csv_title = 'results',
             plot_stat_results = True):
    
    """
    Finds the correlation between seismic data and AWS wind speed 
    via a SVR Regression. Full inspection  w/ bandwidths.

    Parameters:
        spectra (list):
            A list of dictionaries containing the frequency, power, and aws data 
            for each seismic component (EW, NS, Z) for each station and time period.
        fmin (int):
            Minimum frequency value for calculating the power of each bandwidths.
        fmax (int):
            Maximum frequency value for calculating the power of each  bandwidths.
        csv (bool):
            True/False. True to save returns as a csv file.
        csv_title (str):
            Title for the output csv file.
        plot_stat_results (bool):
            Plots all the R² and rmse values against frequency bandwidth centres.

    Outputs:
        results (list):
            A list of dictionaries containing information about all the correlation results for each station.
    """
    
    
    # Setup Result Lists
    results = []
    all_results = []

    # Create Bandwidths
    bands = []
    for f1 in range(fmin, fmax - f_band_width + 1, step_size):
        f2 = f1 + f_band_width
        band = (f1, f2)
        bands.append(band)

    # Loop through stations
    for station_dict in spectra:

        # Setup Variables
        station = list(station_dict.keys())[0] 
        EW_power = station_dict[station][0]['EW']
        NS_power = station_dict[station][0]['NS']
        Z_power = station_dict[station][0]['Z']
        freq = station_dict[station][0]['freq']
        aws_values = station_dict[station][0]['aws_values']

        # Setup results
        station_results = []

        # Setup powers
        X_Z = np.zeros((len(aws_values), len(bands)))
        X_NS = np.zeros((len(aws_values), len(bands)))
        X_EW = np.zeros((len(aws_values), len(bands)))


        for i, (f1, f2) in enumerate(bands):
            # Apply bandwidths to data
            band_width = (freq >= f1) & (freq < f2)
            # Convert to log to better inspect power scales and apply bandwidth
                    # [:, band_width], select frequencies and slice unwanted freq data from the row
                    # .mean(axis=1), mean for the selected frequency row. Reshape for model input.
            X_Z[:, i] = np.log10(Z_power[:, band_width].mean(axis=1) + 1e-20)
            X_NS[:, i] = np.log10(NS_power[:, band_width].mean(axis=1) + 1e-20)
            X_EW[:, i] = np.log10(EW_power[:, band_width].mean(axis=1) + 1e-20)

    
        # Train Model
        Z_X_train, Z_X_test, Z_y_train, Z_y_test = train_test_split(X_Z, aws_values, test_size=0.2,random_state = 42)
        NS_X_train, NS_X_test, NS_y_train, NS_y_test = train_test_split(X_NS, aws_values, test_size=0.2,random_state = 42)
        EW_X_train, EW_X_test, EW_y_train, EW_y_test = train_test_split(X_EW, aws_values, test_size=0.2,random_state = 42)
        
        # Pipeline
        # Scale, SVR
        # (Need to find optimal values)
        Z_model = Pipeline([('scaler', StandardScaler()),('svr', SVR(C=1.0, epsilon=0.2))])
        NS_model = Pipeline([('scaler', StandardScaler()),('svr', SVR(C=1.0, epsilon=0.2))])
        EW_model = Pipeline([('scaler', StandardScaler()),('svr', SVR(C=1.0, epsilon=0.2))])

        # Cross Validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores_Z = cross_val_score(Z_model, X_Z, aws_values, cv=cv, scoring='r2')
        scores_NS = cross_val_score(NS_model, X_NS, aws_values, cv=cv, scoring='r2')
        scores_EW = cross_val_score(EW_model, X_EW, aws_values, cv=cv, scoring='r2')

        # Fit Model
        Z_model.fit(Z_X_train, Z_y_train)
        NS_model.fit(NS_X_train, NS_y_train)
        EW_model.fit(EW_X_train, EW_y_train)

        # Predictions
        Z_pred = Z_model.predict(Z_X_test)
        NS_pred = NS_model.predict(NS_X_test)
        EW_pred = EW_model.predict(EW_X_test)

        # R²
        Z_r = Z_model.score(Z_X_test, Z_y_test)
        NS_r = NS_model.score(NS_X_test, NS_y_test)
        EW_r = EW_model.score(EW_X_test, EW_y_test)
        # Freq Importance
        Z_importance = permutation_importance(Z_model, Z_X_test, Z_y_test, n_repeats=3, scoring='r2').importances_mean
        NS_importance = permutation_importance(NS_model, NS_X_test, NS_y_test, n_repeats=3, scoring='r2').importances_mean
        EW_importance = permutation_importance(EW_model, EW_X_test, EW_y_test, n_repeats=3, scoring='r2').importances_mean

        # Store results
        station_results.append({
            'fmin': f1,
            'fmax': f2,
            'Z_r2': Z_r,
            'NS_r2': NS_r,
            'EW_r2': EW_r,
            'avg_r2': (Z_r + NS_r + EW_r) / 3,
            'Z_y_test': Z_y_test,
            'NS_y_test': NS_y_test,
            'EW_y_test': EW_y_test,
            'Z_pred': Z_pred,
            'NS_pred': NS_pred,
            'EW_pred': EW_pred,
            'Z_cv_r2': scores_Z.mean(),
            'NS_cv_r2': scores_NS.mean(),
            'EW_cv_r2': scores_EW.mean(),
            'avg_cv_r2': (scores_Z.mean() + scores_NS.mean() + scores_EW.mean()) / 3,
            'Z_cv_std': scores_Z.std(),
            'NS_cv_std': scores_NS.std(),
            'EW_cv_std': scores_EW.std(),
            'avg_cv_std': (scores_Z.std() + scores_NS.std() + scores_EW.std()) / 3 })
    
        # Print best result
        # R²
        print(f"Z R²: {Z_r:.4f}")
        print(f"NS R²: {NS_r:.4f}")
        print(f"EW R²: {EW_r:.4f}")
        print(f"Average R²: {(Z_r + NS_r + EW_r) / 3:.4f}")
        # Cross Validation R²
        print(f"Z cv R²: {scores_Z.mean():.4f} + {scores_Z.std():.4f}")
        print(f"NS cv R²: {scores_NS.mean():.4f} + {scores_NS.std():.4f}")
        print(f"EW cv R²: {scores_EW.mean():.4f} + {scores_EW.std():.4f}")
        print(f"Average cv R²: {(scores_Z.mean() + scores_NS.mean() + scores_EW.mean()) / 3:.4f} +/- {(scores_Z.std() + scores_NS.std() + scores_EW.std()) / 3:.4f}")
        
        results.append(station_results)

        # Plot all average R² and rmse results against centre frequency
        if plot_stat_results == True:
            band_centres = []
            for f1, f2 in bands:
                band_centre = [(f1 + f2)/2]
                band_centres.append(band_centre)

            # R²
            fig, axs = plt.subplots(3, 1, figsize=(10, 10))
            axs[0].scatter(band_centres, Z_importance)
            axs[0].set_title('Z')
            axs[1].scatter(band_centres, NS_importance)
            axs[1].set_title('NS')
            axs[2].scatter(band_centres, EW_importance)
            axs[2].set_title('EW')

            plt.suptitle(f'{station}:')
            fig.supxlabel('Frequency (Hz)')
            fig.supylabel('SVR Permutation Importance')

        # Save to csv file setup
        if csv == True:
            for result in station_results:
                all_results.append({
                    'station': station,
                    'fmin': result['fmin'],
                    'fmax': result['fmax'],
                    'Z_r2': result['Z_r2'],
                    'NS_r2': result['NS_r2'],
                    'EW_r2': result['EW_r2'],
                    'avg_r2': result['avg_r2']})

    # Save all results to csv file
    if csv == True:
        df = pd.DataFrame(all_results)
        csv_name = f"{csv_title}.csv"
        df.to_csv(csv_name, index=False)
        print(f"Results saved to {csv_name}")
                                
    return results


def full_elasticnet(spectra,
                    fmin = 1,
                    fmax = 49,
                    f_band_width = 1,
                    step_size = 1,
                    csv = False,
                    csv_title = 'results',
                    plot_stat_results = True):
    
    """
    Finds the correlation between seismic data and AWS wind speed 
    via a ElasticNet Regression. Full inspection w/ bandwidths.

    Parameters:
        spectra (list):
            A list of dictionaries containing the frequency, power, and aws data 
            for each seismic component (EW, NS, Z) for each station and time period.
        fmin (int):
            Minimum frequency value for calculating the power of each bandwidths.
        fmax (int):
            Maximum frequency value for calculating the power of each  bandwidths.
        f_band_width (int):
            Bandwidth size. e.g. f_band_width = 1 for (f1,f2)=(1,2), 2 for (1,3), 3 for (1,4). 
        step_size (int):
            Frequency band step size. Set to less than f_band_width for overlapping bands. 
        csv (bool):
            True/False. True to save returns as a csv file.
        csv_title (str):
            Title for the output csv file.
        plot_best (bool):
            True/False. Plots the best R² and rmse for each station channel.
        plot_title (str):
            Title for plot_best plots.
        variable_units (str):
            AWS variable label, e.g 'wind speed (m/s)'
        plot_stat_results (bool):
            Plots all the R² and rmse values against frequency bandwidth centres.

    Outputs:
        best_r_results (list):
            A list of dictionaries containing information about the best (max) r2 correlation result for each station.
        best_rmse_results (list):
            A list of dictionaries containing information about the best (min) rmse correlation result for each station.
        results (list):
            A list of dictionaries containing information about all the correlation results for each station.
    """
    
    
    # Setup Result Lists
    results = []
    all_results = []

    # Create Bandwidths
    bands = []
    for f1 in range(fmin, fmax - f_band_width + 1, step_size):
        f2 = f1 + f_band_width
        band = (f1, f2)
        bands.append(band)

    # Loop through stations
    for station_dict in spectra:

        # Setup Variables
        station = list(station_dict.keys())[0] 
        EW_power = station_dict[station][0]['EW']
        NS_power = station_dict[station][0]['NS']
        Z_power = station_dict[station][0]['Z']
        freq = station_dict[station][0]['freq']
        aws_values = station_dict[station][0]['aws_values']

        # Setup results
        station_results = []

        # Setup powers
        X_Z = np.zeros((len(aws_values), len(bands)))
        X_NS = np.zeros((len(aws_values), len(bands)))
        X_EW = np.zeros((len(aws_values), len(bands)))


        for i, (f1, f2) in enumerate(bands):
            # Apply bandwidths to data
            band_width = (freq >= f1) & (freq < f2)
            # Convert to log to better inspect power scales and apply bandwidth
                    # [:, band_width], select frequencies and slice unwanted freq data from the row
                    # .mean(axis=1), mean for the selected frequency row. Reshape for model input.
            X_Z[:, i] = np.log10(Z_power[:, band_width].mean(axis=1) + 1e-20)
            X_NS[:, i] = np.log10(NS_power[:, band_width].mean(axis=1) + 1e-20)
            X_EW[:, i] = np.log10(EW_power[:, band_width].mean(axis=1) + 1e-20)

    
        # Train Model
        Z_X_train, Z_X_test, Z_y_train, Z_y_test = train_test_split(X_Z, aws_values, test_size=0.2,random_state = 42)
        NS_X_train, NS_X_test, NS_y_train, NS_y_test = train_test_split(X_NS, aws_values, test_size=0.2,random_state = 42)
        EW_X_train, EW_X_test, EW_y_train, EW_y_test = train_test_split(X_EW, aws_values, test_size=0.2,random_state = 42)

        # Pipeline
        # Scale, ElasticNet (Need to find optimal l1 still)
        Z_model = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNet(alpha=0.08, l1_ratio=0.5))])
        NS_model = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNet(alpha=0.08, l1_ratio=0.5))])
        EW_model = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNet(alpha=0.08, l1_ratio=0.5))])

        # Cross Validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores_Z = cross_val_score(Z_model, X_Z, aws_values, cv=cv, scoring='r2')
        scores_NS = cross_val_score(NS_model, X_NS, aws_values, cv=cv, scoring='r2')
        scores_EW = cross_val_score(EW_model, X_EW, aws_values, cv=cv, scoring='r2')

        # fit model
        Z_model.fit(Z_X_train, Z_y_train)
        NS_model.fit(NS_X_train, NS_y_train)
        EW_model.fit(EW_X_train, EW_y_train)

        # Predictions
        Z_pred = Z_model.predict(Z_X_test)
        NS_pred = NS_model.predict(NS_X_test)
        EW_pred = EW_model.predict(EW_X_test)

        # R²
        Z_r = Z_model.score(Z_X_test, Z_y_test)
        NS_r = NS_model.score(NS_X_test, NS_y_test)
        EW_r = EW_model.score(EW_X_test, EW_y_test)
        # Coeffs
        Z_coef = np.abs(Z_model.named_steps['enet'].coef_)
        NS_coef = np.abs(NS_model.named_steps['enet'].coef_)
        EW_coef = np.abs(EW_model.named_steps['enet'].coef_)

        
        # Store results
        station_results.append({
            'fmin': fmin,
            'fmax': fmax,
            'Z_r2': Z_r,
            'NS_r2': NS_r,
            'EW_r2': EW_r,
            'avg_r2': (Z_r + NS_r + EW_r) / 3,
            'Z_y_test': Z_y_test,
            'NS_y_test': NS_y_test,
            'EW_y_test': EW_y_test,
            'Z_pred': Z_pred,
            'NS_pred': NS_pred,
            'EW_pred': EW_pred,
            'Z_cv_r2': scores_Z.mean(),
            'NS_cv_r2': scores_NS.mean(),
            'EW_cv_r2': scores_EW.mean(),
            'avg_cv_r2': (scores_Z.mean() + scores_NS.mean() + scores_EW.mean()) / 3,
            'Z_cv_std': scores_Z.std(),
            'NS_cv_std': scores_NS.std(),
            'EW_cv_std': scores_EW.std(),
            'avg_cv_std': (scores_Z.std() + scores_NS.std() + scores_EW.std()) / 3 })
    
        # Print best result
        # R²
        print(f"Z R²: {Z_r:.4f}")
        print(f"NS R²: {NS_r:.4f}")
        print(f"EW R²: {EW_r:.4f}")
        print(f"Average R²: {(Z_r + NS_r + EW_r) / 3:.4f}")
        # Cross Validation R²
        print(f"Z cv R²: {scores_Z.mean():.4f} + {scores_Z.std():.4f}")
        print(f"NS cv R²: {scores_NS.mean():.4f} + {scores_NS.std():.4f}")
        print(f"EW cv R²: {scores_EW.mean():.4f} + {scores_EW.std():.4f}")
        print(f"Average cv R²: {(scores_Z.mean() + scores_NS.mean() + scores_EW.mean()) / 3:.4f} +/- {(scores_Z.std() + scores_NS.std() + scores_EW.std()) / 3:.4f}")
        
        results.append(station_results)

        # Plot all average R² and rmse results against centre frequency
        if plot_stat_results == True:
            band_centres = []
            for f1, f2 in bands:
                band_centre = [(f1 + f2)/2]
                band_centres.append(band_centre)

            # R²
            fig, axs = plt.subplots(3, 1, figsize=(10, 10))
            axs[0].scatter(band_centres, Z_coef)
            axs[0].set_title('Z')
            axs[1].scatter(band_centres, NS_coef)
            axs[1].set_title('NS')
            axs[2].scatter(band_centres, EW_coef)
            axs[2].set_title('EW')

            plt.suptitle(f'{station}:')
            fig.supxlabel('Frequency (Hz)')
            fig.supylabel('ElasticNet Coeff')
            
        # Save to csv file setup
        if csv == True:
            for result in station_results:
                all_results.append({
                    'station': station,
                    'fmin': result['fmin'],
                    'fmax': result['fmax'],
                    'Z_r2': result['Z_r2'],
                    'NS_r2': result['NS_r2'],
                    'EW_r2': result['EW_r2'],
                    'avg_r2': result['avg_r2']})

    # Save all results to csv file
    if csv == True:
        df = pd.DataFrame(all_results)
        csv_name = f"{csv_title}.csv"
        df.to_csv(csv_name, index=False)
        print(f"Results saved to {csv_name}")
                                
    return results