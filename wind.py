from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px
from obspy.signal.util import smooth


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
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        raise FileNotFoundError(f"No data file found for station {station_code} and id {id_code}")

    if "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(dict(
                                        year=df["Year Month Day Hour Minutes in YYYY.2"],
                                        month=df["MM.2"],
                                        day=df["DD.2"],
                                        hour=df["HH24.2"],
                                        minute=df["MI format in Universal coordinated time"]))

    return df


def plot_wind_speed(data, 
                    year=None, 
                    month=None, 
                    day=None,
                    hour=None,
                    apply_smooth = True,
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

    if isinstance(day, (tuple, list)) and len(day) == 2:
        start_day, end_day = day
        df_slice = df_slice[(df_slice['datetime'].dt.day >= start_day) & 
                            (df_slice['datetime'].dt.day <= end_day)].copy()
    elif isinstance(day, int):
        df_slice = df_slice[df_slice['datetime'].dt.day == day].copy()

    if isinstance(hour, (tuple, list)) and len(hour) == 2:
        start_hour, end_hour = hour
        df_slice = df_slice[(df_slice['datetime'].dt.hour >= start_hour) & 
                            (df_slice['datetime'].dt.hour <= end_hour)].copy()
    elif isinstance(hour, int):
        df_slice = df_slice[df_slice['datetime'].dt.hour == hour].copy()
    

    if df_slice.empty: 
        print("No data available for the selected time period.") 
        return

    # Convert to numeric
    df_slice['Wind speed in km/h'] = pd.to_numeric(df_slice['Wind speed in km/h'], 
                                                    errors='coerce')
    wind_speed = df_slice['Wind speed in km/h']
    df_slice['Wind_norm'] = wind_speed / wind_speed.max()

    # Smoothing
    if apply_smooth == True:
        wind_speed = smooth(wind_speed.to_numpy(), smoothie)
    else:
        wind_speed = wind_speed.to_numpy()
        
    plt.figure(figsize=(15,6))
    
    plt.plot(df_slice['datetime'], wind_speed, 
            color='black', linewidth=0.5)
    ymax = np.max(wind_speed)
    
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

    # Plotting 
    plt.ylabel('Wind Speed (km/h)')
    plt.xlabel('Time')
    plt.ylim(0, ymax + 10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_rose_wind(data, 
                   year=None, 
                   month=None, 
                   day=None,
                   hour=None):

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
        day (int or tuple): The day(s) for plotting.
                            None for the entire month,
                            A single day (1-31),
                            or a tuple of (start_day, end_day) for a range of days.
        hour (int or tuple):
            The hour(s) for plotting.
            None for entire day,
            A single hour (0-23),
            or a tuple of (start_hour, end_hour) for a range of hours.
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

    if isinstance(day, (tuple, list)) and len(day) == 2:
        start_day, end_day = day
        df_slice = df_slice[(df_slice['datetime'].dt.day >= start_day) & 
                            (df_slice['datetime'].dt.day <= end_day)].copy()
    elif isinstance(day, int):
        df_slice = df_slice[df_slice['datetime'].dt.day == day].copy()

    if isinstance(hour, (tuple, list)) and len(hour) == 2:
        start_hour, end_hour = hour
        df_slice = df_slice[(df_slice['datetime'].dt.hour >= start_hour) & 
                            (df_slice['datetime'].dt.hour <= end_hour)].copy()
    elif isinstance(hour, int):
        df_slice = df_slice[df_slice['datetime'].dt.hour == hour].copy()
    
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
    fig = px.bar_polar(freq, r="percentage", theta="dir_bin", color="speed_bin", color_continuous_scale=px.colors.sequential.Plasma)

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                tickformat=".0f%",
                ticksuffix="%",
                angle=90,
                side="counterclockwise"
            )
        )
    )

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
                         seismic_smoothie = 3):
    
    """
    Compares seismic data with wind data by plotting the North-South and East-West components of both datasets.
    Inspection is only useful for identical time periods but the process works for unrelated time periods. 
    For useful comparisons, trim the seismic data to the same time period you specify for the wind data.
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
    """
    
    if isinstance(wind_year, (tuple, list)) and len(wind_year) == 2: 
        start_year, end_year = wind_year 
        df_slice = wind_data[(wind_data['datetime'].dt.year >= start_year) & 
                        (wind_data['datetime'].dt.year <= end_year) ].copy()
    elif wind_year is None or wind_year < 2010 or wind_year > 2025:
        df_slice = wind_data.copy()
    else:
        df_slice = wind_data[wind_data['datetime'].dt.year == wind_year].copy()
        

    if isinstance(wind_month, (tuple, list)) and len(wind_month) == 2: 
        start_month, end_month = wind_month 
        df_slice = df_slice[(df_slice['datetime'].dt.month >= start_month) & 
                        (df_slice['datetime'].dt.month <= end_month) ].copy()
    elif isinstance(wind_month, int): 
        df_slice = df_slice[df_slice['datetime'].dt.month == wind_month].copy()

    if isinstance(wind_day, (tuple, list)) and len(wind_day) == 2:
        start_day, end_day = wind_day
        df_slice = df_slice[(df_slice['datetime'].dt.day >= start_day) & 
                            (df_slice['datetime'].dt.day <= end_day)].copy()
                            
    elif isinstance(wind_day, int):
        df_slice = df_slice[df_slice['datetime'].dt.day == wind_day].copy()

    if isinstance(wind_hour, (tuple, list)) and len(wind_hour) == 2:
        start_hour, end_hour = wind_hour
        df_slice = df_slice[(df_slice['datetime'].dt.hour >= start_hour) & 
                            (df_slice['datetime'].dt.hour <= end_hour)].copy()
    elif isinstance(wind_hour, int):
        df_slice = df_slice[df_slice['datetime'].dt.hour == wind_hour].copy()
    
    
    if df_slice.empty: 
        print("No wind data available for the selected time period.") 
        return

    # Convert to numeric
    df_slice['Wind speed in km/h'] = pd.to_numeric(df_slice['Wind speed in km/h'], 
                                                    errors='coerce')
    df_slice['Wind direction in degrees true'] = pd.to_numeric(df_slice['Wind direction in degrees true'],
                                                               errors='coerce')
    df_slice['Wind direction in degrees true'] %= 360

    wind_speed = df_slice['Wind speed in km/h']
    df_slice['Wind_norm'] = wind_speed / wind_speed.max()

    WD = df_slice['Wind direction in degrees true']
    #WS = df_slice['Wind_norm']
    WS = wind_speed

    u = []
    v = []
    for i in range(len(WS)):
    # North-South component
        u.append(-WS.iloc[i] * np.cos(np.deg2rad(WD.iloc[i])))
    # East-West component
        v.append(-WS.iloc[i] * np.sin(np.deg2rad(WD.iloc[i])))

    # Wind Smoothing
    if apply_wind_smooth == True:
        u = smooth(u, wind_smoothie)
        v = smooth(v, wind_smoothie)
    
    stations = list(seismic_data.keys())
    nrows = len(stations) + 1

    fig, ax = plt.subplots(nrows, 2, figsize=(6*nrows, 5))
    ymax = max(max(u), max(v))
    ymin = min(min(u), min(v))
    ax[0,0].set_ylim(ymin-(0.1*abs(ymin)), ymax+(0.1*abs(ymax)))
    ax[0,0].plot(df_slice['datetime'], u)
    ax[0,0].set_title('North-South Wind Speed')
    ax[0,0].set_xlabel(r'Time')

    ax[0,1].set_ylim(ymin-(0.1*abs(ymin)), ymax+(0.1*abs(ymax)))
    ax[0,1].plot(df_slice['datetime'], v)
    ax[0,1].set_title('East-West Wind Speed')
    ax[0,1].set_xlabel(r'Time')
    
    seismic_ymax = []
    seismic_ymin = []
    for k in range(len(stations)):
        # Seismic Smoothing
        if apply_seismic_smooth == True:
            EW = smooth(seismic_data[stations[k]][0], seismic_smoothie)
            NS = smooth(seismic_data[stations[k]][1], seismic_smoothie)
        else:
            EW = seismic_data[stations[k]][0]
            NS = seismic_data[stations[k]][1]

        seismic_ymax.append(max(EW.max(), NS.max()))
        seismic_ymin.append(min(EW.min(), NS.min()))

    seismic_ymax = max(seismic_ymax)
    seismic_ymin = min(seismic_ymin)

    for i, k in enumerate(stations):
        #t = seismic_data[k][2]
        # Seismic Smoothing
        if apply_seismic_smooth:
            EW = smooth(seismic_data[k][0], seismic_smoothie)
            NS = smooth(seismic_data[k][1], seismic_smoothie)
        else:
            EW = seismic_data[k][0]
            NS = seismic_data[k][1]
        # North component
        ax[i+1, 0].set_ylim(seismic_ymin-(0.2*abs(seismic_ymin)), seismic_ymax+(0.2*abs(seismic_ymax)))
        ax[i+1, 0].plot(NS)
        ax[i+1, 0].set_title(f'North-South Seismic {k}')
        ax[i+1, 0].set_xlabel(r'Time')

        # East component
        ax[i+1, 1].set_ylim(seismic_ymin-(0.2*abs(seismic_ymin)), seismic_ymax+(0.2*abs(seismic_ymax)))
        ax[i+1, 1].plot(EW)
        ax[i+1, 1].set_title(f'East-West Seismic {k}')
        ax[i+1, 1].set_xlabel(r'Time')

    plt.tight_layout()
    plt.show()




