from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px
from obspy.signal.util import smooth
from obspy import UTCDateTime as UTC
from scipy.stats import linregress



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
    
