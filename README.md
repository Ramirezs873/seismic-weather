# seismic-weather
Investigate near sensor seismic signals and AWS data.

## Main Features
* Read BoM AWS data 
* Plot wind speed
* Plot wind direction
* Compare AWS data to seismic data
## Repository Information
This repository contains the wind.py file. It contains all the functions for the main features. 
For a more efficient workflow, use the accompanying seismic-sensor-analysis repository (https://github.com/Ramirezs873/seismic-sensor-analysis) along side seismic-weather. 

## Instructions
Install this package by adding it your python working directory. 
* wind.py requires a few other libraries to work properly. 
    * Plotly (https://github.com/plotly/plotly.py)
    * ObsPy (https://github.com/obspy/obspy)
    * Pandas (https://github.com/pandas-dev/pandas)
    * NumPy (https://github.com/numpy/numpy)
    * Matplotlib (https://github.com/matplotlib/matplotlib)

To get started:
```
import wind.py
```
Current functions include:
* read_data()
* plot_wind_speed()
* plot_rose_wind()
* compare_seismic_wind()

### Data formats
AWS data is read as a .txt or .parquet file. read_data() searches for title: 'HM01X_Data_{station_code}_{id_code}'
Seismic data is read as a dictionary of isolated components for each station. 
* Following the workflow in seismic-sensor-analysis gives the seismic data in the correct format for compare_seismic_wind(). 

