# seismic-weather
Investigate near sensor seismic weather signals alongside AWS data.

## Main Features
* Gather and Read Data
   * Read BoM AWS Data
   * Download and Read Seismic Data
* Seismic Waveform Preprocessing
   * Trim 
   * Demean
   * Detrend
   * Window Function
   * Filter Frequencies
   * Amplitude Correction
   * Rotation Correction
* Inspect and Analyse Data
   * Plot Seismic Waveforms
   * Plot AWS Variable Time Series
   * Plot Wind Speed and Wind Direction
   * Seismic FFT (with AWS Time Matchup)
   * Create Seismic PPSDs (Implemented Soon)
   * Determine Seismic Station Orientation (Implemented Soon)
   * Inspect Broad Seismic Event Information for Stations (Implemented Soon)
   * Inspect Station Data and Inventory Creation (Implemented Soon)
* Determine Wind Speed and Seismic Power Relationship 
   * Ridge Model
   * ElasticNet Model
   * Support Vector Regression Model
   * Random Forest Model
   * Compare Models
   * Optimise Models
     
## Repository Information
This repository contains the wind.py and seismicwind.py files. wind.py contains functions for all the main features, alongside early development functions, and other workflows which have been superseded. seismicwind.py contains the main features in a nicely packaged format.

If only using wind.py, a more efficient workflow involves using the accompanying seismic-sensor-analysis repository (https://github.com/Ramirezs873/seismic-sensor-analysis) along side wind.py.

seismicwind.py is self contained besides the standard packages listed below.

## Instructions
Install this package by adding seismicwind.py to your python working directory. 
* seismicwind.py requires a few other libraries to work properly. 
    * Plotly (https://github.com/plotly/plotly.py)
    * ObsPy (https://github.com/obspy/obspy)
    * Pandas (https://github.com/pandas-dev/pandas)
    * NumPy (https://github.com/numpy/numpy)
    * Matplotlib (https://github.com/matplotlib/matplotlib)
    * SciPy (https://github.com/scipy/scipy)
    * Scikit-learn (https://github.com/scikit-learn/scikit-learn)
      
### To get started:
```
import seismicwind.py
```
### Current Methods
```
class Seismic:

   # Retrieve Seismic Data
   def get_seis():

   # Seismic Preprocessing
   class Preprocess:

      # Trim Seismic Data
      def trim():

      # Demean Seismic Data
      def demean():

      # Detrend Seismic Data
      def detrend():

      # Window Function Seismic Data
      def window():

      # Filter Seismic Data
      def filter():

      # Amplitude Correct Seismic Data
      def amp_correction():

      # Rotation Correct Seismic Data
      def rotate():

   # Seismic Analysis
   class Analysis:

      # Plot Seismic Waveforms
      def plot_streams():

      # More to be added soon

   # Wind Analysis
   class Wind:

      # Retrieve AWS Data
      def get_aws():

      # Isolate Single AWS Variable (+ Wind Direction)
      def get_var():

      # Plot AWS Variable
      def plot_aws():

      # Plot Wind Speed and Direction
      def plot_rose_wind():

      # Seismic Wind Analysis
      class SeismicWind:

         # FFT Analysis
         def spectra_fft():

         # Regression Models
         class Models:

         # Optimise Models
         def optimise_ridge():
         def optimise_EN():
         def optimise_SVR():
         def optimise_RF():

         # Model Analysis
         def Ridge():
         def EN():
         def SVR():
         def RF():

         # Compare Models
         def compare_models():
         
```

### Data formats
AWS data is read as a .txt or .parquet file. AWS file title should read: 'HM01X_Data_{station_code}_{id_code}'
Seismic data is read as a dictionary of isolated components for each station. 

---
### Config File

The `config.yml` file contains paths to local directories

```
name: Windy 
aws_dir: .../path/to/aws/data/
seis_dir: .../path/to/seismic/data/
```
