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
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
import matplotlib
from obspy.clients.fdsn import Client


class Seismic:

    def __init__(self, config=None):

        # Config File
        self.config = config

        # Seismic Data
        self.st = None

        # Preprocess data
        self.preprocess = self.Preprocess(self)

        # Using seismic data create stuff
        self.analysis = self.Analysis(self)

        # Seismic Wind Analysis
        self.wind = self.Wind(self)


    # Gather Data
    def get_seis(self, 
                 client,
                 network,
                 station,
                 location,
                 channel,
                 t_start,
                 t_end,
                 username = None,
                 password = None,
                 file_name = None):
        """
        Gather seismic waveform data from a FDSN client,
        or from a local file if the file name already exists
        in the directory. 

        Parameters:
            client (str):
                FDSN client name. e.g 'IRIS', 'AUSPASS'.
            network (str):
                Network code. e.g 'IU', 'AU'.
            station (str):
                Station code. e.g. 'CASY', 'CWA86'.
            location (str):
                Location code. e.g '00', '10'.
                Use '*' for all locations.
            channel (str):
                Channel code. e.g 'BH1', 'NHE'.
                Use '?' for all types of partly specified channel. 
                Use '*' for all channels.
            t_start (UTCDateTime): 
                Start time for data retrieval.
            t_end (UTCDateTime): 
                End time for data retrieval.
            username (str):
                Username to access data (if required).
            password (str):
                Password to access data (if required).
            file_name (str):
                Filename to save the data as.
            
        Returns:
            wave_dict (dict):
                Dictionary containing seismic waveform data.

        """

        # Check if file already exists
        stat_number = len(station) # Number of stations
        time = t_start.strftime("%Y-%m-%d") # Start date of data

        # Config Support
        base_path = Path(self.config["seismic_data_path"]) if self.config else Path(".")
        base_path.mkdir(parents=True, exist_ok=True)

        if file_name is None:
            filename = f"{network}_{station}_{time}"
        else:
            filename = file_name

        file_path = (base_path / filename).with_suffix(".mseed")

        if file_path.exists():
            print(f"Reading existing file: {file_path}")
            station_data = read(str(file_path))

        else:
            print("File not found. Downloading data")
    
            # Gather data from target stations
            g = Client(base_url=client, 
                        user=username, 
                        password=password) # FDSN client 

            # Gather waveform data
            station_data = g.get_waveforms(network=network, 
                                            station=station, 
                                            location=location, 
                                            channel=channel, 
                                            starttime=t_start, 
                                            endtime=t_end) 

            # Write to file
            station_data.write(str(file_path), format = 'MSEED')
            print(f"Saved to: {file_path}")
        
        # Write to a dictionary
        wave_dict = defaultdict(list) 
    
        for tr in station_data:
            station = (
            f"{tr.stats.network}."
            f"{tr.stats.station}."
            f"{tr.stats.location}"
            )
            wave_dict[station].append(tr) 

        self.st = wave_dict

        return self 


    # Preprocess data
    class Preprocess:

        def __init__(self, seismic):
            self.seismic = seismic

        # Trim
        def trim(self,
                 t_start=None, 
                 duration=None,
                 save_mseed=False,
                 file_name='default_trim'):

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
                file_name (str):
                    Filename to save the data as.

            Returns:
                new_dict (dict):
                    Dictionary containing selected seismic waveform data.
            """
        
            # Path 
            base_path = Path(self.seismic.config["seismic_data_path"]) if self.seismic.config else Path(".")
            base_path.mkdir(parents=True, exist_ok=True)
            file_path = (base_path / file_name).with_suffix(".mseed")

            # Read file if it exists
            if file_path.exists():
                print(f"Reading existing file: {file_path}")
                stream = read(str(file_path))
                new_dict = defaultdict(list)
                for tr in stream:
                    new_dict[tr.stats.station].append(tr)

                # Rewrite self.st
                self.seismic.st = new_dict

                return self.seismic
            
            else:
                # Establish timespan
                t_end = t_start + duration
                streams = []

                # Wave_dict
                wave_dict = self.seismic.st

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

                # Rewrite self.st
                self.seismic.st = new_dict
        
            return self.seismic

        # Demean
        def demean(self,
                   save_mseed = False,
                   file_name = 'default_demean'):

            """
            Demean seismic waveform data stored in a dictionary without altering the original.
        
            Parameters:
            save_mseed (bool):
                True/False. True to save as mseed file.
            file_name (str):
                Title of saved mseed file.
            Returns:

            new_dict (dict):
                Dictionary containing demeaned seismic waveform data.
            """

            # Path 
            base_path = Path(self.seismic.config["seismic_data_path"]) if self.seismic.config else Path(".")
            base_path.mkdir(parents=True, exist_ok=True)
            file_path = (base_path / file_name).with_suffix(".mseed")

            # Read file if it exists
            if file_path.exists():
                print(f"Reading existing file: {file_path}")
                stream = read(str(file_path))
                new_dict = defaultdict(list)
                for tr in stream:
                    new_dict[tr.stats.station].append(tr)

                # Rewrite self.st
                self.seismic.st = new_dict

            else:
                # Wave_dict
                wave_dict = self.seismic.st

                # Demean and detrend the data and write to a dictionary
                new_dict = defaultdict(list)
                streams = []
        
                for station_name, traces in wave_dict.items():  
                    st = Stream(traces).copy() # Copy to avoid overwriting data
                    st.detrend("demean")
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

                # Rewrite self.st
                self.seismic.st = new_dict

            return self.seismic
        

        # Detrend
        def detrend(self,
                   save_mseed = False,
                   file_name = 'default_detrended'):

            """
            Detrended seismic waveform data stored in a dictionary without altering the original.
        
            Parameters:
            save_mseed (bool):
                True/False. True to save as mseed file.
            file_name (str):
                Title of saved mseed file.
            Returns:

            new_dict (dict):
                Dictionary containing detrended seismic waveform data.
            """

            # Path 
            base_path = Path(self.seismic.config["seismic_data_path"]) if self.seismic.config else Path(".")
            base_path.mkdir(parents=True, exist_ok=True)
            file_path = (base_path / file_name).with_suffix(".mseed")

            # Read file if it exists
            if file_path.exists():
                print(f"Reading existing file: {file_path}")
                stream = read(str(file_path))
                new_dict = defaultdict(list)
                for tr in stream:
                    new_dict[tr.stats.station].append(tr)

                # Rewrite self.st
                self.seismic.st = new_dict

            else:
                # Wave_dict
                wave_dict = self.seismic.st

                # Demean and detrend the data and write to a dictionary
                new_dict = defaultdict(list)
                streams = []
        
                for station_name, traces in wave_dict.items():  
                    st = Stream(traces).copy() # Copy to avoid overwriting data
                    st.detrend("linear")
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

                # Rewrite self.st
                self.seismic.st = new_dict

            return self.seismic

        # Window Function
        def window(self):
            pass

        # Filter
        def filter(self):
            pass

        # Amplitude
        def amp(self):
            pass

        # Rotate
        def rotate(self):
            pass


    # Using seismic data create stuff
    class Analysis:

        def __init__(self, seismic):
            self.seismic = seismic

        # Plot Streams
        def plot_streams(self):
            pass

        # Create PPSD Plots
        def ppsd(self):
            pass


    # Seismic Wind Analysis
    class Wind:

        def __init__(self, seismic):
            self.seismic = seismic
            self.wind_data = None
            self.seismic_wind = self.SeismicWind(self)

        # Wind Data
        def get_wind_data(self):
            pass


        # Wind Analysis
        class SeismicWind:

            def __init__(self, wind):
                self.wind = wind
                self.models = self.Models(self)

            # FFT Analysis
            def spectra_fft(self):
                pass

            # Plot wind speed
            def plot_wind(self):
                pass

            # Plot wind speed and direction
            def plot_rose_wind(self):
                pass


            # Regression Models
            class Models:

                def __init__(self, seismic_wind):
                    self.seismic_wind = seismic_wind

                # Optimisation
                def optimise_ridge(self):
                    pass

                def optimise_EN(self):
                    pass

                def optimise_SVR(self):
                    pass

                def optimise_RF(self):
                    pass


                # Model Analysis
                def RF(self):
                    pass

                def SVR(self):
                    pass

                def EN(self):
                    pass

                def Ridge(self):
                    pass

                def compare_models(self):
                    pass