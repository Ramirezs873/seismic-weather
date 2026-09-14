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

        # AWS data
        self.aws = None


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

    # Helper Functions

    @staticmethod 
    def find_channel(stream, options):
        """
        Find appropriate NS, EW, and Z Channels from a chosen stream.

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
    
    # Help Funcs to add:
        # save_as_mseed()
        # read_path()


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

                return self.seismic.st
            
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
        def window(self,
                   type = 'hann',
                   max_percentage = None,
                   max_length = 1, 
                   side = 'both',
                   save_mseed=False,
                   file_name='default_window'):
            """
            Apply a window function from seismic waveform data stored in a dictionary without altering the original.
        
            Parameters:
                type (str):
                    Type of window function applied. 
                    See the 'Supported Methods' for obspy.core.trace.Trace.taper function for a list of available window functions.
                max_percentage (float):
                    Decimal percentage of window function applied at an end.
                max_length (float):
                    Length in seconds of window function applied at an end.
                side (str):
                    End(s) at which the window function is applied. 
                    Available options are "left", "right", "both".
                save_mseed (bool):
                    True/False. True to save as mseed file.
                file_name (str):
                    Title of saved mseed file.

            Returns:
                new_dict (dict):
                    Dictionary containing tapered seismic waveform data.
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

                # Apply the window function and write to a dictionary
                new_dict = defaultdict(list)
                streams = []
                for station_name, traces in wave_dict.items():  
                    st = Stream(traces).copy() # Copy to avoid overwriting data
                    
                    st.taper(type=type, max_percentage=max_percentage, max_length=max_length, side=side)
                    streams.append(st)
                    new_dict[station_name].extend(st)
                
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

        # Filter
        def filter(self,
                   filter_type = None, 
                   freqmin=None, 
                   freqmax=None, 
                   freq=None, 
                   corners=4, 
                   zerophase=True,
                   save_mseed=False,
                   file_name='default_filter'):
            """
            Apply a filter to seismic waveform data stored in a dictionary.
        
            Parameters:
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
                filename (str):
                    Title of saved mseed file.
            
            Returns:
                filtered_dict (dict):
                    Dictionary containing filtered seismic waveform data.
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

                # Rewrite self.st
                self.seismic.st = filtered_dict
        
            return self.seismic
            

        # Amplitude
        def amp_correction(self,
                           NS_channel=None,
                           EW_channel=None,
                           Z_channel=None,
                           sensitivity = 2.994697576134245E8,
                           NS_correction_factor=None,
                           EW_correction_factor=None,
                           Z_correction_factor=None,
                           save_mseed=False,
                           file_name='default_amp_corr'):
        
            """
            Corrects the amplitude of a seismic waveform 
            from a list of correction factors for each channel.
            
            Parameters:
                wave_list_dict (list):
                    List of dictionaries containing seismic waveform data 
                    in the form (EW_data, NS_data).
                    Dictionary containing seismic waveform data 
                    in the form (EW_data, NS_data).
                NS_channel (list of str):
                    Possible channel codes for North-South instrument component.
                EW_channel (list of str):
                    Possible channel codes for East-West instrument component.
                NS_channel (list of str):
                    Possible channel codes for North-South instrument component.
                Z_channel (list of str):
                    Possible channel codes for vertical instrument component.
                sensitivity (float):
                    Seismic Intrusment Sensitivity. Check Instrument metadeta for the value.
                NS_correction_factor (list of float):
                    NS amplitude correction factor.
                EW_correction_factor (list of float):
                    EW amplitude correction factor.
                Z_correction_factor (list of float):
                    Z amplitude correction factor.
                save_mseed (bool):
                    True/False. True to save as mseed file.
                config (dict):
                    Information from a config file containing the local "seismic_data_path".
                read_file (bool):
                    True/False. True to switch on file checking.
                filename (str):
                    Title of saved mseed file.

                Returns:
                amplitude_corrected_obspy (dict):
                    Dictionary containing amplitude corrected seismic waveform data.
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

                station_list = list(wave_dict.keys())
                amplitude_corrected_obspy = defaultdict(list) 
                streams = []
        
                for (stat, stream, NS_correct, EW_correct, Z_correct) in zip(station_list, 
                                                                                wave_dict.values(), 
                                                                                NS_correction_factor, 
                                                                                EW_correction_factor, 
                                                                                Z_correction_factor):
                    print(f"Processing {stat}...")
        
                    st = Stream(stream)
                    st.sort(['channel'])
                    NS = self.seismic.find_channel(st, NS_channel) 
                    EW = self.seismic.find_channel(st, EW_channel) 
                    Z = self.seismic.find_channel(st, Z_channel)
        
                    t_start = min(tr.stats.starttime for tr in st)
                    fs = st[0].stats.sampling_rate
        
                    # Adjust for Sensitivity, Convert to m/s
                    NS_v = NS[0].data / sensitivity
                    EW_v = EW[0].data / sensitivity
                    Z_v = Z[0].data / sensitivity
        
                    NS_corrected = NS_v * NS_correct if NS else None
                    EW_corrected = EW_v * EW_correct if EW else None
                    Z_corrected = Z_v * Z_correct if Z else None
        
                    # Create new obspy stream with aligned data
                    st = Stream()
                    NS_name = NS[0].stats.channel if NS else None
                    EW_name = EW[0].stats.channel if EW else None
                    Z_name  = Z[0].stats.channel if Z else None
                    network_name = wave_dict[stat][0].stats.network 
                    station_name = wave_dict[stat][0].stats.station
        
                    
                    components = {EW_name: EW_corrected, NS_name: NS_corrected, Z_name: Z_corrected}
                    if NS_name is None or EW_name is None or Z_name is None:
                        print(f"Skipping {stat}. Missing channel")
                        continue
        
                    for channel, data in components.items():
                        tr = Trace(data=data)
                        tr.stats.network = network_name
                        tr.stats.station = station_name
                        tr.stats.channel = channel
                        tr.stats.starttime = UTC(t_start)
                        tr.stats.sampling_rate = fs
                        st.append(tr)
                    streams.append(st)
                    amplitude_corrected_obspy[stat] = st
                
                # Save as mseed file
                if save_mseed == True:
                    merged_stream = streams[0].copy() # Copy to avoid overwriting data
                    for st in streams[1:]:
                        merged_stream += st
                    merged_stream.merge()
        
                    merged_stream.write(f'{str(file_path)}', format="MSEED")
                    print(f'Saved as {file_path}')

                # Rewrite self.st
                self.seismic.st = amplitude_corrected_obspy
        
            return self.seismic
    

        # Rotate
        def rotate(self,
                   NS_channel = None, 
                   EW_channel = None, 
                   Z_channel = None,
                   misalignment_angle = None,
                   save_mseed=False,
                   file_name='default_rotate'):
    
            """
            Apply a complex transform to a seismic waveform to correct for rotation errors.

            Parameters:
                NS_channel (list of str):
                    Possible channel codes for North-South instrument component.
                EW_channel (list of str):
                    Possible channel codes for East-West instrument component.
                NS_channel (list of str):
                    Possible channel codes for North-South instrument component.
                Z_channel (list of str):
                    Possible channel codes for vertical instrument component.
                misalignment_angle (list of float):
                    The angle in degrees which the waveform is misaligned.
                save_mseed (bool):
                    True/False. True to save as mseed file.
                filename (str):
                    Title of saved mseed file.

            Returns:
                aligned_obspy (dict):
                    Dictionary containing rotation corrected seismic waveform data.
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

                station_list = list(wave_dict.keys())
                aligned_wave_dict = {}
                aligned_obspy = defaultdict(list)
                streams = []
        
                for (stat, stream, angle) in zip(station_list, wave_dict.values(), misalignment_angle):
                    print(f"Processing {stat}...")
                    st = Stream(stream)
                    st.sort(['channel'])
                    NS = self.seismic.find_channel(st, NS_channel) 
                    EW = self.seismic.find_channel(st, EW_channel) 
                    Z = self.seismic.find_channel(st, Z_channel)
        
                    t_start = min(tr.stats.starttime for tr in st)
                    t_end   = max(tr.stats.endtime for tr in st)
                    fs = st[0].stats.sampling_rate
                    npts = int(round((t_end - t_start) * fs))
        
                    # Make each channel a continuous data set by filling in the gaps with NaNs. 
                    # This ensures that the cross correlation and rotation are applied to the entire signal, even if there are gaps in the data.
                    E = np.full(npts, np.nan)
                    N = np.full(npts, np.nan)
                    Z_2 = np.full(npts, np.nan)
        
                    for tr in EW:
                        i0 = int(round((tr.stats.starttime - t_start) * fs))
                        i1 = i0 + len(tr.data)
                        if i1 > npts:
                            i1 = npts
                            data = tr.data[:(i1 - i0)]
                        else:
                            data = tr.data
        
                        E[i0:i1] = data
        
                    for tr in NS:
                        i0 = int(round((tr.stats.starttime - t_start) * fs))
                        i1 = i0 + len(tr.data)
                        if i1 > npts:
                            i1 = npts
                            data = tr.data[:(i1 - i0)]
                        else:
                            data = tr.data
        
                        N[i0:i1] = data
        
                    for tr in Z:
                        i0 = int(round((tr.stats.starttime - t_start) * fs))
                        i1 = i0 + len(tr.data)
                        if i1 > npts:
                            i1 = npts
                            data = tr.data[:(i1 - i0)]
                        else:
                            data = tr.data
        
                        Z_2[i0:i1] = data
        
                    n = min(len(N), len(E), len(Z_2)) 
                    y = N[:n] 
                    x = E[:n] 
                    z = Z_2[:n]
        
        
                    S_k = x + 1j*y
                    S_k_aligned = S_k *np.exp(-1j * np.deg2rad(angle))
        
                    x = np.real(S_k_aligned)
                    y = np.imag(S_k_aligned)
                    
                    #times = NS.times("timestamp")[:n]        
                    aligned_wave_dict[stat] =  x, y, z, fs, t_start
        
                    # Create new obspy stream with aligned data
                    st = Stream()
                    NS_name = NS[0].stats.channel if NS else None
                    EW_name = EW[0].stats.channel if EW else None
                    Z_name  = Z[0].stats.channel if Z else None
                    network_name = wave_dict[stat][0].stats.network 
                    station_name = wave_dict[stat][0].stats.station
        
                    components = {EW_name: x, NS_name: y, Z_name: z}
                    if NS_name is None or EW_name is None or Z_name is None:
                        print(f"Skipping {stat}. Missing channel")
                        continue
        
                    for channel, data in components.items():
                        tr = Trace(data=data)
                        tr.stats.network = network_name
                        tr.stats.station = station_name
                        tr.stats.channel = channel
                        tr.stats.starttime = UTC(t_start)
                        tr.stats.sampling_rate = fs
                        st.append(tr)
        
                    streams.append(st)
                    aligned_obspy[stat] = st
        
                # Save as mseed file
                if save_mseed == True:
                    merged_stream = streams[0].copy() # Copy to avoid overwriting data
                    for st in streams[1:]:
                        merged_stream += st
                    merged_stream.merge()
        
                    merged_stream.write(f'{str(file_path)}', format="MSEED")
                    print(f'Saved as {file_path}')

                # Rewrite self.st
                self.seismic.st = aligned_obspy
                    
            return self.seismic
            


    # Seismic Analysis
    class Analysis:

        def __init__(self, seismic):
            self.seismic = seismic

        # Plot Streams
        def plot_streams(self,
                         color='black'):
    
            """
            Plot seismic waveform data stored in a dictionary.
            
            Parameters:
            color (str):
                Colour for the waveform plots. Default is 'black'.
            """
            # Wave_dict
            wave_dict = self.seismic.st
            
            # Loop through and plot streams
            for station_name in wave_dict:
                if not wave_dict[station_name]:    # Skip empty stations
                    print(f"Skipping empty station: {station_name}")
                    continue
                st = Stream(wave_dict[station_name])
                print(f"Plotting {station_name}")
                st.plot(color=color)
            

        # Create PPSD Plots
        def ppsd(self):
            pass


        # Inspect Amplitudes
        def amp(self):
            pass


        # Cross Correlation
        def cc_correction(self):
            pass


        # Tabulate Cross Correlation
        def cc_table(self):
            pass


        # Event Data
        def events(self):
            pass


        # Station Inventory
        def inventory(self):
            pass
        


    # Seismic Wind Analysis
    class Wind:

        def __init__(self, seismic):
            self.seismic = seismic
            self.aws = None
            self.wd = None
            self.seismic_wind = self.SeismicWind(seismic=self.seismic)

        # Wind Data
        def get_aws(self,
                    path,
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
                self.seismic:
                    Seismic data
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
            self.aws = df

            return self.aws


        def get_var(self,
                    variable='Wind speed in km/hr',
                    year=None, 
                    month=None, 
                    day=None,
                    hour=None):
            
            # Check if time inputs are single valued or a range.
            # Year
            if isinstance(year, (tuple, list)) and len(year) == 2: 
                start_year, end_year = year 
                df_slice = self.aws[(self.aws['datetime'].dt.year >= start_year) & 
                                (self.aws['datetime'].dt.year <= end_year) ].copy()
            elif year is None or year < 2010 or year > 2025:
                df_slice = self.aws.copy()
            else:
                df_slice = self.aws[self.aws['datetime'].dt.year == year].copy()
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
        
            df_slice['Wind direction in degrees true'] = pd.to_numeric(df_slice['Wind direction in degrees true'], errors='coerce')
            df_slice['Wind direction in degrees true'] %= 360
            WD = df_slice['Wind direction in degrees true']
        
            
            var = var.to_numpy()
            WD = WD.to_numpy()
        
            # Clean Data
            valid_mask = ~np.isnan(var)
        
            var = var[valid_mask]
            WD = WD[valid_mask]
            time = df_slice['datetime'].to_numpy()[valid_mask]

            self.wd = [time, var, WD]
            self.seismic_wind.wd = self.wd
            
            return self.wd

        # Plot aws speed
        def plot_aws(self,
                     ylabel = 'AWS Varaible',
                     apply_smooth = False,
                     smoothie = 3):
            """
            Plots the aws for a given year and month from the provided DataFrame.
            
            Parameters:
                apply_smooth (bool): 
                    True/False. Applying ObsPy smooth() function.
                smoothie (int):
                    Number of values to calculate moving average for smoothing.
            """
            # Create Figure 
            plt.figure(figsize=(15,6))
    
            # Plot
            plt.plot(self.wd[0], self.wd[1], 
                    color='black', linewidth=0.5)
            
            # Plot Formating
            plt.ylabel(f'{ylabel}')
            plt.xlabel('Time')
            ymax = np.max(self.wd[1]) # For y axis limit
            plt.ylim(0, ymax *1.1)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

            

        # Plot wind speed and direction
        def plot_rose_wind(self,
                           year=None, 
                           month=None, 
                           day=None,
                           hour=None):
        
            """
            Plots a rose plot of wind speed and direction 
            for a given year and month from the provided DataFrame.
        
            Parameters:
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
                df_slice = self.aws[(self.aws['datetime'].dt.year >= start_year) & 
                                (self.aws['datetime'].dt.year <= end_year) ].copy()
            elif year is None or year < 2010 or year > 2025:
                df_slice = self.aws.copy()
            else:
                df_slice = self.aws[self.aws['datetime'].dt.year == year].copy()
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
            



        # Seismic Wind Analysis
        class SeismicWind:

            def __init__(self, seismic):
                self.seismic = seismic
                self.aws = None
                self.models = self.Models(self)
                self.fft = None
                self.wd = None

            # FFT Analysis
            def spectra_fft(self,
                            pad_value=np.nan):
                
                """
                Trim 30 minute segments of seismic data which match AWS dataset.
                Apply a real fast fourier transform to compute the frequency spectrum 
                for each seismic component and calculate the powerfor each time step. 
            
                Parameters:
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

                # Wave_dict
                wave_dict = self.seismic.st

                 # Setup 
                station_list = list(wave_dict.keys())
                aws_times = self.wd[0]   # timestamps
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
                                    'aws_values' : self.wd[1],
                                    'wind_direction' : self.wd[2]})
            
                    all_spectra.append({station: spectra})
                    self.fft = all_spectra 

                return self.fft
            


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