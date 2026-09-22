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
        

        # Config Support
        base_path = Path(self.config["seismic_data_path"]) if self.config else Path(".")
        base_path.mkdir(parents=True, exist_ok=True)

        if file_name is None:
            time = t_start.strftime("%Y-%m-%d") # Start date of data
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
        # Create bands
        # Model Setup


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
                     ylabel = 'AWS Varaible'):
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
            

            # Regression Model
            class Models:


                def __init__(self, seismic_wind):
                    self.seismic_wind = seismic_wind


                # Optimisation
                def optimise_ridge(self,
                                   fmin = 1,
                                   fmax = 49,
                                   f_band_width = 1,
                                   step_size = 1,
                                   n_splits = 5,
                                   min_WS = None):
                    # Bands
                    # Create Bandwidths
                    bands = []
                    for f1 in range(fmin, fmax - f_band_width + 1, step_size):
                        f2 = f1 + f_band_width
                        band = (f1, f2)
                        bands.append(band)
                    
                    # Create Band Centers
                    band_centres = []
                    for f1, f2 in bands:
                        band_centre = [(f1 + f2)/2]
                        band_centres.append(band_centre)
                
                    # Loop through stations
                    for station_dict in self.seismic_wind.fft:
                
                        # Setup Variables
                        station = list(station_dict.keys())[0] 
                        EW_power = station_dict[station][0]['EW']
                        NS_power = station_dict[station][0]['NS']
                        Z_power = station_dict[station][0]['Z']
                        freq = station_dict[station][0]['freq']
                        aws_values = station_dict[station][0]['aws_values']
                        wind_direction = station_dict[station][0]['wind_direction']
                
                        # Angle wrap around problem
                        dir_radians = np.deg2rad(wind_direction)
                        dir_sin = np.sin(dir_radians)
                        dir_cos = np.cos(dir_radians)
                        
                        # Setup powers
                        X_Z = np.zeros((len(aws_values), len(bands)))
                        X_NS = np.zeros((len(aws_values), len(bands)))
                        X_EW = np.zeros((len(aws_values), len(bands)))
                
                        # Apply bandwidths to data
                        for i, (f1,f2) in enumerate(bands):
                            band_width = (freq >= f1) & (freq < f2)
                
                            # Convert to log to better inspect power scales and apply bandwidth
                            # [:, band_width], select frequencies and slice unwanted freq data from the row
                            # .mean(axis=1), mean for the selected frequency row. Reshape for model input.
                            X_Z[:, i] = np.log10(Z_power[:, band_width].mean(axis=1) + 1e-20)
                            X_NS[:, i] = np.log10(NS_power[:, band_width].mean(axis=1) + 1e-20)
                            X_EW[:, i] = np.log10(EW_power[:, band_width].mean(axis=1) + 1e-20)
                
                        # Stack all together to process all together as a larger dataset
                        X_all = np.column_stack([X_Z, X_NS, X_EW])
                        y_all = np.column_stack([aws_values, dir_sin, dir_cos])
                
                
                        # Should the wind speed threshold be applied before or after training the model?
                        # Apply minimum WS threshold
                        if min_WS is not None:
                            mask = y_all[:, 0] > min_WS
                            X_all = X_all[mask]
                            y_all = y_all[mask]
                            
                        # Train Model
                        X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
                
                        # Find best alpha
                        
                        # setup alpha list
                        alphas = np.logspace(-4, 4, 100)
                
                
                        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                
                        # Ridge
                        # Pipeline
                        # Scale, RidgeCV 
                        var_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', RidgeCV(alphas=alphas, cv=cv, scoring ='r2'))])
                        sin_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', RidgeCV(alphas=alphas, cv=cv, scoring ='r2'))])
                        cos_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', RidgeCV(alphas=alphas, cv=cv, scoring ='r2'))])
                
                        var_ridge.fit(X_all_train, y_all_train[:, 0])
                        sin_ridge.fit(X_all_train, y_all_train[:, 1])
                        cos_ridge.fit(X_all_train, y_all_train[:, 2])
                
                        print("WS alpha:", var_ridge.named_steps['ridge'].alpha_,
                                "CV R²:", var_ridge.named_steps['ridge'].best_score_)
                        print("--------------------------------------------------")
                        print("Sin2 alpha:", sin_ridge.named_steps['ridge'].alpha_,
                                "CV R²:", sin_ridge.named_steps['ridge'].best_score_)
                        print("--------------------------------------------------")
                        print("Cos2 alpha:", cos_ridge.named_steps['ridge'].alpha_,
                                "CV R²:", cos_ridge.named_steps['ridge'].best_score_)


                def optimise_EN(self,
                                fmin = 1,
                                fmax = 49,
                                f_band_width = 1,
                                step_size = 1,
                                n_splits = 5,
                                min_WS = None):
                    
                    # Bands
                    # Create Bandwidths
                    bands = []
                    for f1 in range(fmin, fmax - f_band_width + 1, step_size):
                        f2 = f1 + f_band_width
                        band = (f1, f2)
                        bands.append(band)
                    # Num of Bands
                    n_bands = len(bands)
                    # Create Band Centers
                    band_centres = []
                    for f1, f2 in bands:
                        band_centre = [(f1 + f2)/2]
                        band_centres.append(band_centre)
                
                    # Loop through stations
                    for station_dict in self.seismic_wind.fft:
                
                        # Setup Variables
                        station = list(station_dict.keys())[0] 
                        EW_power = station_dict[station][0]['EW']
                        NS_power = station_dict[station][0]['NS']
                        Z_power = station_dict[station][0]['Z']
                        freq = station_dict[station][0]['freq']
                        aws_values = station_dict[station][0]['aws_values']
                        wind_direction = station_dict[station][0]['wind_direction']
                
                        # Angle wrap around problem
                        dir_radians = np.deg2rad(wind_direction)
                        dir_sin = np.sin(dir_radians)
                        dir_cos = np.cos(dir_radians)
                        
                        
                        # Setup powers
                        X_Z = np.zeros((len(aws_values), len(bands)))
                        X_NS = np.zeros((len(aws_values), len(bands)))
                        X_EW = np.zeros((len(aws_values), len(bands)))
                
                        # Apply bandwidths to data
                        for i, (f1,f2) in enumerate(bands):
                            band_width = (freq >= f1) & (freq < f2)
                
                            # Convert to log to better inspect power scales and apply bandwidth
                            # [:, band_width], select frequencies and slice unwanted freq data from the row
                            # .mean(axis=1), mean for the selected frequency row. Reshape for model input.
                            X_Z[:, i] = np.log10(Z_power[:, band_width].mean(axis=1) + 1e-20)
                            X_NS[:, i] = np.log10(NS_power[:, band_width].mean(axis=1) + 1e-20)
                            X_EW[:, i] = np.log10(EW_power[:, band_width].mean(axis=1) + 1e-20)
                
                        # Stack all together to process all together as a larger dataset
                        X_all = np.column_stack([X_Z, X_NS, X_EW])
                        y_all = np.column_stack([aws_values, dir_sin, dir_cos])
                
                
                        # Should the wind speed threshold be applied before or after training the model?
                        # Apply minimum WS threshold
                        if min_WS is not None:
                            mask = y_all[:, 0] > min_WS
                            X_all = X_all[mask]
                            y_all = y_all[mask]
                            
                        # Train Model
                        X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
                
                        # Find best alpha & l1 ratio combination
                        # setup alpha list
                        alphas = np.logspace(-3, 3, 20)
                        l1s = np.linspace(0.01, 1.0, 15)
                
                        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                
                        # ElasticNet
                        # Pipeline
                        # Scale, ElasticNet 
                        var_EN = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNetCV(alphas=alphas, l1_ratio=l1s, cv=cv, max_iter=50000))])
                        sin_EN = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNetCV(alphas=alphas, l1_ratio=l1s, cv=cv, max_iter=50000))])
                        cos_EN = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNetCV(alphas=alphas, l1_ratio=l1s, cv=cv, max_iter=50000))])
                
                        var_EN.fit(X_all_train, y_all_train[:, 0])
                        sin_EN.fit(X_all_train, y_all_train[:, 1])
                        cos_EN.fit(X_all_train, y_all_train[:, 2])
                
                        # Get optimised values
                        var_alpha = var_EN.named_steps['enet'].alpha_
                        var_l1 = var_EN.named_steps['enet'].l1_ratio_
                
                        sin_alpha = sin_EN.named_steps['enet'].alpha_
                        sin_l1 = sin_EN.named_steps['enet'].l1_ratio_
                
                        cos_alpha = cos_EN.named_steps['enet'].alpha_
                        cos_l1 = cos_EN.named_steps['enet'].l1_ratio_
                
                        # Find CV R2 for these params
                        var_EN_best = Pipeline([('scaler', StandardScaler()),('enet', ElasticNet(alpha=var_alpha, l1_ratio=var_l1, max_iter=50000))])
                        sin_EN_best = Pipeline([('scaler', StandardScaler()),('enet', ElasticNet(alpha=sin_alpha, l1_ratio=sin_l1, max_iter=50000))])
                        cos_EN_best = Pipeline([('scaler', StandardScaler()),('enet', ElasticNet(alpha=cos_alpha, l1_ratio=cos_l1, max_iter=50000))])      
                
                        # Cross Validation
                        var_scores = cross_val_score(var_EN_best, X_all_train, y_all_train[:, 0], cv=cv, scoring='r2')
                        sin_scores = cross_val_score(sin_EN_best, X_all_train, y_all_train[:, 1], cv=cv, scoring='r2')
                        cos_scores = cross_val_score(cos_EN_best, X_all_train, y_all_train[:, 2], cv=cv, scoring='r2')
                
                        # Print results
                        print("WS alpha:", var_alpha,
                                "WS l1_ratio:", var_l1,
                                "WS CV R²:", var_scores.mean(), "+/-", var_scores.std())
                        print("--------------------------------------------------")
                        print("Sin2 alpha:", sin_alpha,
                                "Sin2 l1_ratio:", sin_l1,
                                "Sin2 CV R²:", sin_scores.mean(), "+/-", sin_scores.std())
                        print("--------------------------------------------------")
                        print("Cos2 alpha:", cos_alpha,
                                "Cos2 l1_ratio:", cos_l1,
                                "Cos2 CV R²:", cos_scores.mean(), "+/-", cos_scores.std())

                def optimise_SVR(self,
                                fmin = 1,
                                fmax = 49,
                                f_band_width = 1,
                                step_size = 1,
                                n_splits = 5,
                                min_WS = None):
                    # Bands
                    # Create Bandwidths
                    bands = []
                    for f1 in range(fmin, fmax - f_band_width + 1, step_size):
                        f2 = f1 + f_band_width
                        band = (f1, f2)
                        bands.append(band)
                    # Num of Bands
                    n_bands = len(bands)
                    # Create Band Centers
                    band_centres = []
                    for f1, f2 in bands:
                        band_centre = [(f1 + f2)/2]
                        band_centres.append(band_centre)
                
                    # Loop through stations
                    for station_dict in self.seismic_wind.fft:
                
                        # Setup Variables
                        station = list(station_dict.keys())[0] 
                        EW_power = station_dict[station][0]['EW']
                        NS_power = station_dict[station][0]['NS']
                        Z_power = station_dict[station][0]['Z']
                        freq = station_dict[station][0]['freq']
                        aws_values = station_dict[station][0]['aws_values']
                        wind_direction = station_dict[station][0]['wind_direction']
                
                        # Angle wrap around problem
                        dir_radians = np.deg2rad(wind_direction)
                        dir_sin = np.sin(dir_radians)
                        dir_cos = np.cos(dir_radians)
                        
                       
                        # Setup powers
                        X_Z = np.zeros((len(aws_values), len(bands)))
                        X_NS = np.zeros((len(aws_values), len(bands)))
                        X_EW = np.zeros((len(aws_values), len(bands)))
                
                        # Apply bandwidths to data
                        for i, (f1,f2) in enumerate(bands):
                            band_width = (freq >= f1) & (freq < f2)
                
                            # Convert to log to better inspect power scales and apply bandwidth
                            # [:, band_width], select frequencies and slice unwanted freq data from the row
                            # .mean(axis=1), mean for the selected frequency row. Reshape for model input.
                            X_Z[:, i] = np.log10(Z_power[:, band_width].mean(axis=1) + 1e-20)
                            X_NS[:, i] = np.log10(NS_power[:, band_width].mean(axis=1) + 1e-20)
                            X_EW[:, i] = np.log10(EW_power[:, band_width].mean(axis=1) + 1e-20)
                
                        # Stack all together to process all together as a larger dataset
                        X_all = np.column_stack([X_Z, X_NS, X_EW])
                        y_all = np.column_stack([aws_values, dir_sin, dir_cos])
                
                
                        # Should the wind speed threshold be applied before or after training the model?
                        # Apply minimum WS threshold
                        if min_WS is not None:
                            mask = y_all[:, 0] > min_WS
                            X_all = X_all[mask]
                            y_all = y_all[mask]
                            
                        # Train Model
                        X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
                
                        # Find best alpha
                        
                        # setup parameter lists
                        c_list = np.logspace(-4, 7, 10)
                        gamma_list = np.logspace(-4, 4, 10)
                        epsilon_list = np.linspace(0.05, 3, 15)
                
                        best_var_r2 = -np.inf
                        best_sin_r2 = -np.inf
                        best_cos_r2 = -np.inf
                
                        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                
                        for c in c_list:
                            for g in gamma_list:
                                for e in epsilon_list:
                                    # Pipeline
                                    # Scale, SVR
                                    var_svr = Pipeline([('scaler', StandardScaler()),('svr', SVR(C=c, gamma = g, epsilon=e))])
                                    sin_svr = Pipeline([('scaler', StandardScaler()),('svr', SVR(C=c, gamma = g, epsilon=e))])
                                    cos_svr = Pipeline([('scaler', StandardScaler()),('svr', SVR(C=c, gamma = g, epsilon=e))])
                
                                    # Cross Validate
                                    var_score = cross_val_score(var_svr, X_all_train, y_all_train[:, 0], cv=cv, scoring ='r2', n_jobs=-1)
                                    sin_score = cross_val_score(sin_svr, X_all_train, y_all_train[:, 1], cv=cv, scoring ='r2', n_jobs=-1)
                                    cos_score = cross_val_score(cos_svr, X_all_train, y_all_train[:, 2], cv=cv, scoring ='r2', n_jobs=-1)
                        
                                    # CV R2
                                    var_mean_r2 = var_score.mean()
                                    sin_mean_r2 = sin_score.mean()
                                    cos_mean_r2 = cos_score.mean()
                
                                    # Note Parameters
                                    params = [c, g, e]
                
                                    # Is it better than previous iteration? If yes then save params
                                    # Save best var model
                                    if var_mean_r2 > best_var_r2:
                                        best_var_r2 = var_mean_r2
                                        best_var_params = params.copy()
                
                                    # Save best sin2 model
                                    if sin_mean_r2 > best_sin_r2:
                                        best_sin_r2 = sin_mean_r2
                                        best_sin_params = params.copy()
                
                                    # Save best cos2 model
                                    if cos_mean_r2 > best_cos_r2:
                                        best_cos_r2 = cos_mean_r2
                                        best_cos_params = params.copy()
                
                        print("WS:")
                        print("C:", best_var_params[0])
                        print("gamma:", best_var_params[1])
                        print("epsilon:", best_var_params[2])
                        print("CV R²:", best_var_r2)
                        print("--------------------------------------------------")
                        print("Sin2:")
                        print("C:", best_sin_params[0])
                        print("gamma:", best_sin_params[1])
                        print("epsilon:", best_sin_params[2])
                        print("CV R²:", best_sin_r2)
                        print("--------------------------------------------------")
                        print("Cos2:")
                        print("C:", best_cos_params[0])
                        print("gamma:", best_cos_params[1])
                        print("epsilon:", best_cos_params[2])
                        print("CV R²:", best_cos_r2)

                def optimise_RF(self,
                                fmin = 1,
                                fmax = 49,
                                f_band_width = 1,
                                step_size = 1,
                                n_splits = 5,
                                min_WS = None,
                                WS_only = True):
                    
                        # Bands
                        # Create Bandwidths
                        bands = []
                        for f1 in range(fmin, fmax - f_band_width + 1, step_size):
                            f2 = f1 + f_band_width
                            band = (f1, f2)
                            bands.append(band)
                        # Create Band Centers
                        band_centres = []
                        for f1, f2 in bands:
                            band_centre = [(f1 + f2)/2]
                            band_centres.append(band_centre)
                    
                        # Loop through stations
                        for station_dict in self.seismic_wind.fft:
                    
                            # Setup Variables
                            station = list(station_dict.keys())[0] 
                            EW_power = station_dict[station][0]['EW']
                            NS_power = station_dict[station][0]['NS']
                            Z_power = station_dict[station][0]['Z']
                            freq = station_dict[station][0]['freq']
                            aws_values = station_dict[station][0]['aws_values']
                            wind_direction = station_dict[station][0]['wind_direction']
                    
                            # Angle wrap around problem
                            dir_radians = np.deg2rad(wind_direction)
                            dir_sin = np.sin(dir_radians)
                            dir_cos = np.cos(dir_radians)
                            
                            # Setup results
                            station_results = []
                            
                            # Setup powers
                            X_Z = np.zeros((len(aws_values), len(bands)))
                            X_NS = np.zeros((len(aws_values), len(bands)))
                            X_EW = np.zeros((len(aws_values), len(bands)))
                    
                            # Apply bandwidths to data
                            for i, (f1,f2) in enumerate(bands):
                                band_width = (freq >= f1) & (freq < f2)
                    
                                # Convert to log to better inspect power scales and apply bandwidth
                                # [:, band_width], select frequencies and slice unwanted freq data from the row
                                # .mean(axis=1), mean for the selected frequency row. Reshape for model input.
                                X_Z[:, i] = np.log10(Z_power[:, band_width].mean(axis=1) + 1e-20)
                                X_NS[:, i] = np.log10(NS_power[:, band_width].mean(axis=1) + 1e-20)
                                X_EW[:, i] = np.log10(EW_power[:, band_width].mean(axis=1) + 1e-20)
                    
                            # Stack all together to process all together as a larger dataset
                            X_all = np.column_stack([X_Z, X_NS, X_EW])
                            y_all = np.column_stack([aws_values, dir_sin, dir_cos])
                    
                    
                            # Should the wind speed threshold be applied before or after training the model?
                            # Apply minimum WS threshold
                            if min_WS is not None:
                                mask = y_all[:, 0] > min_WS
                                X_all = X_all[mask]
                                y_all = y_all[mask]
                                
                            # Train Model
                            X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
                    
                            # Find best paramters
                            # setup parameter lists
                            n_estimators = [50, 100, 150, 200]
                            depths = [5, 10, 15, None]
                            min_samp_split = [2, 5, 10]
                            min_samp_leaf = [1, 3, 5]
                            max_feat = [0.25, 0.5, 0.75, 1.0]
                    
                            best_var_r2 = -np.inf
                            if WS_only == False:
                                best_sin_r2 = -np.inf
                                best_cos_r2 = -np.inf
                    
                            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                    
                            for n_est in n_estimators:
                                for d in depths:
                                    for min_split in min_samp_split:
                                        for min_leaf in min_samp_leaf:
                                            for feat in max_feat:
                                                # Random Forest
                                                var_rf = RandomForestRegressor(n_estimators=n_est, max_depth=d,
                                                                               min_samples_split=min_split,min_samples_leaf=min_leaf,
                                                                               max_features=feat, random_state=42, n_jobs=-1)
                                                if WS_only == False:
                                                    sin_rf = RandomForestRegressor(n_estimators=n_est, max_depth=d,
                                                                                    min_samples_split=min_split,min_samples_leaf=min_leaf,
                                                                                    max_features=feat, random_state=42, n_jobs=-1)
                                                    cos_rf = RandomForestRegressor(n_estimators=n_est, max_depth=d,
                                                                                    min_samples_split=min_split,min_samples_leaf=min_leaf,
                                                                                    max_features=feat, random_state=42, n_jobs=-1)
                    
                                                # Cross Validate
                                                var_score = cross_val_score(var_rf, X_all_train, y_all_train[:, 0], cv=cv, scoring ='r2', n_jobs=-1)
                                                if WS_only == False:
                                                    sin_score = cross_val_score(sin_rf, X_all_train, y_all_train[:, 1], cv=cv, scoring ='r2', n_jobs=-1)
                                                    cos_score = cross_val_score(cos_rf, X_all_train, y_all_train[:, 2], cv=cv, scoring ='r2', n_jobs=-1)
                                    
                                                # CV R2
                                                var_mean_r2 = var_score.mean()
                                                if WS_only == False:
                                                    sin_mean_r2 = sin_score.mean()
                                                    cos_mean_r2 = cos_score.mean()
                            
                                                # Note Parameters
                                                params = [n_est, d, min_split, min_leaf, feat]
                            
                                                # Is it better than previous iteration? If yes then save params
                                                # Save best var model
                                                if var_mean_r2 > best_var_r2:
                                                    best_var_r2 = var_mean_r2
                                                    best_var_params = params.copy()
                    
                                                if WS_only == False:
                                                    # Save best sin2 model
                                                    if sin_mean_r2 > best_sin_r2:
                                                        best_sin_r2 = sin_mean_r2
                                                        best_sin_params = params.copy()
                                
                                                    # Save best cos2 model
                                                    if cos_mean_r2 > best_cos_r2:
                                                        best_cos_r2 = cos_mean_r2
                                                        best_cos_params = params.copy()
                                        
                    
                            print("WS:")
                            print("n_estimators:", best_var_params[0])
                            print("depths:", best_var_params[1])
                            print("min_sample_split:", best_var_params[2])
                            print("min_sample_leaf:", best_var_params[3])
                            print("max_features:", best_var_params[4])
                            print("CV R²:", best_var_r2)
                            print("--------------------------------------------------")
                            if WS_only == False:
                                print("Sin2:")
                                print("n_estimators:", best_sin_params[0])
                                print("depths:", best_sin_params[1])
                                print("min_sample_split:", best_sin_params[2])
                                print("min_sample_leaf:", best_sin_params[3])
                                print("max_features:", best_sin_params[4])
                                print("CV R²:", best_sin_r2)
                                print("--------------------------------------------------")
                                print("n_estimators:", best_cos_params[0])
                                print("depths:", best_cos_params[1])
                                print("min_sample_split:", best_cos_params[2])
                                print("min_sample_leaf:", best_cos_params[3])
                                print("max_features:", best_cos_params[4])
                                print("CV R²:", best_cos_r2)

                # Model Analysis
                def RF(self,
                       fmin = 1,
                       fmax = 49,
                       f_band_width = 1,
                       step_size = 1,
                       n_repeats = 3,
                       min_WS = None,
                       poly_degree = 2,
                       plot_stat_results = True,
                       plot_results = True,
                       plot_residuals = True,
                       plot_power_aws = True,
                       variable_name = 'AWS Wind Speed (km/hr)'):
                    
                    """
                    Predicts AWS variable and Wind Direction from multiple seismic frequency band power features 
                    using a Random Forest regression model. Combines all seismic components into one model.
                    Includes a y_test array of AWS and WD. 
                    Multi Output model.
                
                    Parameters:
                        fmin (int):
                            Minimum frequency value for calculating the power of each bandwidths.
                        fmax (int):
                            Maximum frequency value for calculating the power of each  bandwidths.
                        f_band_width (int):
                            Bandwidth size. e.g. f_band_width = 1 for (f1,f2)=(1,2), 2 for (1,3), 3 for (1,4). 
                        step_size (int):
                            Frequency band step size. Set to less than f_band_width for overlapping bands. 
                        n_repeats (int):
                            Number of times to repeat the permutation importance calculation for each seismic component.
                        min_WS (int):
                            Minimum wind speed (km/hr) to be considered in analysis.
                        poly_degree (int):
                            Polynomial degree for the regression in plot_power_aws (Seis Power vs Obs WS)
                        plot_stat_results (bool):
                            Plots all the R² and rmse values against frequency bandwidth centres.
                        plot_results (bool):
                            Plots the predicted vs observed values for each seismic component.
                        plot_cv (bool):
                            Plot the cross validation results as boxplots       
                        plot_power_aws (bool):
                            Plot the relationship between seismic power and AWS values.
                        variable_name (str):
                            The name of the variable being predicted (e.g., 'AWS Wind Speed (km/hr)').
                
                        Outputs:
                            results (list):
                                A list of dictionaries containing information about all the correlation results for each station.
                        """
                        
                    # Setup Result Lists
                    results = []
                
                    # Bands
                    # Create Bandwidths
                    bands = []
                    for f1 in range(fmin, fmax - f_band_width + 1, step_size):
                        f2 = f1 + f_band_width
                        band = (f1, f2)
                        bands.append(band)
                    # Num of Bands
                    n_bands = len(bands)
                    # Create Band Centers
                    band_centres = []
                    for f1, f2 in bands:
                        band_centre = [(f1 + f2)/2]
                        band_centres.append(band_centre)
                
                    # Loop through stations
                    for station_dict in self.seismic_wind.fft:
                
                        # Setup Variables
                        station = list(station_dict.keys())[0] 
                        EW_power = station_dict[station][0]['EW']
                        NS_power = station_dict[station][0]['NS']
                        Z_power = station_dict[station][0]['Z']
                        freq = station_dict[station][0]['freq']
                        aws_values = station_dict[station][0]['aws_values']
                        wind_direction = station_dict[station][0]['wind_direction']
                
                        # Angle wrap around problem
                        dir_radians = np.deg2rad(wind_direction)
                        dir_sin = np.sin(dir_radians)
                        dir_cos = np.cos(dir_radians)
                        
                        # Setup results
                        station_results = []
                        
                        # Setup powers
                        X_Z = np.zeros((len(aws_values), len(bands)))
                        X_NS = np.zeros((len(aws_values), len(bands)))
                        X_EW = np.zeros((len(aws_values), len(bands)))
                
                        # Apply bandwidths to data
                        for i, (f1,f2) in enumerate(bands):
                            band_width = (freq >= f1) & (freq < f2)
                
                            # Convert to log to better inspect power scales and apply bandwidth
                            # [:, band_width], select frequencies and slice unwanted freq data from the row
                            # .mean(axis=1), mean for the selected frequency row. Reshape for model input.
                            X_Z[:, i] = np.log10(Z_power[:, band_width].mean(axis=1) + 1e-20)
                            X_NS[:, i] = np.log10(NS_power[:, band_width].mean(axis=1) + 1e-20)
                            X_EW[:, i] = np.log10(EW_power[:, band_width].mean(axis=1) + 1e-20)
                
                        # Stack all together to process all together as a larger dataset
                        X_all = np.column_stack([X_Z, X_NS, X_EW])
                        y_all = np.column_stack([aws_values, dir_sin, dir_cos])
                        
                        # Should the wind speed threshold be applied before or after training the model?
                        # Apply minimum WS threshold
                        if min_WS is not None:
                            mask = y_all[:, 0] > min_WS
                            X_all = X_all[mask]
                            y_all = y_all[mask]
                            
                        # Train Model
                        X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
                
                        # Define Random Forest Models
                        var_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
                        sin_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
                        cos_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
                
                        # Cross Validation
                        cv = KFold(n_splits=5, shuffle=True, random_state=42)
                        var_scores = cross_val_score(var_model, X_all_train, y_all_train[:, 0], cv=cv, scoring='r2')
                        sin_scores = cross_val_score(sin_model, X_all_train, y_all_train[:, 1], cv=cv, scoring='r2')
                        cos_scores = cross_val_score(cos_model, X_all_train, y_all_train[:, 2], cv=cv, scoring='r2')
                
                        # Fit Model
                        var_model.fit(X_all_train, y_all_train[:,0])
                        sin_model.fit(X_all_train, y_all_train[:, 1])
                        cos_model.fit(X_all_train, y_all_train[:, 2])
                
                        # Predictions
                        var_pred = var_model.predict(X_all_test)
                        sin_pred = sin_model.predict(X_all_test)
                        cos_pred = cos_model.predict(X_all_test)
                
                        # Round WS Prediction to one decimal so it matches with AWS Measurement
                        var_pred = np.round(var_pred, decimals=1)
                
                        # Observed
                        y_var_test = y_all_test[:, 0]
                        y_sin_test = y_all_test[:, 1]
                        y_cos_test = y_all_test[:, 2]
                
                    
                        # Predicted direction
                        dir_pred_radian = np.arctan2(sin_pred, cos_pred)
                        dir_pred = np.rad2deg(dir_pred_radian)
                
                        # Round to nearest 10 degrees
                        dir_pred = np.round(dir_pred, decimals=-1)
                        
                        # Wrap back to 0-360 degrees
                        dir_pred = dir_pred % 360 
                
                        # Fix angles near end points
                        dir_test_radian = np.arctan2(y_sin_test, y_cos_test)
                        dir_test = np.rad2deg(dir_test_radian) % 360
                        dir_fix = np.abs((dir_pred - dir_test + 180) % 360 - 180)
                
                        # Mean direction
                        dir_mean = np.mean(dir_fix)
                
                        # rmse
                        var_rmse = np.sqrt(mean_squared_error(y_var_test, var_pred))
                        sin_rmse = np.sqrt(mean_squared_error(y_sin_test, sin_pred))
                        cos_rmse = np.sqrt(mean_squared_error(y_cos_test, cos_pred))
                        dir_rmse = np.sqrt(np.mean(dir_fix**2))
                
                        # R²
                        var_r2 = r2_score(y_var_test, var_pred)
                        sin_r2 = r2_score(y_sin_test, sin_pred)
                        cos_r2 = r2_score(y_cos_test, cos_pred)
                
                        # Freq Importance
                        var_importance = permutation_importance(var_model, X_all_test, y_var_test, n_repeats=n_repeats, scoring='r2').importances_mean
                        sin_importance = permutation_importance(sin_model, X_all_test, y_sin_test, n_repeats=n_repeats, scoring='r2').importances_mean
                        cos_importance = permutation_importance(cos_model, X_all_test, y_cos_test, n_repeats=n_repeats, scoring='r2').importances_mean
                        #dir_importance = np.sqrt(sin_importance**2 + cos_importance**2)
                
                        # Store results
                        station_results.append({
                            'fmin': fmin,
                            'fmax': fmax,
                            'var_r2': var_r2,
                            'sin_r2': sin_r2,
                            'cos_r2': cos_r2,
                            'var_rmse': var_rmse,
                            'sin_rmse': sin_rmse,
                            'cos_rmse': cos_rmse,
                            'y_var_test': y_var_test,
                            'y_sin_test': y_sin_test,
                            'y_cos_test': y_cos_test,
                            'var_pred': var_pred,
                            'sin_pred': sin_pred,
                            'cos_pred': cos_pred,
                            'var_cv_r2': var_scores.mean(),
                            'sin_cv_r2': sin_scores.mean(),
                            'cos_cv_r2': cos_scores.mean(),
                            'var_cv_std': var_scores.std(),
                            'sin_cv_std': sin_scores.std(),
                            'cos_cv_std': cos_scores.std(),
                            'dir_pred': dir_pred,
                            'dir_test': dir_test,
                            'dir_mean_error': dir_mean,
                            'dir_rmse': dir_rmse}) 
                        
                        # Print best result
                        # R²
                        print(f"{variable_name} R²: {var_r2:.4f}")
                        print(f"Sin R²: {sin_r2:.4f}")
                        print(f"Cos R²: {cos_r2:.4f}")
                
                        # rmse
                        print(f"{variable_name} rmse: {var_rmse:.4f}")
                        print(f"Sin rmse: {sin_rmse:.4f}")
                        print(f"Cos rmse: {cos_rmse:.4f}")
                
                        # Cross Validation R²
                        print(f"{variable_name} cv R²: {var_scores.mean():.4f} +/- {var_scores.std():.4f}")
                        print(f"Sin cv R²: {sin_scores.mean():.4f} +/- {sin_scores.std():.4f}")
                        print(f"Cos cv R²: {cos_scores.mean():.4f} +/- {cos_scores.std():.4f}")
                        results.append(station_results)
                
                        # WD
                        print(f"Wind direction Mean Error: {dir_mean:.2f}°")
                        print(f"Wind direction RMSE: {dir_rmse:.2f}°")
                
                        # Plot all average R² and rmse results against centre frequency
                        if plot_stat_results == True:
                            
                            # AWS Variable
                            Z_var_importance = var_importance[:n_bands]
                            NS_var_importance = var_importance[n_bands:2*n_bands]
                            EW_var_importance = var_importance[2*n_bands:3*n_bands]
                            # AWS Direction
                            #Z_dir_importance = dir_importance[:n_bands]
                            #NS_dir_importance = dir_importance[n_bands:2*n_bands]
                            #EW_dir_importance = dir_importance[2*n_bands:3*n_bands]
                
                            # Plot
                            fig, ax = plt.subplots(1, 3, figsize=(8, 6))
                            ax[0].plot(band_centres, Z_var_importance)
                            ax[0].set_title(f'Z {variable_name} \n Permutation Importance \n Acrosss Freq Spectrum')
                            ax[0].set_ylim(top=1)
                            ax[1].plot(band_centres, NS_var_importance)
                            ax[1].set_title(f'NS {variable_name} \n Permutation Importance \n Acrosss Freq Spectrum')
                            ax[1].set_ylim(top=1)
                            ax[2].plot(band_centres, EW_var_importance)
                            ax[2].set_title(f'EW {variable_name} \n Permutation Importance \n Acrosss Freq Spectrum')
                            ax[2].set_ylim(top=1)
                            #ax[1, 0].plot(band_centres, Z_dir_importance)
                            #ax[1, 0].set_title(f'Combined Z Wind Direction \n Permutation Importance \n Acrosss Freq Spectrum')
                            #ax[1, 0].set_ylim(top=1)
                            #ax[1, 1].plot(band_centres, NS_dir_importance)
                            #ax[1, 1].set_title(f'Combined NS Wind Direction \n Permutation Importance \n Acrosss Freq Spectrum')
                            #ax[1, 1].set_ylim(top=1)
                            #ax[1, 2].plot(band_centres, EW_dir_importance)
                            #ax[1, 2].set_title(f'Combined EW Wind Direction \n Permutation Importance \n Acrosss Freq Spectrum')
                            #ax[1, 2].set_ylim(top=1)
                
                            fig.suptitle(f'{station}:')
                            fig.supxlabel('Frequency (Hz)')
                            fig.supylabel('RF Permutation Importance')
                            fig.tight_layout()
                
                        # Plot key results
                        if plot_results == True:
                
                            # Plot obs vs pred WS
                            # AWS Variable
                            # Plot
                            plt.figure(figsize=(10, 10))
                            plt.scatter(y_var_test, var_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                
                            # y = x line
                            plt.plot([y_var_test.min(), y_var_test.max()],
                                    [y_var_test.min(), y_var_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            # Make it pretty
                            plt.xlabel(f"Observed {variable_name}", fontsize = 20)
                            plt.ylabel(f"Predicted {variable_name}", fontsize = 20)
                            plt.title(f"{station} {variable_name}", fontsize = 25)
                            plt.xticks(fontsize = 20)
                            plt.yticks(fontsize = 20)
                            plt.legend(fontsize = 20, loc ='upper left')
                            plt.tight_layout()
                
                            # Plot obs vs pred Wind Direction
                            # sin2theta and cos2theta
                            fig, ax = plt.subplots(2,1, figsize=(10, 10))
                            ax[0].scatter(y_sin_test, sin_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                            ax[1].scatter(y_cos_test, cos_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                            
                            # y = x line
                            ax[0].plot([y_sin_test.min(), y_sin_test.max()],
                                    [y_sin_test.min(), y_sin_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            ax[1].plot([y_cos_test.min(), y_cos_test.max()],
                                    [y_cos_test.min(), y_cos_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            # Make it pretty
                            ax[0].set_xlabel("Observed Wind Direction Sin2Theta", fontsize = 20)
                            ax[0].set_ylabel("Predicted Wind Direction Sin2Theta", fontsize = 20)
                            ax[0].set_title(f"{station} Wind Direction Sin2Theta", fontsize = 25)
                            ax[1].set_xlabel("Observed Wind Direction Cos2Theta", fontsize = 20)
                            ax[1].set_ylabel("Predicted Wind Direction Cos2Theta", fontsize = 20)
                            ax[1].set_title(f"{station} Wind Direction Cos2Theta", fontsize = 25)
                            ax[0].tick_params(axis='both', which='major', labelsize=20)
                            ax[1].tick_params(axis='both', which='major', labelsize=20)
                            fig.tight_layout()
                
                            # Sin2 and Cos2 Converted back to angle
                            plt.figure(figsize=(10, 10))
                            plt.scatter(dir_test, dir_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                            
                            # y = x line
                            plt.plot([dir_test.min(), dir_test.max()],
                                    [dir_test.min(), dir_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            # Make it pretty
                            plt.xlabel("Observed Wind Direction (°)", fontsize = 20)
                            plt.ylabel("Predicted Wind Direction (°)", fontsize = 20)
                            plt.title(f"{station} Wind Direction (°)", fontsize = 25)
                            plt.xticks(fontsize = 20)
                            plt.yticks(fontsize = 20)
                            plt.legend(fontsize = 20, loc ='upper left')
                            plt.tight_layout()
                
                            # Plot Wind Direction
                            # Wind direction
                            cardinals = {"N": 0,           
                                        "E": (np.pi / 2),
                                        "S": (np.pi),
                                        "W": (3 * np.pi / 2)}
                            # Plot
                            plt.figure(figsize=(10, 6))
                            plt.polar()      
                            plt.scatter(dir_test_radian, np.ones(len(dir_test_radian)) * 0.8, alpha=0.7, label = 'Observed Wind Direction')
                            plt.scatter(np.deg2rad(dir_pred), np.ones(len(dir_test_radian)) * 0.85, alpha=0.7, label = 'Predicted Wind Direction')
                
                            # Make it Pretty
                            plt.gca().set_theta_zero_location('N')
                            plt.gca().set_theta_direction(-1)
                            plt.title(f"{station} Wind Direction", pad = 50)
                            plt.gca().set_rlabel_position(0)
                            plt.ylim(0,1)
                            # Add cardinal direction labels
                            for label, angle in cardinals.items():
                                            plt.text(
                                                angle,
                                                1.2,
                                                label,
                                                ha="center",
                                                va="center",
                                                fontsize=12,
                                                fontweight="bold",
                                                clip_on=False)
                            # Legend
                            plt.legend(loc = 'upper right', bbox_to_anchor = (1.55, 1.18))
                            # Design
                            plt.tight_layout()
                
                            # Plot WS, WD, Seis Power
                            # AWS Variable and Wind direction
                
                            # Best frequency index for seismic power
                            var_best = np.argmax(var_importance)
                
                            if var_best < n_bands:
                
                                best_component = "Z"
                                best_index = var_best
                                best_var = X_all_test[:, var_best]
                
                            elif var_best < 2*n_bands:
                
                                best_component = "NS"
                                best_index = var_best - n_bands
                                best_var = X_all_test[:, var_best]
                
                            else:
                
                                best_component = "EW"
                                best_index = var_best - 2*n_bands
                                best_var = X_all_test[:, var_best]
                
                            # Best Centre frequencies
                            freq_best = band_centres[best_index]
                
                            # Create plot
                            plt.figure(figsize=(10,6))
                            ax = plt.subplot(projection='polar')
                            plt.polar()
                            # Colour bar
                            power_colours = best_var
                            cmap = matplotlib.cm.get_cmap('plasma')
                            new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('snipped_cmap', cmap(np.linspace(0, 0.90, 256)))
                            #cmap2 = matplotlib.cm.get_cmap('cool') # For error magnitude
                            # Plot measurements
                            obs = ax.scatter(dir_test_radian, y_var_test, alpha=0.7, c=power_colours, cmap=new_cmap, s = 60, label = f'Observed {variable_name}', marker='x')
                            pred = ax.scatter(np.deg2rad(dir_pred), var_pred, alpha=0.7, c=power_colours, cmap=new_cmap, s = 60, label = f'Predicted {variable_name}', marker='^')
                            # Plot error mag line between pred and obs point
                
                            # WS error
                            # Not useful yet, but could be used to colour the line between obs and pred points
                            #ws_error = np.abs(y_var_test - var_pred)
                            # Combined error magnitude. Cant just combine WS and WD so left out for now
                            #error_mag = np.sqrt(dir_fix**2 + ws_error**2)
                            #error_norm = matplotlib.colors.Normalize(vmin=error_mag.min(), vmax=error_mag.max())
                
                            for i in range(len(dir_test_radian)):
                                ax.plot([dir_test_radian[i], np.deg2rad(dir_pred[i])], [y_var_test[i], var_pred[i]], color='grey', alpha=1, linewidth=0.8)
                            # Make it pretty
                            ax.set_theta_zero_location('N')
                            ax.set_theta_direction(-1)
                            ax.set_ylabel(f'{variable_name}', labelpad=55, fontsize = 12)
                            ax.set_title(f"{station}: {variable_name} and Wind Direction", pad = 60, fontsize = 25)
                            ax.set_rlabel_position(0)
                            ax.tick_params(labelsize = 12)
                            rmax = ax.get_rmax()
                            # Add cardinal direction labels
                            for label, angle in cardinals.items():
                                            ax.text(
                                                angle,
                                                rmax * 1.3,
                                                label,
                                                ha="center",
                                                va="center",
                                                fontsize=15,
                                                fontweight="bold",
                                                clip_on=False)
                            # More pretty
                            plt.legend(loc = 'upper right', bbox_to_anchor = (1.45, 1.2), fontsize = 8) 
                            # Seismic Colour Bar
                            cbar = plt.colorbar(obs, ax=ax, pad=0.1)
                            cbar.set_label(f'Log Seismic Power\n{best_component}, {freq_best[0]:.1f} Hz', size = 12)
                            # Error Colour Bar
                            # Need to think of a good way to represent error well. Cant just combine WS and WD
                            #sm = matplotlib.cm.ScalarMappable(norm=error_norm, cmap=cmap2)
                            #sm.set_array([])   # required placeholder
                            #cbar2 = plt.colorbar(sm, ax=ax, pad=0.1)
                            #cbar2.set_label("Error Magnitude", fontsize=12)
                
                                                        
                            cbar.ax.tick_params(labelsize=12)
                            #cbar2.ax.tick_params(labelsize=12)
                            plt.tight_layout()
                
                        # Plot WS Residiuals
                        if plot_residuals == True:
                
                            fig, ax = plt.subplots(1, 3, figsize=(8, 8))
                            ax[0].scatter(var_pred, y_var_test - var_pred)
                            ax[0].axhline(0, color='red', linestyle='--')
                            ax[0].set_xlabel(f"Predicted {variable_name}")
                            ax[0].set_ylabel("Residual (Observed - Predicted)")
                            ax[0].set_title(f"Residual Plot for {station}, {variable_name}")
                            ax[1].scatter(sin_pred, y_sin_test - sin_pred)
                            ax[1].axhline(0, color='red', linestyle='--')
                            ax[1].set_xlabel("Predicted Wind Direction Sin")
                            ax[1].set_ylabel("Residual (Observed - Predicted)")
                            ax[1].set_title(f"Residual Plot for {station}, Wind Direction Sin")
                            ax[2].scatter(cos_pred, y_cos_test - cos_pred)
                            ax[2].axhline(0, color='red', linestyle='--')
                            ax[2].set_xlabel("Predicted Wind Direction Cos")
                            ax[2].set_ylabel("Residual (Observed - Predicted)")
                            ax[2].set_title(f"Residual Plot for {station}, Wind Direction Cos")
                            fig.tight_layout()
                
                        # Plot Observed WS and seis power
                        if plot_power_aws == True:
                
                            # LogSeismic Power vs Obs AWS  
                
                            # Best frequency index for power
                            var_best = np.argmax(var_importance)
                
                            if var_best < n_bands:
                
                                best_component = "Z"
                                best_index = var_best
                                best_var = X_all_test[:, var_best]
                
                                
                            elif var_best < 2*n_bands:
                
                                best_component = "NS"
                                best_index = var_best - n_bands
                                best_var = X_all_test[:, var_best]
                
                            
                            else:
                
                                best_component = "EW"
                                best_index = var_best - 2*n_bands
                                best_var = X_all_test[:, var_best]
                
                                
                            # Best Centre frequencies
                            freq = band_centres[best_index]
                
                            # Sort for plot
                            idx = np.argsort(best_var)
                            x_sort = best_var[idx]
                
                            # Polynomial 
                            # Reshape x values
                            poly_x = best_var.reshape(-1,1)
                            # Create Pipeline 
                            poly = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression()).fit(poly_x, y_var_test)
                            # Calculate R² 
                            poly_r2 = poly.score(poly_x, y_var_test)
                            # Predict
                            poly_pred = poly.predict(poly_x)[idx]
                            # Coefficients
                            linear_model = poly.named_steps['linearregression']
                            coefficients = linear_model.coef_
                            intercept = linear_model.intercept_
                            # Equation
                            if poly_degree == 1:
                                equation = f"y = {coefficients[1]:.2f}x + {intercept:2f}"
                            elif poly_degree == 2:
                                equation = f"y = {coefficients[2]:.2f}x$^{2}$ + {coefficients[1]:.2f}x + {intercept:.2f}"
                            else:
                                print('Calculation only for poly_degree = 1 or 2')
                                equation = 'Equation not calculated'
                
                            # Create plots
                            plt.figure(figsize=(10,10))
                            plt.scatter(best_var, y_var_test, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                
                            # Plot
                            plt.plot(x_sort, poly_pred, 'r--', linewidth = 4, label = f'{equation} \nR² = {poly_r2:.2f}')
                            plt.title(f"{station}: {best_component}, {freq[0]:.1f} Hz", fontsize = 25)
                            # Make pretty
                            plt.legend(fontsize = 20)
                            plt.xlabel("Log Seismic Power", fontsize = 20)
                            plt.ylabel(f"Observed {variable_name}", fontsize = 20)
                            plt.xticks(fontsize = 20)
                            plt.yticks(fontsize = 20)
                            plt.tight_layout()
                
                        # Find best 10 frequency bands for each component     
                        top_ten = np.argsort(var_importance)[::-1][:10]
                
                        for index in top_ten:
                
                            if index < n_bands:
                                component = "Z"
                                freq_index = index
                
                            elif index < 2 * n_bands:
                                component = "NS"
                                freq_index = index - n_bands
                
                            else:
                                component = "EW"
                                freq_index = index - 2 * n_bands
                
                            frequency = band_centres[freq_index]
                
                            # Print them
                            print(f"{component}: {frequency[0]:.1f} Hz, importance = {var_importance[index]:.4f}")
                
                    return results

                def SVR(self,
                        fmin = 1,
                        fmax = 49,
                        f_band_width = 1,
                        step_size = 1,
                        n_repeats = 3,
                        min_WS = None,
                        poly_degree = 2,
                        plot_stat_results = True,
                        plot_results = True,
                        plot_residuals = True,
                        plot_power_aws = True,
                        variable_name = 'AWS Wind Speed (km/hr)'):

                    """
                        Predicts AWS variable and Wind Direction from multiple seismic frequency band power features 
                        using a SVR regression model. Combines all seismic components into one model.
                        Includes a y_test array of AWS and WD. 
                        Multi Output model.
                    
                        Parameters:
                            spectra (list):
                                A list of dictionaries containing the frequency, power, and Wind Speed data 
                                for each seismic component (EW, NS, Z) for each station and time period.
                            fmin (int):
                                Minimum frequency value for calculating the power of each bandwidths.
                            fmax (int):
                                Maximum frequency value for calculating the power of each  bandwidths.
                            f_band_width (int):
                                Bandwidth size. e.g. f_band_width = 1 for (f1,f2)=(1,2), 2 for (1,3), 3 for (1,4). 
                            step_size (int):
                                Frequency band step size. Set to less than f_band_width for overlapping bands. 
                            n_repeats (int):
                                Number of times to repeat the permutation importance calculation for each seismic component.
                            min_WS (int):
                                Minimum wind speed (km/hr) to be considered in analysis.
                            poly_degree (int):
                                Polynomial degree for the regression in plot_power_aws (Seis Power vs Obs WS)
                            plot_stat_results (bool):
                                Plots all the R² and rmse values against frequency bandwidth centres.
                            plot_results (bool):
                                Plots the predicted vs observed values for each seismic component.
                            plot_cv (bool):
                                Plot the cross validation results as boxplots       
                            plot_power_aws (bool):
                                Plot the relationship between seismic power and AWS values.
                            variable_name (str):
                                The name of the variable being predicted (e.g., 'AWS Wind Speed (km/hr)').
                    
                        Outputs:
                            results (list):
                                A list of dictionaries containing information about all the correlation results for each station.
                        """
                        
                    # Setup Result Lists
                    results = []
                
                    # Bands
                    # Create Bandwidths
                    bands = []
                    for f1 in range(fmin, fmax - f_band_width + 1, step_size):
                        f2 = f1 + f_band_width
                        band = (f1, f2)
                        bands.append(band)
                    # Num of Bands
                    n_bands = len(bands)
                    # Create Band Centers
                    band_centres = []
                    for f1, f2 in bands:
                        band_centre = [(f1 + f2)/2]
                        band_centres.append(band_centre)
                
                    # Loop through stations
                    for station_dict in self.seismic_wind.fft:
                
                        # Setup Variables
                        station = list(station_dict.keys())[0] 
                        EW_power = station_dict[station][0]['EW']
                        NS_power = station_dict[station][0]['NS']
                        Z_power = station_dict[station][0]['Z']
                        freq = station_dict[station][0]['freq']
                        aws_values = station_dict[station][0]['aws_values']
                        wind_direction = station_dict[station][0]['wind_direction']
                
                        # Angle wrap around problem
                        dir_radians = np.deg2rad(wind_direction)
                        dir_sin = np.sin(dir_radians)
                        dir_cos = np.cos(dir_radians)
                        
                        # Setup results
                        station_results = []
                        
                        # Setup powers
                        X_Z = np.zeros((len(aws_values), len(bands)))
                        X_NS = np.zeros((len(aws_values), len(bands)))
                        X_EW = np.zeros((len(aws_values), len(bands)))
                
                        # Apply bandwidths to data
                        for i, (f1,f2) in enumerate(bands):
                            band_width = (freq >= f1) & (freq < f2)
                
                            # Convert to log to better inspect power scales and apply bandwidth
                            # [:, band_width], select frequencies and slice unwanted freq data from the row
                            # .mean(axis=1), mean for the selected frequency row. Reshape for model input.
                            X_Z[:, i] = np.log10(Z_power[:, band_width].mean(axis=1) + 1e-20)
                            X_NS[:, i] = np.log10(NS_power[:, band_width].mean(axis=1) + 1e-20)
                            X_EW[:, i] = np.log10(EW_power[:, band_width].mean(axis=1) + 1e-20)
                
                        # Stack all together to process all together as a larger dataset
                        X_all = np.column_stack([X_Z, X_NS, X_EW])
                        y_all = np.column_stack([aws_values, dir_sin, dir_cos])
                        
                        # Should the wind speed threshold be applied before or after training the model?
                        # Apply minimum WS threshold
                        if min_WS is not None:
                            mask = y_all[:, 0] > min_WS
                            X_all = X_all[mask]
                            y_all = y_all[mask]
                            
                        # Train Model
                        X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
                
                        # Pipeline
                        # Scale, SVR
                        # (Need to find optimal values)
                        var_model = Pipeline([('scaler', StandardScaler()),('svr', SVR(C=1.0, epsilon=0.2))])
                        sin_model = Pipeline([('scaler', StandardScaler()),('svr', SVR(C=1.0, epsilon=0.2))])
                        cos_model = Pipeline([('scaler', StandardScaler()),('svr', SVR(C=1.0, epsilon=0.2))])
                
                        # Cross Validation
                        cv = KFold(n_splits=5, shuffle=True, random_state=42)
                        var_scores = cross_val_score(var_model, X_all_train, y_all_train[:, 0], cv=cv, scoring='r2')
                        sin_scores = cross_val_score(sin_model, X_all_train, y_all_train[:, 1], cv=cv, scoring='r2')
                        cos_scores = cross_val_score(cos_model, X_all_train, y_all_train[:, 2], cv=cv, scoring='r2')
                
                        # Fit Model
                        var_model.fit(X_all_train, y_all_train[:,0])
                        sin_model.fit(X_all_train, y_all_train[:, 1])
                        cos_model.fit(X_all_train, y_all_train[:, 2])
                
                        # Predictions
                        var_pred = var_model.predict(X_all_test)
                        sin_pred = sin_model.predict(X_all_test)
                        cos_pred = cos_model.predict(X_all_test)
                
                        # Round WS Prediction to one decimal so it matches with AWS Measurement
                        var_pred = np.round(var_pred, decimals=1)
                
                        # Observed
                        y_var_test = y_all_test[:, 0]
                        y_sin_test = y_all_test[:, 1]
                        y_cos_test = y_all_test[:, 2]
                
                    
                        # Predicted direction
                        dir_pred_radian = np.arctan2(sin_pred, cos_pred)
                        dir_pred = np.rad2deg(dir_pred_radian)
                
                        # Round to nearest 10 degrees
                        dir_pred = np.round(dir_pred, decimals=-1)
                        
                        # Wrap back to 0-360 degrees
                        dir_pred = dir_pred % 360 
                
                        # Fix angles near end points
                        dir_test_radian = np.arctan2(y_sin_test, y_cos_test)
                        dir_test = np.rad2deg(dir_test_radian) % 360
                        dir_fix = np.abs((dir_pred - dir_test + 180) % 360 - 180)
                
                        # Mean direction
                        dir_mean = np.mean(dir_fix)
                
                        # rmse
                        var_rmse = np.sqrt(mean_squared_error(y_var_test, var_pred))
                        sin_rmse = np.sqrt(mean_squared_error(y_sin_test, sin_pred))
                        cos_rmse = np.sqrt(mean_squared_error(y_cos_test, cos_pred))
                        dir_rmse = np.sqrt(np.mean(dir_fix**2))
                
                        # R²
                        var_r2 = r2_score(y_var_test, var_pred)
                        sin_r2 = r2_score(y_sin_test, sin_pred)
                        cos_r2 = r2_score(y_cos_test, cos_pred)
                
                        # Freq Importance
                        var_importance = permutation_importance(var_model, X_all_test, y_var_test, n_repeats=n_repeats, scoring='r2').importances_mean
                        sin_importance = permutation_importance(sin_model, X_all_test, y_sin_test, n_repeats=n_repeats, scoring='r2').importances_mean
                        cos_importance = permutation_importance(cos_model, X_all_test, y_cos_test, n_repeats=n_repeats, scoring='r2').importances_mean
                        #dir_importance = np.sqrt(sin_importance**2 + cos_importance**2)
                
                        # Store results
                        station_results.append({
                            'fmin': fmin,
                            'fmax': fmax,
                            'var_r2': var_r2,
                            'sin_r2': sin_r2,
                            'cos_r2': cos_r2,
                            'var_rmse': var_rmse,
                            'sin_rmse': sin_rmse,
                            'cos_rmse': cos_rmse,
                            'y_var_test': y_var_test,
                            'y_sin_test': y_sin_test,
                            'y_cos_test': y_cos_test,
                            'var_pred': var_pred,
                            'sin_pred': sin_pred,
                            'cos_pred': cos_pred,
                            'var_cv_r2': var_scores.mean(),
                            'sin_cv_r2': sin_scores.mean(),
                            'cos_cv_r2': cos_scores.mean(),
                            'var_cv_std': var_scores.std(),
                            'sin_cv_std': sin_scores.std(),
                            'cos_cv_std': cos_scores.std(),
                            'dir_pred': dir_pred,
                            'dir_test': dir_test,
                            'dir_mean_error': dir_mean,
                            'dir_rmse': dir_rmse}) 
                        
                        # Print best result
                        # R²
                        print(f"{variable_name} R²: {var_r2:.4f}")
                        print(f"Sin R²: {sin_r2:.4f}")
                        print(f"Cos R²: {cos_r2:.4f}")
                
                        # rmse
                        print(f"{variable_name} rmse: {var_rmse:.4f}")
                        print(f"Sin rmse: {sin_rmse:.4f}")
                        print(f"Cos rmse: {cos_rmse:.4f}")
                
                        # Cross Validation R²
                        print(f"{variable_name} cv R²: {var_scores.mean():.4f} +/- {var_scores.std():.4f}")
                        print(f"Sin cv R²: {sin_scores.mean():.4f} +/- {sin_scores.std():.4f}")
                        print(f"Cos cv R²: {cos_scores.mean():.4f} +/- {cos_scores.std():.4f}")
                        results.append(station_results)
                
                        # WD
                        print(f"Wind direction Mean Error: {dir_mean:.2f}°")
                        print(f"Wind direction RMSE: {dir_rmse:.2f}°")
                
                        # Plot all average R² and rmse results against centre frequency
                        if plot_stat_results == True:
                            
                            # AWS Variable
                            Z_var_importance = var_importance[:n_bands]
                            NS_var_importance = var_importance[n_bands:2*n_bands]
                            EW_var_importance = var_importance[2*n_bands:3*n_bands]
                            # AWS Direction
                            #Z_dir_importance = dir_importance[:n_bands]
                            #NS_dir_importance = dir_importance[n_bands:2*n_bands]
                            #EW_dir_importance = dir_importance[2*n_bands:3*n_bands]
                
                            # Plot
                            fig, ax = plt.subplots(1, 3, figsize=(8, 6))
                            ax[0].plot(band_centres, Z_var_importance)
                            ax[0].set_title(f'Z {variable_name} \n Permutation Importance \n Acrosss Freq Spectrum')
                            ax[0].set_ylim(top=1)
                            ax[1].plot(band_centres, NS_var_importance)
                            ax[1].set_title(f'NS {variable_name} \n Permutation Importance \n Acrosss Freq Spectrum')
                            ax[1].set_ylim(top=1)
                            ax[2].plot(band_centres, EW_var_importance)
                            ax[2].set_title(f'EW {variable_name} \n Permutation Importance \n Acrosss Freq Spectrum')
                            ax[2].set_ylim(top=1)
                            #ax[1, 0].plot(band_centres, Z_dir_importance)
                            #ax[1, 0].set_title(f'Combined Z Wind Direction \n Permutation Importance \n Acrosss Freq Spectrum')
                            #ax[1, 0].set_ylim(top=1)
                            #ax[1, 1].plot(band_centres, NS_dir_importance)
                            #ax[1, 1].set_title(f'Combined NS Wind Direction \n Permutation Importance \n Acrosss Freq Spectrum')
                            #ax[1, 1].set_ylim(top=1)
                            #ax[1, 2].plot(band_centres, EW_dir_importance)
                            #ax[1, 2].set_title(f'Combined EW Wind Direction \n Permutation Importance \n Acrosss Freq Spectrum')
                            #ax[1, 2].set_ylim(top=1)
                
                            fig.suptitle(f'{station}:')
                            fig.supxlabel('Frequency (Hz)')
                            fig.supylabel('RF Permutation Importance')
                            fig.tight_layout()
                
                        # Plot key results
                        if plot_results == True:
                
                            # Plot obs vs pred WS
                            # AWS Variable
                            # Plot
                            plt.figure(figsize=(10, 10))
                            plt.scatter(y_var_test, var_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                
                            # y = x line
                            plt.plot([y_var_test.min(), y_var_test.max()],
                                    [y_var_test.min(), y_var_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            # Make it pretty
                            plt.xlabel(f"Observed {variable_name}", fontsize = 20)
                            plt.ylabel(f"Predicted {variable_name}", fontsize = 20)
                            plt.title(f"{station} {variable_name}", fontsize = 25)
                            plt.xticks(fontsize = 20)
                            plt.yticks(fontsize = 20)
                            plt.legend(fontsize = 20, loc ='upper left')
                            plt.tight_layout()
                
                            # Plot obs vs pred Wind Direction
                            # sin2theta and cos2theta
                            fig, ax = plt.subplots(2,1, figsize=(10, 10))
                            ax[0].scatter(y_sin_test, sin_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                            ax[1].scatter(y_cos_test, cos_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                            
                            # y = x line
                            ax[0].plot([y_sin_test.min(), y_sin_test.max()],
                                    [y_sin_test.min(), y_sin_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            ax[1].plot([y_cos_test.min(), y_cos_test.max()],
                                    [y_cos_test.min(), y_cos_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            # Make it pretty
                            ax[0].set_xlabel("Observed Wind Direction Sin2Theta", fontsize = 20)
                            ax[0].set_ylabel("Predicted Wind Direction Sin2Theta", fontsize = 20)
                            ax[0].set_title(f"{station} Wind Direction Sin2Theta", fontsize = 25)
                            ax[1].set_xlabel("Observed Wind Direction Cos2Theta", fontsize = 20)
                            ax[1].set_ylabel("Predicted Wind Direction Cos2Theta", fontsize = 20)
                            ax[1].set_title(f"{station} Wind Direction Cos2Theta", fontsize = 25)
                            ax[0].tick_params(axis='both', which='major', labelsize=20)
                            ax[1].tick_params(axis='both', which='major', labelsize=20)
                            fig.tight_layout()
                
                            # Sin2 and Cos2 Converted back to angle
                            plt.figure(figsize=(10, 10))
                            plt.scatter(dir_test, dir_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                            
                            # y = x line
                            plt.plot([dir_test.min(), dir_test.max()],
                                    [dir_test.min(), dir_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            # Make it pretty
                            plt.xlabel("Observed Wind Direction (°)", fontsize = 20)
                            plt.ylabel("Predicted Wind Direction (°)", fontsize = 20)
                            plt.title(f"{station} Wind Direction (°)", fontsize = 25)
                            plt.xticks(fontsize = 20)
                            plt.yticks(fontsize = 20)
                            plt.legend(fontsize = 20, loc ='upper left')
                            plt.tight_layout()
                
                            # Plot Wind Direction
                            # Wind direction
                            cardinals = {"N": 0,           
                                        "E": (np.pi / 2),
                                        "S": (np.pi),
                                        "W": (3 * np.pi / 2)}
                            # Plot
                            plt.figure(figsize=(10, 6))
                            plt.polar()      
                            plt.scatter(dir_test_radian, np.ones(len(dir_test_radian)) * 0.8, alpha=0.7, label = 'Observed Wind Direction')
                            plt.scatter(np.deg2rad(dir_pred), np.ones(len(dir_test_radian)) * 0.85, alpha=0.7, label = 'Predicted Wind Direction')
                
                            # Make it Pretty
                            plt.gca().set_theta_zero_location('N')
                            plt.gca().set_theta_direction(-1)
                            plt.title(f"{station} Wind Direction", pad = 50)
                            plt.gca().set_rlabel_position(0)
                            plt.ylim(0,1)
                            # Add cardinal direction labels
                            for label, angle in cardinals.items():
                                            plt.text(
                                                angle,
                                                1.2,
                                                label,
                                                ha="center",
                                                va="center",
                                                fontsize=12,
                                                fontweight="bold",
                                                clip_on=False)
                            # Legend
                            plt.legend(loc = 'upper right', bbox_to_anchor = (1.55, 1.18))
                            # Design
                            plt.tight_layout()
                
                            # Plot WS, WD, Seis Power
                            # AWS Variable and Wind direction
                
                            # Best frequency index for seismic power
                            var_best = np.argmax(var_importance)
                
                            if var_best < n_bands:
                
                                best_component = "Z"
                                best_index = var_best
                                best_var = X_all_test[:, var_best]
                
                            elif var_best < 2*n_bands:
                
                                best_component = "NS"
                                best_index = var_best - n_bands
                                best_var = X_all_test[:, var_best]
                
                            else:
                
                                best_component = "EW"
                                best_index = var_best - 2*n_bands
                                best_var = X_all_test[:, var_best]
                
                            # Best Centre frequencies
                            freq_best = band_centres[best_index]
                
                            # Create plot
                            plt.figure(figsize=(10,6))
                            ax = plt.subplot(projection='polar')
                            plt.polar()
                            # Colour bar
                            power_colours = best_var
                            cmap = matplotlib.cm.get_cmap('plasma')
                            new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('snipped_cmap', cmap(np.linspace(0, 0.90, 256)))
                            #cmap2 = matplotlib.cm.get_cmap('cool') # For error magnitude
                            # Plot measurements
                            obs = ax.scatter(dir_test_radian, y_var_test, alpha=0.7, c=power_colours, cmap=new_cmap, s = 60, label = f'Observed {variable_name}', marker='x')
                            pred = ax.scatter(np.deg2rad(dir_pred), var_pred, alpha=0.7, c=power_colours, cmap=new_cmap, s = 60, label = f'Predicted {variable_name}', marker='^')
                            # Plot error mag line between pred and obs point
                
                            # WS error
                            # Not useful yet, but could be used to colour the line between obs and pred points
                            #ws_error = np.abs(y_var_test - var_pred)
                            # Combined error magnitude. Cant just combine WS and WD so left out for now
                            #error_mag = np.sqrt(dir_fix**2 + ws_error**2)
                            #error_norm = matplotlib.colors.Normalize(vmin=error_mag.min(), vmax=error_mag.max())
                
                            for i in range(len(dir_test_radian)):
                                ax.plot([dir_test_radian[i], np.deg2rad(dir_pred[i])], [y_var_test[i], var_pred[i]], color='grey', alpha=1, linewidth=0.8)
                            # Make it pretty
                            ax.set_theta_zero_location('N')
                            ax.set_theta_direction(-1)
                            ax.set_ylabel(f'{variable_name}', labelpad=55, fontsize = 12)
                            ax.set_title(f"{station}: {variable_name} and Wind Direction", pad = 60, fontsize = 25)
                            ax.set_rlabel_position(0)
                            ax.tick_params(labelsize = 12)
                            rmax = ax.get_rmax()
                            # Add cardinal direction labels
                            for label, angle in cardinals.items():
                                            ax.text(
                                                angle,
                                                rmax * 1.3,
                                                label,
                                                ha="center",
                                                va="center",
                                                fontsize=15,
                                                fontweight="bold",
                                                clip_on=False)
                            # More pretty
                            plt.legend(loc = 'upper right', bbox_to_anchor = (1.45, 1.2), fontsize = 8) 
                            # Seismic Colour Bar
                            cbar = plt.colorbar(obs, ax=ax, pad=0.1)
                            cbar.set_label(f'Log Seismic Power\n{best_component}, {freq_best[0]:.1f} Hz', size = 12)
                            # Error Colour Bar
                            # Need to think of a good way to represent error well. Cant just combine WS and WD
                            #sm = matplotlib.cm.ScalarMappable(norm=error_norm, cmap=cmap2)
                            #sm.set_array([])   # required placeholder
                            #cbar2 = plt.colorbar(sm, ax=ax, pad=0.1)
                            #cbar2.set_label("Error Magnitude", fontsize=12)
                
                                                        
                            cbar.ax.tick_params(labelsize=12)
                            #cbar2.ax.tick_params(labelsize=12)
                            plt.tight_layout()
                
                        # Plot WS Residiuals
                        if plot_residuals == True:
                
                            fig, ax = plt.subplots(1, 3, figsize=(8, 8))
                            ax[0].scatter(var_pred, y_var_test - var_pred)
                            ax[0].axhline(0, color='red', linestyle='--')
                            ax[0].set_xlabel(f"Predicted {variable_name}")
                            ax[0].set_ylabel("Residual (Observed - Predicted)")
                            ax[0].set_title(f"Residual Plot for {station}, {variable_name}")
                            ax[1].scatter(sin_pred, y_sin_test - sin_pred)
                            ax[1].axhline(0, color='red', linestyle='--')
                            ax[1].set_xlabel("Predicted Wind Direction Sin")
                            ax[1].set_ylabel("Residual (Observed - Predicted)")
                            ax[1].set_title(f"Residual Plot for {station}, Wind Direction Sin")
                            ax[2].scatter(cos_pred, y_cos_test - cos_pred)
                            ax[2].axhline(0, color='red', linestyle='--')
                            ax[2].set_xlabel("Predicted Wind Direction Cos")
                            ax[2].set_ylabel("Residual (Observed - Predicted)")
                            ax[2].set_title(f"Residual Plot for {station}, Wind Direction Cos")
                            fig.tight_layout()
                
                        # Plot Observed WS and seis power
                        if plot_power_aws == True:
                
                            # LogSeismic Power vs Obs AWS  
                
                            # Best frequency index for power
                            var_best = np.argmax(var_importance)
                
                            if var_best < n_bands:
                
                                best_component = "Z"
                                best_index = var_best
                                best_var = X_all_test[:, var_best]
                
                                
                            elif var_best < 2*n_bands:
                
                                best_component = "NS"
                                best_index = var_best - n_bands
                                best_var = X_all_test[:, var_best]
                
                            
                            else:
                
                                best_component = "EW"
                                best_index = var_best - 2*n_bands
                                best_var = X_all_test[:, var_best]
                
                                
                            # Best Centre frequencies
                            freq = band_centres[best_index]
                
                            # Sort for plot
                            idx = np.argsort(best_var)
                            x_sort = best_var[idx]
                
                            # Polynomial 
                            # Reshape x values
                            poly_x = best_var.reshape(-1,1)
                            # Create Pipeline 
                            poly = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression()).fit(poly_x, y_var_test)
                            # Calculate R² 
                            poly_r2 = poly.score(poly_x, y_var_test)
                            # Predict
                            poly_pred = poly.predict(poly_x)[idx]
                            # Coefficients
                            linear_model = poly.named_steps['linearregression']
                            coefficients = linear_model.coef_
                            intercept = linear_model.intercept_
                            # Equation
                            if poly_degree == 1:
                                equation = f"y = {coefficients[1]:.2f}x + {intercept:2f}"
                            elif poly_degree == 2:
                                equation = f"y = {coefficients[2]:.2f}x$^{2}$ + {coefficients[1]:.2f}x + {intercept:.2f}"
                            else:
                                print('Calculation only for poly_degree = 1 or 2')
                                equation = 'Equation not calculated'
                
                            # Create plots
                            plt.figure(figsize=(10,10))
                            plt.scatter(best_var, y_var_test, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                
                            # Plot
                            plt.plot(x_sort, poly_pred, 'r--', linewidth = 4, label = f'{equation} \nR² = {poly_r2:.2f}')
                            plt.title(f"{station}: {best_component}, {freq[0]:.1f} Hz", fontsize = 25)
                            # Make pretty
                            plt.legend(fontsize = 20)
                            plt.xlabel("Log Seismic Power", fontsize = 20)
                            plt.ylabel(f"Observed {variable_name}", fontsize = 20)
                            plt.xticks(fontsize = 20)
                            plt.yticks(fontsize = 20)
                            plt.tight_layout()
                
                        # Find best 10 frequency bands for each component     
                        top_ten = np.argsort(var_importance)[::-1][:10]
                
                        for index in top_ten:
                
                            if index < n_bands:
                                component = "Z"
                                freq_index = index
                
                            elif index < 2 * n_bands:
                                component = "NS"
                                freq_index = index - n_bands
                
                            else:
                                component = "EW"
                                freq_index = index - 2 * n_bands
                
                            frequency = band_centres[freq_index]
                
                            # Print them
                            print(f"{component}: {frequency[0]:.1f} Hz, importance = {var_importance[index]:.4f}")
                
                    return results


                def EN(self,
                       fmin = 1,
                       fmax = 49,
                       f_band_width = 1,
                       step_size = 1,
                       n_repeats = 3,
                       min_WS = None,
                       poly_degree = 2,
                       plot_stat_results = True,
                       plot_results = True,
                       plot_residuals = True,
                       plot_power_aws = True,
                       variable_name = 'AWS Wind Speed (km/hr)'):
                    
                    """
                    Predicts AWS variable and Wind Direction from multiple seismic frequency band power features 
                    using a ElasticNet regression model. Combines all seismic components into one model.
                    Includes a y_test array of AWS and WD. 
                    Multi Output model.
                
                    Parameters:
                        spectra (list):
                            A list of dictionaries containing the frequency, power, and Wind Speed data 
                            for each seismic component (EW, NS, Z) for each station and time period.
                        fmin (int):
                            Minimum frequency value for calculating the power of each bandwidths.
                        fmax (int):
                            Maximum frequency value for calculating the power of each  bandwidths.
                        f_band_width (int):
                            Bandwidth size. e.g. f_band_width = 1 for (f1,f2)=(1,2), 2 for (1,3), 3 for (1,4). 
                        step_size (int):
                            Frequency band step size. Set to less than f_band_width for overlapping bands. 
                        n_repeats (int):
                            Number of times to repeat the permutation importance calculation for each seismic component.
                        min_WS (int):
                            Minimum wind speed (km/hr) to be considered in analysis.
                        poly_degree (int):
                            Polynomial degree for the regression in plot_power_aws (Seis Power vs Obs WS)
                        plot_stat_results (bool):
                            Plots all the R² and rmse values against frequency bandwidth centres.
                        plot_results (bool):
                            Plots the predicted vs observed values for each seismic component.
                        plot_cv (bool):
                            Plot the cross validation results as boxplots       
                        plot_power_aws (bool):
                            Plot the relationship between seismic power and AWS values.
                        variable_name (str):
                            The name of the variable being predicted (e.g., 'AWS Wind Speed (km/hr)').
                
                    Outputs:
                        results (list):
                            A list of dictionaries containing information about all the correlation results for each station.
                    """
                    
                    # Setup Result Lists
                    results = []
                
                    # Bands
                    # Create Bandwidths
                    bands = []
                    for f1 in range(fmin, fmax - f_band_width + 1, step_size):
                        f2 = f1 + f_band_width
                        band = (f1, f2)
                        bands.append(band)
                    # Num of Bands
                    n_bands = len(bands)
                    # Create Band Centers
                    band_centres = []
                    for f1, f2 in bands:
                        band_centre = [(f1 + f2)/2]
                        band_centres.append(band_centre)
                
                    # Loop through stations
                    for station_dict in self.seismic_wind.fft:
                
                        # Setup Variables
                        station = list(station_dict.keys())[0] 
                        EW_power = station_dict[station][0]['EW']
                        NS_power = station_dict[station][0]['NS']
                        Z_power = station_dict[station][0]['Z']
                        freq = station_dict[station][0]['freq']
                        aws_values = station_dict[station][0]['aws_values']
                        wind_direction = station_dict[station][0]['wind_direction']
                
                        # Angle wrap around problem
                        dir_radians = np.deg2rad(wind_direction)
                        dir_sin = np.sin(dir_radians)
                        dir_cos = np.cos(dir_radians)
                        
                        # Setup results
                        station_results = []
                        
                        # Setup powers
                        X_Z = np.zeros((len(aws_values), len(bands)))
                        X_NS = np.zeros((len(aws_values), len(bands)))
                        X_EW = np.zeros((len(aws_values), len(bands)))
                
                        # Apply bandwidths to data
                        for i, (f1,f2) in enumerate(bands):
                            band_width = (freq >= f1) & (freq < f2)
                
                            # Convert to log to better inspect power scales and apply bandwidth
                            # [:, band_width], select frequencies and slice unwanted freq data from the row
                            # .mean(axis=1), mean for the selected frequency row. Reshape for model input.
                            X_Z[:, i] = np.log10(Z_power[:, band_width].mean(axis=1) + 1e-20)
                            X_NS[:, i] = np.log10(NS_power[:, band_width].mean(axis=1) + 1e-20)
                            X_EW[:, i] = np.log10(EW_power[:, band_width].mean(axis=1) + 1e-20)
                
                        # Stack all together to process all together as a larger dataset
                        X_all = np.column_stack([X_Z, X_NS, X_EW])
                        y_all = np.column_stack([aws_values, dir_sin, dir_cos])
                        
                        # Should the wind speed threshold be applied before or after training the model?
                        # Apply minimum WS threshold
                        if min_WS is not None:
                            mask = y_all[:, 0] > min_WS
                            X_all = X_all[mask]
                            y_all = y_all[mask]
                            
                        # Train Model
                        X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
                
                        # Pipeline
                        # Scale, ElasticNet (Need to find optimal l1 still)
                        var_model = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNet(alpha=0.08, l1_ratio=0.5))])
                        sin_model = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNet(alpha=0.08, l1_ratio=0.5))])
                        cos_model = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNet(alpha=0.08, l1_ratio=0.5))])
                
                        # Cross Validation
                        cv = KFold(n_splits=5, shuffle=True, random_state=42)
                        var_scores = cross_val_score(var_model, X_all_train, y_all_train[:, 0], cv=cv, scoring='r2')
                        sin_scores = cross_val_score(sin_model, X_all_train, y_all_train[:, 1], cv=cv, scoring='r2')
                        cos_scores = cross_val_score(cos_model, X_all_train, y_all_train[:, 2], cv=cv, scoring='r2')
                
                        # Fit Model
                        var_model.fit(X_all_train, y_all_train[:,0])
                        sin_model.fit(X_all_train, y_all_train[:, 1])
                        cos_model.fit(X_all_train, y_all_train[:, 2])
                
                        # Predictions
                        var_pred = var_model.predict(X_all_test)
                        sin_pred = sin_model.predict(X_all_test)
                        cos_pred = cos_model.predict(X_all_test)
                
                        # Round WS Prediction to one decimal so it matches with AWS Measurement
                        var_pred = np.round(var_pred, decimals=1)
                
                        # Observed
                        y_var_test = y_all_test[:, 0]
                        y_sin_test = y_all_test[:, 1]
                        y_cos_test = y_all_test[:, 2]
                
                    
                        # Predicted direction
                        dir_pred_radian = np.arctan2(sin_pred, cos_pred)
                        dir_pred = np.rad2deg(dir_pred_radian)
                
                        # Round to nearest 10 degrees
                        dir_pred = np.round(dir_pred, decimals=-1)
                        
                        # Wrap back to 0-360 degrees
                        dir_pred = dir_pred % 360 
                
                        # Fix angles near end points
                        dir_test_radian = np.arctan2(y_sin_test, y_cos_test)
                        dir_test = np.rad2deg(dir_test_radian) % 360
                        dir_fix = np.abs((dir_pred - dir_test + 180) % 360 - 180)
                
                        # Mean direction
                        dir_mean = np.mean(dir_fix)
                
                        # rmse
                        var_rmse = np.sqrt(mean_squared_error(y_var_test, var_pred))
                        sin_rmse = np.sqrt(mean_squared_error(y_sin_test, sin_pred))
                        cos_rmse = np.sqrt(mean_squared_error(y_cos_test, cos_pred))
                        dir_rmse = np.sqrt(np.mean(dir_fix**2))
                
                        # R²
                        var_r2 = r2_score(y_var_test, var_pred)
                        sin_r2 = r2_score(y_sin_test, sin_pred)
                        cos_r2 = r2_score(y_cos_test, cos_pred)
                
                        # Freq Importance
                        var_importance = permutation_importance(var_model, X_all_test, y_var_test, n_repeats=n_repeats, scoring='r2').importances_mean
                        sin_importance = permutation_importance(sin_model, X_all_test, y_sin_test, n_repeats=n_repeats, scoring='r2').importances_mean
                        cos_importance = permutation_importance(cos_model, X_all_test, y_cos_test, n_repeats=n_repeats, scoring='r2').importances_mean
                        #dir_importance = np.sqrt(sin_importance**2 + cos_importance**2)
                
                        # Coeffs? see full_elasticnet?
                
                        # Store results
                        station_results.append({
                            'fmin': fmin,
                            'fmax': fmax,
                            'var_r2': var_r2,
                            'sin_r2': sin_r2,
                            'cos_r2': cos_r2,
                            'var_rmse': var_rmse,
                            'sin_rmse': sin_rmse,
                            'cos_rmse': cos_rmse,
                            'y_var_test': y_var_test,
                            'y_sin_test': y_sin_test,
                            'y_cos_test': y_cos_test,
                            'var_pred': var_pred,
                            'sin_pred': sin_pred,
                            'cos_pred': cos_pred,
                            'var_cv_r2': var_scores.mean(),
                            'sin_cv_r2': sin_scores.mean(),
                            'cos_cv_r2': cos_scores.mean(),
                            'var_cv_std': var_scores.std(),
                            'sin_cv_std': sin_scores.std(),
                            'cos_cv_std': cos_scores.std(),
                            'dir_pred': dir_pred,
                            'dir_test': dir_test,
                            'dir_mean_error': dir_mean,
                            'dir_rmse': dir_rmse}) 
                        
                        # Print best result
                        # R²
                        print(f"{variable_name} R²: {var_r2:.4f}")
                        print(f"Sin R²: {sin_r2:.4f}")
                        print(f"Cos R²: {cos_r2:.4f}")
                
                        # rmse
                        print(f"{variable_name} rmse: {var_rmse:.4f}")
                        print(f"Sin rmse: {sin_rmse:.4f}")
                        print(f"Cos rmse: {cos_rmse:.4f}")
                
                        # Cross Validation R²
                        print(f"{variable_name} cv R²: {var_scores.mean():.4f} +/- {var_scores.std():.4f}")
                        print(f"Sin cv R²: {sin_scores.mean():.4f} +/- {sin_scores.std():.4f}")
                        print(f"Cos cv R²: {cos_scores.mean():.4f} +/- {cos_scores.std():.4f}")
                        results.append(station_results)
                
                        # WD
                        print(f"Wind direction Mean Error: {dir_mean:.2f}°")
                        print(f"Wind direction RMSE: {dir_rmse:.2f}°")
                
                        # Plot all average R² and rmse results against centre frequency
                        if plot_stat_results == True:
                            
                            # AWS Variable
                            Z_var_importance = var_importance[:n_bands]
                            NS_var_importance = var_importance[n_bands:2*n_bands]
                            EW_var_importance = var_importance[2*n_bands:3*n_bands]
                            # AWS Direction
                            #Z_dir_importance = dir_importance[:n_bands]
                            #NS_dir_importance = dir_importance[n_bands:2*n_bands]
                            #EW_dir_importance = dir_importance[2*n_bands:3*n_bands]
                
                            # Plot
                            fig, ax = plt.subplots(1, 3, figsize=(8, 6))
                            ax[0].plot(band_centres, Z_var_importance)
                            ax[0].set_title(f'Z {variable_name} \n Permutation Importance \n Acrosss Freq Spectrum')
                            ax[0].set_ylim(top=1)
                            ax[1].plot(band_centres, NS_var_importance)
                            ax[1].set_title(f'NS {variable_name} \n Permutation Importance \n Acrosss Freq Spectrum')
                            ax[1].set_ylim(top=1)
                            ax[2].plot(band_centres, EW_var_importance)
                            ax[2].set_title(f'EW {variable_name} \n Permutation Importance \n Acrosss Freq Spectrum')
                            ax[2].set_ylim(top=1)
                            #ax[1, 0].plot(band_centres, Z_dir_importance)
                            #ax[1, 0].set_title(f'Combined Z Wind Direction \n Permutation Importance \n Acrosss Freq Spectrum')
                            #ax[1, 0].set_ylim(top=1)
                            #ax[1, 1].plot(band_centres, NS_dir_importance)
                            #ax[1, 1].set_title(f'Combined NS Wind Direction \n Permutation Importance \n Acrosss Freq Spectrum')
                            #ax[1, 1].set_ylim(top=1)
                            #ax[1, 2].plot(band_centres, EW_dir_importance)
                            #ax[1, 2].set_title(f'Combined EW Wind Direction \n Permutation Importance \n Acrosss Freq Spectrum')
                            #ax[1, 2].set_ylim(top=1)
                
                            fig.suptitle(f'{station}:')
                            fig.supxlabel('Frequency (Hz)')
                            fig.supylabel('RF Permutation Importance')
                            fig.tight_layout()
                
                        # Plot key results
                        if plot_results == True:
                
                            # Plot obs vs pred WS
                            # AWS Variable
                            # Plot
                            plt.figure(figsize=(10, 10))
                            plt.scatter(y_var_test, var_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                
                            # y = x line
                            plt.plot([y_var_test.min(), y_var_test.max()],
                                    [y_var_test.min(), y_var_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            # Make it pretty
                            plt.xlabel(f"Observed {variable_name}", fontsize = 20)
                            plt.ylabel(f"Predicted {variable_name}", fontsize = 20)
                            plt.title(f"{station} {variable_name}", fontsize = 25)
                            plt.xticks(fontsize = 20)
                            plt.yticks(fontsize = 20)
                            plt.legend(fontsize = 20, loc ='upper left')
                            plt.tight_layout()
                
                            # Plot obs vs pred Wind Direction
                            # sin2theta and cos2theta
                            fig, ax = plt.subplots(2,1, figsize=(10, 10))
                            ax[0].scatter(y_sin_test, sin_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                            ax[1].scatter(y_cos_test, cos_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                            
                            # y = x line
                            ax[0].plot([y_sin_test.min(), y_sin_test.max()],
                                    [y_sin_test.min(), y_sin_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            ax[1].plot([y_cos_test.min(), y_cos_test.max()],
                                    [y_cos_test.min(), y_cos_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            # Make it pretty
                            ax[0].set_xlabel("Observed Wind Direction Sin2Theta", fontsize = 20)
                            ax[0].set_ylabel("Predicted Wind Direction Sin2Theta", fontsize = 20)
                            ax[0].set_title(f"{station} Wind Direction Sin2Theta", fontsize = 25)
                            ax[1].set_xlabel("Observed Wind Direction Cos2Theta", fontsize = 20)
                            ax[1].set_ylabel("Predicted Wind Direction Cos2Theta", fontsize = 20)
                            ax[1].set_title(f"{station} Wind Direction Cos2Theta", fontsize = 25)
                            ax[0].tick_params(axis='both', which='major', labelsize=20)
                            ax[1].tick_params(axis='both', which='major', labelsize=20)
                            fig.tight_layout()
                
                            # Sin2 and Cos2 Converted back to angle
                            plt.figure(figsize=(10, 10))
                            plt.scatter(dir_test, dir_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                            
                            # y = x line
                            plt.plot([dir_test.min(), dir_test.max()],
                                    [dir_test.min(), dir_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            # Make it pretty
                            plt.xlabel("Observed Wind Direction (°)", fontsize = 20)
                            plt.ylabel("Predicted Wind Direction (°)", fontsize = 20)
                            plt.title(f"{station} Wind Direction (°)", fontsize = 25)
                            plt.xticks(fontsize = 20)
                            plt.yticks(fontsize = 20)
                            plt.legend(fontsize = 20, loc ='upper left')
                            plt.tight_layout()
                
                            # Plot Wind Direction
                            # Wind direction
                            cardinals = {"N": 0,           
                                        "E": (np.pi / 2),
                                        "S": (np.pi),
                                        "W": (3 * np.pi / 2)}
                            # Plot
                            plt.figure(figsize=(10, 6))
                            plt.polar()      
                            plt.scatter(dir_test_radian, np.ones(len(dir_test_radian)) * 0.8, alpha=0.7, label = 'Observed Wind Direction')
                            plt.scatter(np.deg2rad(dir_pred), np.ones(len(dir_test_radian)) * 0.85, alpha=0.7, label = 'Predicted Wind Direction')
                
                            # Make it Pretty
                            plt.gca().set_theta_zero_location('N')
                            plt.gca().set_theta_direction(-1)
                            plt.title(f"{station} Wind Direction", pad = 50)
                            plt.gca().set_rlabel_position(0)
                            plt.ylim(0,1)
                            # Add cardinal direction labels
                            for label, angle in cardinals.items():
                                            plt.text(
                                                angle,
                                                1.2,
                                                label,
                                                ha="center",
                                                va="center",
                                                fontsize=12,
                                                fontweight="bold",
                                                clip_on=False)
                            # Legend
                            plt.legend(loc = 'upper right', bbox_to_anchor = (1.55, 1.18))
                            # Design
                            plt.tight_layout()
                
                            # Plot WS, WD, Seis Power
                            # AWS Variable and Wind direction
                
                            # Best frequency index for seismic power
                            var_best = np.argmax(var_importance)
                
                            if var_best < n_bands:
                
                                best_component = "Z"
                                best_index = var_best
                                best_var = X_all_test[:, var_best]
                
                            elif var_best < 2*n_bands:
                
                                best_component = "NS"
                                best_index = var_best - n_bands
                                best_var = X_all_test[:, var_best]
                
                            else:
                
                                best_component = "EW"
                                best_index = var_best - 2*n_bands
                                best_var = X_all_test[:, var_best]
                
                            # Best Centre frequencies
                            freq_best = band_centres[best_index]
                
                            # Create plot
                            plt.figure(figsize=(10,6))
                            ax = plt.subplot(projection='polar')
                            plt.polar()
                            # Colour bar
                            power_colours = best_var
                            cmap = matplotlib.cm.get_cmap('plasma')
                            new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('snipped_cmap', cmap(np.linspace(0, 0.90, 256)))
                            #cmap2 = matplotlib.cm.get_cmap('cool') # For error magnitude
                            # Plot measurements
                            obs = ax.scatter(dir_test_radian, y_var_test, alpha=0.7, c=power_colours, cmap=new_cmap, s = 60, label = f'Observed {variable_name}', marker='x')
                            pred = ax.scatter(np.deg2rad(dir_pred), var_pred, alpha=0.7, c=power_colours, cmap=new_cmap, s = 60, label = f'Predicted {variable_name}', marker='^')
                            # Plot error mag line between pred and obs point
                
                            # WS error
                            # Not useful yet, but could be used to colour the line between obs and pred points
                            #ws_error = np.abs(y_var_test - var_pred)
                            # Combined error magnitude. Cant just combine WS and WD so left out for now
                            #error_mag = np.sqrt(dir_fix**2 + ws_error**2)
                            #error_norm = matplotlib.colors.Normalize(vmin=error_mag.min(), vmax=error_mag.max())
                
                            for i in range(len(dir_test_radian)):
                                ax.plot([dir_test_radian[i], np.deg2rad(dir_pred[i])], [y_var_test[i], var_pred[i]], color='grey', alpha=1, linewidth=0.8)
                            # Make it pretty
                            ax.set_theta_zero_location('N')
                            ax.set_theta_direction(-1)
                            ax.set_ylabel(f'{variable_name}', labelpad=55, fontsize = 12)
                            ax.set_title(f"{station}: {variable_name} and Wind Direction", pad = 60, fontsize = 25)
                            ax.set_rlabel_position(0)
                            ax.tick_params(labelsize = 12)
                            rmax = ax.get_rmax()
                            # Add cardinal direction labels
                            for label, angle in cardinals.items():
                                            ax.text(
                                                angle,
                                                rmax * 1.3,
                                                label,
                                                ha="center",
                                                va="center",
                                                fontsize=15,
                                                fontweight="bold",
                                                clip_on=False)
                            # More pretty
                            plt.legend(loc = 'upper right', bbox_to_anchor = (1.45, 1.2), fontsize = 8) 
                            # Seismic Colour Bar
                            cbar = plt.colorbar(obs, ax=ax, pad=0.1)
                            cbar.set_label(f'Log Seismic Power\n{best_component}, {freq_best[0]:.1f} Hz', size = 12)
                            # Error Colour Bar
                            # Need to think of a good way to represent error well. Cant just combine WS and WD
                            #sm = matplotlib.cm.ScalarMappable(norm=error_norm, cmap=cmap2)
                            #sm.set_array([])   # required placeholder
                            #cbar2 = plt.colorbar(sm, ax=ax, pad=0.1)
                            #cbar2.set_label("Error Magnitude", fontsize=12)
                
                                                        
                            cbar.ax.tick_params(labelsize=12)
                            #cbar2.ax.tick_params(labelsize=12)
                            plt.tight_layout()
                
                        # Plot WS Residiuals
                        if plot_residuals == True:
                
                            fig, ax = plt.subplots(1, 3, figsize=(8, 8))
                            ax[0].scatter(var_pred, y_var_test - var_pred)
                            ax[0].axhline(0, color='red', linestyle='--')
                            ax[0].set_xlabel(f"Predicted {variable_name}")
                            ax[0].set_ylabel("Residual (Observed - Predicted)")
                            ax[0].set_title(f"Residual Plot for {station}, {variable_name}")
                            ax[1].scatter(sin_pred, y_sin_test - sin_pred)
                            ax[1].axhline(0, color='red', linestyle='--')
                            ax[1].set_xlabel("Predicted Wind Direction Sin")
                            ax[1].set_ylabel("Residual (Observed - Predicted)")
                            ax[1].set_title(f"Residual Plot for {station}, Wind Direction Sin")
                            ax[2].scatter(cos_pred, y_cos_test - cos_pred)
                            ax[2].axhline(0, color='red', linestyle='--')
                            ax[2].set_xlabel("Predicted Wind Direction Cos")
                            ax[2].set_ylabel("Residual (Observed - Predicted)")
                            ax[2].set_title(f"Residual Plot for {station}, Wind Direction Cos")
                            fig.tight_layout()
                
                        # Plot Observed WS and seis power
                        if plot_power_aws == True:
                
                            # LogSeismic Power vs Obs AWS  
                
                            # Best frequency index for power
                            var_best = np.argmax(var_importance)
                
                            if var_best < n_bands:
                
                                best_component = "Z"
                                best_index = var_best
                                best_var = X_all_test[:, var_best]
                
                                
                            elif var_best < 2*n_bands:
                
                                best_component = "NS"
                                best_index = var_best - n_bands
                                best_var = X_all_test[:, var_best]
                
                            
                            else:
                
                                best_component = "EW"
                                best_index = var_best - 2*n_bands
                                best_var = X_all_test[:, var_best]
                
                                
                            # Best Centre frequencies
                            freq = band_centres[best_index]
                
                            # Sort for plot
                            idx = np.argsort(best_var)
                            x_sort = best_var[idx]
                
                            # Polynomial 
                            # Reshape x values
                            poly_x = best_var.reshape(-1,1)
                            # Create Pipeline 
                            poly = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression()).fit(poly_x, y_var_test)
                            # Calculate R² 
                            poly_r2 = poly.score(poly_x, y_var_test)
                            # Predict
                            poly_pred = poly.predict(poly_x)[idx]
                            # Coefficients
                            linear_model = poly.named_steps['linearregression']
                            coefficients = linear_model.coef_
                            intercept = linear_model.intercept_
                            # Equation
                            if poly_degree == 1:
                                equation = f"y = {coefficients[1]:.2f}x + {intercept:2f}"
                            elif poly_degree == 2:
                                equation = f"y = {coefficients[2]:.2f}x$^{2}$ + {coefficients[1]:.2f}x + {intercept:.2f}"
                            else:
                                print('Calculation only for poly_degree = 1 or 2')
                                equation = 'Equation not calculated'
                
                            # Create plots
                            plt.figure(figsize=(10,10))
                            plt.scatter(best_var, y_var_test, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                
                            # Plot
                            plt.plot(x_sort, poly_pred, 'r--', linewidth = 4, label = f'{equation} \nR² = {poly_r2:.2f}')
                            plt.title(f"{station}: {best_component}, {freq[0]:.1f} Hz", fontsize = 25)
                            # Make pretty
                            plt.legend(fontsize = 20)
                            plt.xlabel("Log Seismic Power", fontsize = 20)
                            plt.ylabel(f"Observed {variable_name}", fontsize = 20)
                            plt.xticks(fontsize = 20)
                            plt.yticks(fontsize = 20)
                            plt.tight_layout()
                
                        # Find best 10 frequency bands for each component     
                        top_ten = np.argsort(var_importance)[::-1][:10]
                
                        for index in top_ten:
                
                            if index < n_bands:
                                component = "Z"
                                freq_index = index
                
                            elif index < 2 * n_bands:
                                component = "NS"
                                freq_index = index - n_bands
                
                            else:
                                component = "EW"
                                freq_index = index - 2 * n_bands
                
                            frequency = band_centres[freq_index]
                
                            # Print them
                            print(f"{component}: {frequency[0]:.1f} Hz, importance = {var_importance[index]:.4f}")
                
                    return results

                
                def Ridge(self,
                          fmin = 1,
                          fmax = 49,
                          f_band_width = 1,
                          step_size = 1,
                          n_repeats = 3,
                          min_WS = None,
                          poly_degree = 2,
                          plot_stat_results = True,
                          plot_results = True,
                          plot_residuals = True,
                          plot_power_aws = True,
                          variable_name = 'AWS Wind Speed (km/hr)'):

                    """
                    Predicts AWS variable and Wind Direction from multiple seismic frequency band power features 
                    using a Ridge regression model. Combines all seismic components into one model.
                    Includes a y_test array of AWS and WD. 
                    Multi Output model.
                
                    Parameters:
                        spectra (list):
                            A list of dictionaries containing the frequency, power, and Wind Speed data 
                            for each seismic component (EW, NS, Z) for each station and time period.
                        fmin (int):
                            Minimum frequency value for calculating the power of each bandwidths.
                        fmax (int):
                            Maximum frequency value for calculating the power of each  bandwidths.
                        f_band_width (int):
                            Bandwidth size. e.g. f_band_width = 1 for (f1,f2)=(1,2), 2 for (1,3), 3 for (1,4). 
                        step_size (int):
                            Frequency band step size. Set to less than f_band_width for overlapping bands. 
                        n_repeats (int):
                            Number of times to repeat the permutation importance calculation for each seismic component.
                        min_WS (int):
                            Minimum wind speed (km/hr) to be considered in analysis.
                        poly_degree (int):
                            Polynomial degree for the regression in plot_power_aws (Seis Power vs Obs WS)
                        plot_stat_results (bool):
                            Plots all the R² and rmse values against frequency bandwidth centres.
                        plot_results (bool):
                            Plots the predicted vs observed values for each seismic component.
                        plot_cv (bool):
                            Plot the cross validation results as boxplots       
                        plot_power_aws (bool):
                            Plot the relationship between seismic power and AWS values.
                        variable_name (str):
                            The name of the variable being predicted (e.g., 'AWS Wind Speed (km/hr)').
                
                    Outputs:
                        results (list):
                            A list of dictionaries containing information about all the correlation results for each station.
                    """
                    
                    # Setup Result Lists
                    results = []
                
                    # Bands
                    # Create Bandwidths
                    bands = []
                    for f1 in range(fmin, fmax - f_band_width + 1, step_size):
                        f2 = f1 + f_band_width
                        band = (f1, f2)
                        bands.append(band)
                    # Num of Bands
                    n_bands = len(bands)
                    # Create Band Centers
                    band_centres = []
                    for f1, f2 in bands:
                        band_centre = [(f1 + f2)/2]
                        band_centres.append(band_centre)
                
                    # Loop through stations
                    for station_dict in self.seismic_wind.fft:
                
                        # Setup Variables
                        station = list(station_dict.keys())[0] 
                        EW_power = station_dict[station][0]['EW']
                        NS_power = station_dict[station][0]['NS']
                        Z_power = station_dict[station][0]['Z']
                        freq = station_dict[station][0]['freq']
                        aws_values = station_dict[station][0]['aws_values']
                        wind_direction = station_dict[station][0]['wind_direction']
                
                        # Angle wrap around problem
                        dir_radians = np.deg2rad(wind_direction)
                        dir_sin = np.sin(dir_radians)
                        dir_cos = np.cos(dir_radians)
                        
                        # Setup results
                        station_results = []
                        
                        # Setup powers
                        X_Z = np.zeros((len(aws_values), len(bands)))
                        X_NS = np.zeros((len(aws_values), len(bands)))
                        X_EW = np.zeros((len(aws_values), len(bands)))
                
                        # Apply bandwidths to data
                        for i, (f1,f2) in enumerate(bands):
                            band_width = (freq >= f1) & (freq < f2)
                
                            # Convert to log to better inspect power scales and apply bandwidth
                            # [:, band_width], select frequencies and slice unwanted freq data from the row
                            # .mean(axis=1), mean for the selected frequency row. Reshape for model input.
                            X_Z[:, i] = np.log10(Z_power[:, band_width].mean(axis=1) + 1e-20)
                            X_NS[:, i] = np.log10(NS_power[:, band_width].mean(axis=1) + 1e-20)
                            X_EW[:, i] = np.log10(EW_power[:, band_width].mean(axis=1) + 1e-20)
                
                        # Stack all together to process all together as a larger dataset
                        X_all = np.column_stack([X_Z, X_NS, X_EW])
                        y_all = np.column_stack([aws_values, dir_sin, dir_cos])
                        
                        # Should the wind speed threshold be applied before or after training the model?
                        # Apply minimum WS threshold
                        if min_WS is not None:
                            mask = y_all[:, 0] > min_WS
                            X_all = X_all[mask]
                            y_all = y_all[mask]
                            
                        # Train Model
                        X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
                
                        # Pipeline
                        # Scale, Ridge # need to optimise alpha
                        var_model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=0.08))])
                        sin_model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=0.08))])
                        cos_model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=0.08))])
                
                        # Cross Validation
                        cv = KFold(n_splits=5, shuffle=True, random_state=42)
                        var_scores = cross_val_score(var_model, X_all_train, y_all_train[:, 0], cv=cv, scoring='r2')
                        sin_scores = cross_val_score(sin_model, X_all_train, y_all_train[:, 1], cv=cv, scoring='r2')
                        cos_scores = cross_val_score(cos_model, X_all_train, y_all_train[:, 2], cv=cv, scoring='r2')
                
                        # Fit Model
                        var_model.fit(X_all_train, y_all_train[:,0])
                        sin_model.fit(X_all_train, y_all_train[:, 1])
                        cos_model.fit(X_all_train, y_all_train[:, 2])
                
                        # Predictions
                        var_pred = var_model.predict(X_all_test)
                        sin_pred = sin_model.predict(X_all_test)
                        cos_pred = cos_model.predict(X_all_test)
                
                        # Round WS Prediction to one decimal so it matches with AWS Measurement
                        var_pred = np.round(var_pred, decimals=1)
                
                        # Observed
                        y_var_test = y_all_test[:, 0]
                        y_sin_test = y_all_test[:, 1]
                        y_cos_test = y_all_test[:, 2]
                
                    
                        # Predicted direction
                        dir_pred_radian = np.arctan2(sin_pred, cos_pred)
                        dir_pred = np.rad2deg(dir_pred_radian)
                
                        # Round to nearest 10 degrees
                        dir_pred = np.round(dir_pred, decimals=-1)
                        
                        # Wrap back to 0-360 degrees
                        dir_pred = dir_pred % 360 
                
                        # Fix angles near end points
                        dir_test_radian = np.arctan2(y_sin_test, y_cos_test)
                        dir_test = np.rad2deg(dir_test_radian) % 360
                        dir_fix = np.abs((dir_pred - dir_test + 180) % 360 - 180)
                
                        # Mean direction
                        dir_mean = np.mean(dir_fix)
                
                        # rmse
                        var_rmse = np.sqrt(mean_squared_error(y_var_test, var_pred))
                        sin_rmse = np.sqrt(mean_squared_error(y_sin_test, sin_pred))
                        cos_rmse = np.sqrt(mean_squared_error(y_cos_test, cos_pred))
                        dir_rmse = np.sqrt(np.mean(dir_fix**2))
                
                        # R²
                        var_r2 = r2_score(y_var_test, var_pred)
                        sin_r2 = r2_score(y_sin_test, sin_pred)
                        cos_r2 = r2_score(y_cos_test, cos_pred)
                
                        # Freq Importance
                        var_importance = permutation_importance(var_model, X_all_test, y_var_test, n_repeats=n_repeats, scoring='r2').importances_mean
                        sin_importance = permutation_importance(sin_model, X_all_test, y_sin_test, n_repeats=n_repeats, scoring='r2').importances_mean
                        cos_importance = permutation_importance(cos_model, X_all_test, y_cos_test, n_repeats=n_repeats, scoring='r2').importances_mean
                        #dir_importance = np.sqrt(sin_importance**2 + cos_importance**2)
                
                        # Coeffs? see full_ridge?
                
                        # Store results
                        station_results.append({
                            'fmin': fmin,
                            'fmax': fmax,
                            'var_r2': var_r2,
                            'sin_r2': sin_r2,
                            'cos_r2': cos_r2,
                            'var_rmse': var_rmse,
                            'sin_rmse': sin_rmse,
                            'cos_rmse': cos_rmse,
                            'y_var_test': y_var_test,
                            'y_sin_test': y_sin_test,
                            'y_cos_test': y_cos_test,
                            'var_pred': var_pred,
                            'sin_pred': sin_pred,
                            'cos_pred': cos_pred,
                            'var_cv_r2': var_scores.mean(),
                            'sin_cv_r2': sin_scores.mean(),
                            'cos_cv_r2': cos_scores.mean(),
                            'var_cv_std': var_scores.std(),
                            'sin_cv_std': sin_scores.std(),
                            'cos_cv_std': cos_scores.std(),
                            'dir_pred': dir_pred,
                            'dir_test': dir_test,
                            'dir_mean_error': dir_mean,
                            'dir_rmse': dir_rmse}) 
                        
                        # Print best result
                        # R²
                        print(f"{variable_name} R²: {var_r2:.4f}")
                        print(f"Sin R²: {sin_r2:.4f}")
                        print(f"Cos R²: {cos_r2:.4f}")
                
                        # rmse
                        print(f"{variable_name} rmse: {var_rmse:.4f}")
                        print(f"Sin rmse: {sin_rmse:.4f}")
                        print(f"Cos rmse: {cos_rmse:.4f}")
                
                        # Cross Validation R²
                        print(f"{variable_name} cv R²: {var_scores.mean():.4f} +/- {var_scores.std():.4f}")
                        print(f"Sin cv R²: {sin_scores.mean():.4f} +/- {sin_scores.std():.4f}")
                        print(f"Cos cv R²: {cos_scores.mean():.4f} +/- {cos_scores.std():.4f}")
                        results.append(station_results)
                
                        # WD
                        print(f"Wind direction Mean Error: {dir_mean:.2f}°")
                        print(f"Wind direction RMSE: {dir_rmse:.2f}°")
                
                        # Plot all average R² and rmse results against centre frequency
                        if plot_stat_results == True:
                            
                            # AWS Variable
                            Z_var_importance = var_importance[:n_bands]
                            NS_var_importance = var_importance[n_bands:2*n_bands]
                            EW_var_importance = var_importance[2*n_bands:3*n_bands]
                            # AWS Direction
                            #Z_dir_importance = dir_importance[:n_bands]
                            #NS_dir_importance = dir_importance[n_bands:2*n_bands]
                            #EW_dir_importance = dir_importance[2*n_bands:3*n_bands]
                
                            # Plot
                            fig, ax = plt.subplots(1, 3, figsize=(8, 6))
                            ax[0].plot(band_centres, Z_var_importance)
                            ax[0].set_title(f'Z {variable_name} \n Permutation Importance \n Acrosss Freq Spectrum')
                            ax[0].set_ylim(top=1)
                            ax[1].plot(band_centres, NS_var_importance)
                            ax[1].set_title(f'NS {variable_name} \n Permutation Importance \n Acrosss Freq Spectrum')
                            ax[1].set_ylim(top=1)
                            ax[2].plot(band_centres, EW_var_importance)
                            ax[2].set_title(f'EW {variable_name} \n Permutation Importance \n Acrosss Freq Spectrum')
                            ax[2].set_ylim(top=1)
                            #ax[1, 0].plot(band_centres, Z_dir_importance)
                            #ax[1, 0].set_title(f'Combined Z Wind Direction \n Permutation Importance \n Acrosss Freq Spectrum')
                            #ax[1, 0].set_ylim(top=1)
                            #ax[1, 1].plot(band_centres, NS_dir_importance)
                            #ax[1, 1].set_title(f'Combined NS Wind Direction \n Permutation Importance \n Acrosss Freq Spectrum')
                            #ax[1, 1].set_ylim(top=1)
                            #ax[1, 2].plot(band_centres, EW_dir_importance)
                            #ax[1, 2].set_title(f'Combined EW Wind Direction \n Permutation Importance \n Acrosss Freq Spectrum')
                            #ax[1, 2].set_ylim(top=1)
                
                            fig.suptitle(f'{station}:')
                            fig.supxlabel('Frequency (Hz)')
                            fig.supylabel('RF Permutation Importance')
                            fig.tight_layout()
                
                        # Plot key results
                        if plot_results == True:
                
                            # Plot obs vs pred WS
                            # AWS Variable
                            # Plot
                            plt.figure(figsize=(10, 10))
                            plt.scatter(y_var_test, var_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                
                            # y = x line
                            plt.plot([y_var_test.min(), y_var_test.max()],
                                    [y_var_test.min(), y_var_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            # Make it pretty
                            plt.xlabel(f"Observed {variable_name}", fontsize = 20)
                            plt.ylabel(f"Predicted {variable_name}", fontsize = 20)
                            plt.title(f"{station} {variable_name}", fontsize = 25)
                            plt.xticks(fontsize = 20)
                            plt.yticks(fontsize = 20)
                            plt.legend(fontsize = 20, loc ='upper left')
                            plt.tight_layout()
                
                            # Plot obs vs pred Wind Direction
                            # sin2theta and cos2theta
                            fig, ax = plt.subplots(2,1, figsize=(10, 10))
                            ax[0].scatter(y_sin_test, sin_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                            ax[1].scatter(y_cos_test, cos_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                            
                            # y = x line
                            ax[0].plot([y_sin_test.min(), y_sin_test.max()],
                                    [y_sin_test.min(), y_sin_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            ax[1].plot([y_cos_test.min(), y_cos_test.max()],
                                    [y_cos_test.min(), y_cos_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            # Make it pretty
                            ax[0].set_xlabel("Observed Wind Direction Sin2Theta", fontsize = 20)
                            ax[0].set_ylabel("Predicted Wind Direction Sin2Theta", fontsize = 20)
                            ax[0].set_title(f"{station} Wind Direction Sin2Theta", fontsize = 25)
                            ax[1].set_xlabel("Observed Wind Direction Cos2Theta", fontsize = 20)
                            ax[1].set_ylabel("Predicted Wind Direction Cos2Theta", fontsize = 20)
                            ax[1].set_title(f"{station} Wind Direction Cos2Theta", fontsize = 25)
                            ax[0].tick_params(axis='both', which='major', labelsize=20)
                            ax[1].tick_params(axis='both', which='major', labelsize=20)
                            fig.tight_layout()
                
                            # Sin2 and Cos2 Converted back to angle
                            plt.figure(figsize=(10, 10))
                            plt.scatter(dir_test, dir_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                            
                            # y = x line
                            plt.plot([dir_test.min(), dir_test.max()],
                                    [dir_test.min(), dir_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            # Make it pretty
                            plt.xlabel("Observed Wind Direction (°)", fontsize = 20)
                            plt.ylabel("Predicted Wind Direction (°)", fontsize = 20)
                            plt.title(f"{station} Wind Direction (°)", fontsize = 25)
                            plt.xticks(fontsize = 20)
                            plt.yticks(fontsize = 20)
                            plt.legend(fontsize = 20, loc ='upper left')
                            plt.tight_layout()
                
                            # Plot Wind Direction
                            # Wind direction
                            cardinals = {"N": 0,           
                                        "E": (np.pi / 2),
                                        "S": (np.pi),
                                        "W": (3 * np.pi / 2)}
                            # Plot
                            plt.figure(figsize=(10, 6))
                            plt.polar()      
                            plt.scatter(dir_test_radian, np.ones(len(dir_test_radian)) * 0.8, alpha=0.7, label = 'Observed Wind Direction')
                            plt.scatter(np.deg2rad(dir_pred), np.ones(len(dir_test_radian)) * 0.85, alpha=0.7, label = 'Predicted Wind Direction')
                
                            # Make it Pretty
                            plt.gca().set_theta_zero_location('N')
                            plt.gca().set_theta_direction(-1)
                            plt.title(f"{station} Wind Direction", pad = 50)
                            plt.gca().set_rlabel_position(0)
                            plt.ylim(0,1)
                            # Add cardinal direction labels
                            for label, angle in cardinals.items():
                                            plt.text(
                                                angle,
                                                1.2,
                                                label,
                                                ha="center",
                                                va="center",
                                                fontsize=12,
                                                fontweight="bold",
                                                clip_on=False)
                            # Legend
                            plt.legend(loc = 'upper right', bbox_to_anchor = (1.55, 1.18))
                            # Design
                            plt.tight_layout()
                
                            # Plot WS, WD, Seis Power
                            # AWS Variable and Wind direction
                
                            # Best frequency index for seismic power
                            var_best = np.argmax(var_importance)
                
                            if var_best < n_bands:
                
                                best_component = "Z"
                                best_index = var_best
                                best_var = X_all_test[:, var_best]
                
                            elif var_best < 2*n_bands:
                
                                best_component = "NS"
                                best_index = var_best - n_bands
                                best_var = X_all_test[:, var_best]
                
                            else:
                
                                best_component = "EW"
                                best_index = var_best - 2*n_bands
                                best_var = X_all_test[:, var_best]
                
                            # Best Centre frequencies
                            freq_best = band_centres[best_index]
                
                            # Create plot
                            plt.figure(figsize=(10,6))
                            ax = plt.subplot(projection='polar')
                            plt.polar()
                            # Colour bar
                            power_colours = best_var
                            cmap = matplotlib.cm.get_cmap('plasma')
                            new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('snipped_cmap', cmap(np.linspace(0, 0.90, 256)))
                            #cmap2 = matplotlib.cm.get_cmap('cool') # For error magnitude
                            # Plot measurements
                            obs = ax.scatter(dir_test_radian, y_var_test, alpha=0.7, c=power_colours, cmap=new_cmap, s = 60, label = f'Observed {variable_name}', marker='x')
                            pred = ax.scatter(np.deg2rad(dir_pred), var_pred, alpha=0.7, c=power_colours, cmap=new_cmap, s = 60, label = f'Predicted {variable_name}', marker='^')
                            # Plot error mag line between pred and obs point
                
                            # WS error
                            # Not useful yet, but could be used to colour the line between obs and pred points
                            #ws_error = np.abs(y_var_test - var_pred)
                            # Combined error magnitude. Cant just combine WS and WD so left out for now
                            #error_mag = np.sqrt(dir_fix**2 + ws_error**2)
                            #error_norm = matplotlib.colors.Normalize(vmin=error_mag.min(), vmax=error_mag.max())
                
                            for i in range(len(dir_test_radian)):
                                ax.plot([dir_test_radian[i], np.deg2rad(dir_pred[i])], [y_var_test[i], var_pred[i]], color='grey', alpha=1, linewidth=0.8)
                            # Make it pretty
                            ax.set_theta_zero_location('N')
                            ax.set_theta_direction(-1)
                            ax.set_ylabel(f'{variable_name}', labelpad=55, fontsize = 12)
                            ax.set_title(f"{station}: {variable_name} and Wind Direction", pad = 60, fontsize = 25)
                            ax.set_rlabel_position(0)
                            ax.tick_params(labelsize = 12)
                            rmax = ax.get_rmax()
                            # Add cardinal direction labels
                            for label, angle in cardinals.items():
                                            ax.text(
                                                angle,
                                                rmax * 1.3,
                                                label,
                                                ha="center",
                                                va="center",
                                                fontsize=15,
                                                fontweight="bold",
                                                clip_on=False)
                            # More pretty
                            plt.legend(loc = 'upper right', bbox_to_anchor = (1.45, 1.2), fontsize = 8) 
                            # Seismic Colour Bar
                            cbar = plt.colorbar(obs, ax=ax, pad=0.1)
                            cbar.set_label(f'Log Seismic Power\n{best_component}, {freq_best[0]:.1f} Hz', size = 12)
                            # Error Colour Bar
                            # Need to think of a good way to represent error well. Cant just combine WS and WD
                            #sm = matplotlib.cm.ScalarMappable(norm=error_norm, cmap=cmap2)
                            #sm.set_array([])   # required placeholder
                            #cbar2 = plt.colorbar(sm, ax=ax, pad=0.1)
                            #cbar2.set_label("Error Magnitude", fontsize=12)
                
                                                        
                            cbar.ax.tick_params(labelsize=12)
                            #cbar2.ax.tick_params(labelsize=12)
                            plt.tight_layout()
                
                        # Plot WS Residiuals
                        if plot_residuals == True:
                
                            fig, ax = plt.subplots(1, 3, figsize=(8, 8))
                            ax[0].scatter(var_pred, y_var_test - var_pred)
                            ax[0].axhline(0, color='red', linestyle='--')
                            ax[0].set_xlabel(f"Predicted {variable_name}")
                            ax[0].set_ylabel("Residual (Observed - Predicted)")
                            ax[0].set_title(f"Residual Plot for {station}, {variable_name}")
                            ax[1].scatter(sin_pred, y_sin_test - sin_pred)
                            ax[1].axhline(0, color='red', linestyle='--')
                            ax[1].set_xlabel("Predicted Wind Direction Sin")
                            ax[1].set_ylabel("Residual (Observed - Predicted)")
                            ax[1].set_title(f"Residual Plot for {station}, Wind Direction Sin")
                            ax[2].scatter(cos_pred, y_cos_test - cos_pred)
                            ax[2].axhline(0, color='red', linestyle='--')
                            ax[2].set_xlabel("Predicted Wind Direction Cos")
                            ax[2].set_ylabel("Residual (Observed - Predicted)")
                            ax[2].set_title(f"Residual Plot for {station}, Wind Direction Cos")
                            fig.tight_layout()
                
                        # Plot Observed WS and seis power
                        if plot_power_aws == True:
                
                            # LogSeismic Power vs Obs AWS  
                
                            # Best frequency index for power
                            var_best = np.argmax(var_importance)
                
                            if var_best < n_bands:
                
                                best_component = "Z"
                                best_index = var_best
                                best_var = X_all_test[:, var_best]
                
                                
                            elif var_best < 2*n_bands:
                
                                best_component = "NS"
                                best_index = var_best - n_bands
                                best_var = X_all_test[:, var_best]
                
                            
                            else:
                
                                best_component = "EW"
                                best_index = var_best - 2*n_bands
                                best_var = X_all_test[:, var_best]
                
                                
                            # Best Centre frequencies
                            freq = band_centres[best_index]
                
                            # Sort for plot
                            idx = np.argsort(best_var)
                            x_sort = best_var[idx]
                
                            # Polynomial 
                            # Reshape x values
                            poly_x = best_var.reshape(-1,1)
                            # Create Pipeline 
                            poly = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression()).fit(poly_x, y_var_test)
                            # Calculate R² 
                            poly_r2 = poly.score(poly_x, y_var_test)
                            # Predict
                            poly_pred = poly.predict(poly_x)[idx]
                            # Coefficients
                            linear_model = poly.named_steps['linearregression']
                            coefficients = linear_model.coef_
                            intercept = linear_model.intercept_
                            # Equation
                            if poly_degree == 1:
                                equation = f"y = {coefficients[1]:.2f}x + {intercept:2f}"
                            elif poly_degree == 2:
                                equation = f"y = {coefficients[2]:.2f}x$^{2}$ + {coefficients[1]:.2f}x + {intercept:.2f}"
                            else:
                                print('Calculation only for poly_degree = 1 or 2')
                                equation = 'Equation not calculated'
                
                            # Create plots
                            plt.figure(figsize=(10,10))
                            plt.scatter(best_var, y_var_test, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                
                            # Plot
                            plt.plot(x_sort, poly_pred, 'r--', linewidth = 4, label = f'{equation} \nR² = {poly_r2:.2f}')
                            plt.title(f"{station}: {best_component}, {freq[0]:.1f} Hz", fontsize = 25)
                            # Make pretty
                            plt.legend(fontsize = 20)
                            plt.xlabel("Log Seismic Power", fontsize = 20)
                            plt.ylabel(f"Observed {variable_name}", fontsize = 20)
                            plt.xticks(fontsize = 20)
                            plt.yticks(fontsize = 20)
                            plt.tight_layout()
                
                        # Find best 10 frequency bands for each component     
                        top_ten = np.argsort(var_importance)[::-1][:10]
                
                        for index in top_ten:
                
                            if index < n_bands:
                                component = "Z"
                                freq_index = index
                
                            elif index < 2 * n_bands:
                                component = "NS"
                                freq_index = index - n_bands
                
                            else:
                                component = "EW"
                                freq_index = index - 2 * n_bands
                
                            frequency = band_centres[freq_index]
                
                            # Print them
                            print(f"{component}: {frequency[0]:.1f} Hz, importance = {var_importance[index]:.4f}")
                
                    return results

                def compare_models(self,
                                   fmin = 3,
                                   fmax = 49,
                                   f_band_width = 1,
                                   step_size = 1,
                                   n_splits = 5,
                                   min_WS = 14,
                                   n_estimators=100, 
                                   max_depth=15,
                                   min_samples_split=2,
                                   min_samples_leaf=3,
                                   max_features=0.25,
                                   C= 129.15, 
                                   gamma = 0.006, 
                                   epsilon=0.05,
                                   alpha_EN=0.08, 
                                   l1_ratio=0.86, 
                                   alpha_Ridge = 10.24):
                    
                    """
                    Using multiple seismic frequency band widths as model features, 
                    this function compares the cross validation scores from 
                    Random Forest, SVR, ElasticNet, and Ridge.
                    Default parameter values are chosen based on the optimise functions
                    outputs, for best wind speed cross validation, in this file.
                    These parameters were calculated using processed CWA86 data from
                    Dec 25th-31st, 2022, and the corresponding AWS data from CASY Skiway.
                    The inputs considered a minimum wind speed threshold of 14 m/s
                    and only looked at seismic frequencies from 3 Hz to 49 Hz.
                
                    Parameters:
                        spectra (list):
                            A list of dictionaries containing the frequency, power, and Wind Speed data 
                            for each seismic component (EW, NS, Z) for each station and time period.
                        fmin (int):
                            Minimum frequency value for calculating the power of each bandwidths.
                        fmax (int):
                            Maximum frequency value for calculating the power of each  bandwidths.
                        f_band_width (int):
                            Bandwidth size. e.g. f_band_width = 1 for (f1,f2)=(1,2), 2 for (1,3), 3 for (1,4). 
                        step_size (int):
                            Frequency band step size. Set to less than f_band_width for overlapping bands. 
                        n_splits (int):
                            Number of folds the dataset is divided into during cross validation.
                        min_WS (float):
                            Minimum (threshold) wind speed for wind to exert force on a seismic sensor.
                            Hand wavey - approximately 10-15 m/s.
                
                    Random Forest Parameters:
                        n_estimators:
                            Default = 100
                        max_depth:
                            Default = 15
                        min_samples_split
                            Default = 2
                        min_samples_leaf:
                            Default = 3
                        max_features:
                            Default = 0.25
                    
                    SVR Parameters:
                        C:
                            Default = 129.15
                        gamma:
                            Default = 0.006
                        epsilon:
                            Default = 0.05
                    
                    ElasticNet Parameters:
                        alpha_EN:
                            'alpha', Default = 0.08
                        l1_ratio:
                            Default = 0.86 (More Lasso than Ridge)
                    
                    Ridge Parameters:
                        alpha_Ridge:
                            'alpha', Default = 10.24
                
                    Returns:
                        Results (dict):
                            A dictionary containing the cross validation R² results
                            for WS, sin2, cos2 for each station inspected.
                    """
                
                
                    # Setup Result Lists
                    results = []
                
                    # Bands
                    # Create Bandwidths
                    bands = []
                    for f1 in range(fmin, fmax - f_band_width + 1, step_size):
                        f2 = f1 + f_band_width
                        band = (f1, f2)
                        bands.append(band)
                    # Num of Bands
                    n_bands = len(bands)
                    # Create Band Centers
                    band_centres = []
                    for f1, f2 in bands:
                        band_centre = [(f1 + f2)/2]
                        band_centres.append(band_centre)
                
                    # Loop through stations
                    for station_dict in self.seismic_wind.fft:
                
                        # Setup Variables
                        station = list(station_dict.keys())[0] 
                        EW_power = station_dict[station][0]['EW']
                        NS_power = station_dict[station][0]['NS']
                        Z_power = station_dict[station][0]['Z']
                        freq = station_dict[station][0]['freq']
                        aws_values = station_dict[station][0]['aws_values']
                        wind_direction = station_dict[station][0]['wind_direction']
                
                        # Angle wrap around problem
                        dir_radians = np.deg2rad(wind_direction)
                        dir_sin = np.sin(dir_radians)
                        dir_cos = np.cos(dir_radians)
                        
                        # Setup results
                        station_results = []
                        
                        # Setup powers
                        X_Z = np.zeros((len(aws_values), len(bands)))
                        X_NS = np.zeros((len(aws_values), len(bands)))
                        X_EW = np.zeros((len(aws_values), len(bands)))
                
                        # Apply bandwidths to data
                        for i, (f1,f2) in enumerate(bands):
                            band_width = (freq >= f1) & (freq < f2)
                
                            # Convert to log to better inspect power scales and apply bandwidth
                            # [:, band_width], select frequencies and slice unwanted freq data from the row
                            # .mean(axis=1), mean for the selected frequency row. Reshape for model input.
                            X_Z[:, i] = np.log10(Z_power[:, band_width].mean(axis=1) + 1e-20)
                            X_NS[:, i] = np.log10(NS_power[:, band_width].mean(axis=1) + 1e-20)
                            X_EW[:, i] = np.log10(EW_power[:, band_width].mean(axis=1) + 1e-20)
                
                        # Stack all together to process all together as a larger dataset
                        X_all = np.column_stack([X_Z, X_NS, X_EW])
                        y_all = np.column_stack([aws_values, dir_sin, dir_cos])
                
                
                        # Should the wind speed threshold be applied before or after training the model?
                        # Apply minimum WS threshold
                        if min_WS is not None:
                            mask = y_all[:, 0] > min_WS
                            X_all = X_all[mask]
                            y_all = y_all[mask]
                            
                        # Train Model
                        X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
                
                        # Define every model
                
                        # Ridge
                        # Pipeline
                        # Scale, Ridge # need to optimise alpha
                        var_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha_Ridge))])
                        sin_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha_Ridge))])
                        cos_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha_Ridge))])
                
                        # ElasticNet
                        # Pipeline
                        # Scale, ElasticNet (Need to find optimal l1 still)
                        var_EN = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNet(alpha=alpha_EN, l1_ratio=l1_ratio, max_iter=50000))])
                        sin_EN = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNet(alpha=alpha_EN, l1_ratio=l1_ratio, max_iter=50000))])
                        cos_EN = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNet(alpha=alpha_EN, l1_ratio=l1_ratio, max_iter=50000))])
                
                        # SVR
                        # Scale, SVR
                        # (Need to find optimal values)
                        var_SVR = Pipeline([('scaler', StandardScaler()),('svr', SVR(C=C, gamma = gamma, epsilon=epsilon))])
                        sin_SVR = Pipeline([('scaler', StandardScaler()),('svr', SVR(C=C, gamma = gamma, epsilon=epsilon))])
                        cos_SVR = Pipeline([('scaler', StandardScaler()),('svr', SVR(C=C, gamma = gamma, epsilon=epsilon))])
                
                        # Random Forest
                        var_rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                        min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,
                                                        max_features=max_features, random_state=42, n_jobs=-1)
                        sin_rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                        min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,
                                                        max_features=max_features, random_state=42, n_jobs=-1)
                        cos_rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                        min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,
                                                        max_features=max_features, random_state=42, n_jobs=-1)
                
                        # Cross Validation for all
                        var_models = [var_ridge, var_EN, var_SVR, var_rf]
                        sin_models = [sin_ridge, sin_EN, sin_SVR, sin_rf]
                        cos_models = [cos_ridge, cos_EN, cos_SVR, cos_rf]
                        var_cv = []
                        sin_cv = []
                        cos_cv = []
                
                        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                
                        for model in range(len(var_models)):
                            var_scores = cross_val_score(var_models[model], X_all_train, y_all_train[:, 0], cv=cv, scoring='r2')
                            sin_scores = cross_val_score(sin_models[model], X_all_train, y_all_train[:, 1], cv=cv, scoring='r2')
                            cos_scores = cross_val_score(cos_models[model], X_all_train, y_all_train[:, 2], cv=cv, scoring='r2')
                
                            var_cv.append(var_scores)
                            sin_cv.append(sin_scores)
                            cos_cv.append(cos_scores)
                
                        # Store results
                        station_results.append({
                            'fmin': fmin,
                            'fmax': fmax,
                            'Ridge_var_cv_r2': round(float(var_cv[0].mean()), 4),
                            'EN_var_cv_r2': round(float(var_cv[1].mean()), 4),
                            'SVR_var_cv_r2': round(float(var_cv[2].mean()), 4),
                            'rf_var_cv_r2': round(float(var_cv[3].mean()), 4),
                            'Ridge_sin_cv_r2': round(float(sin_cv[0].mean()), 4),
                            'EN_sin_cv_r2': round(float(sin_cv[1].mean()), 4),
                            'SVR_sin_cv_r2': round(float(sin_cv[2].mean()), 4),
                            'rf_sin_cv_r2': round(float(sin_cv[3].mean()), 4),
                            'Ridge_cos_cv_r2': round(float(cos_cv[0].mean()), 4),
                            'EN_cos_cv_r2': round(float(cos_cv[1].mean()), 4),
                            'SVR_cos_cv_r2': round(float(cos_cv[2].mean()), 4),
                            'rf_cos_cv_r2': round(float(cos_cv[3].mean()), 4)
                        })
                
                        # Print results
                        print(f"{station} WS Cross Validation Results:")
                        print(f"Ridge R²: {var_cv[0].mean():.4f} +/- {var_cv[0].std():.4f}")
                        print(f"ElasticNet R²: {var_cv[1].mean():.4f} +/- {var_cv[1].std():.4f}")
                        print(f"SVR R²: {var_cv[2].mean():.4f} +/- {var_cv[2].std():.4f}")
                        print(f"Random Forest R²: {var_cv[3].mean():.4f} +/- {var_cv[3].std():.4f}")
                        
                        print('----------------------------------------------------------------')
                
                        print(f"{station} Sin2Theta Cross Validation Results:")
                        print(f"Ridge R²: {sin_cv[0].mean():.4f} +/- {sin_cv[0].std():.4f}")
                        print(f"ElasticNet R²: {sin_cv[1].mean():.4f} +/- {sin_cv[1].std():.4f}")
                        print(f"SVR R²: {sin_cv[2].mean():.4f} +/- {sin_cv[2].std():.4f}")
                        print(f"Random Forest R²: {sin_cv[3].mean():.4f} +/- {sin_cv[3].std():.4f}")
                
                        print('----------------------------------------------------------------')
                
                        print(f"{station} Cos2Theta Cross Validation Results:")
                        print(f"Ridge R²: {cos_cv[0].mean():.4f} +/- {cos_cv[0].std():.4f}")
                        print(f"ElasticNet R²: {cos_cv[1].mean():.4f} +/- {cos_cv[1].std():.4f}")
                        print(f"SVR R²: {cos_cv[2].mean():.4f} +/- {cos_cv[2].std():.4f}")
                        print(f"Random Forest R²: {cos_cv[3].mean():.4f} +/- {cos_cv[3].std():.4f}")
                
                        # Store results
                        results.append(station_results)
                
                    return results

                def polar_RF(self,
                             fmin = 1,
                             fmax = 49,
                             f_band_width = 1,
                             step_size = 1,
                             n_repeats = 3,
                             min_WS = None,
                             poly_degree = 2,
                             plot_importance = True,
                             plot_results = True,
                             plot_polar = True,
                             plot_power_aws = True,
                             variable_name = 'AWS Wind Speed (km/hr)',
                             alpha = 0.08):

                    """
                    Predicts AWS variable and Wind Direction from multiple seismic frequency band power features 
                    using a Ridge regression model. Combines all seismic components into one model.
                    Includes a y_test array of AWS and WD. 
                    Multi Output model.
                
                    Parameters:
                        spectra (list):
                            A list of dictionaries containing the frequency, power, and Wind Speed data 
                            for each seismic component (EW, NS, Z) for each station and time period.
                        fmin (int):
                            Minimum frequency value for calculating the power of each bandwidths.
                        fmax (int):
                            Maximum frequency value for calculating the power of each  bandwidths.
                        f_band_width (int):
                            Bandwidth size. e.g. f_band_width = 1 for (f1,f2)=(1,2), 2 for (1,3), 3 for (1,4). 
                        step_size (int):
                            Frequency band step size. Set to less than f_band_width for overlapping bands. 
                        n_repeats (int):
                            Number of times to repeat the permutation importance calculation for each seismic component.
                        min_WS (int):
                            Minimum wind speed (km/hr) to be considered in analysis.
                        poly_degree (int):
                            Polynomial degree for the regression in plot_power_aws (Seis Power vs Obs WS)
                        plot_stat_results (bool):
                            Plots all the R² and rmse values against frequency bandwidth centres.
                        plot_results (bool):
                            Plots the predicted vs observed values for each seismic component.
                        plot_cv (bool):
                            Plot the cross validation results as boxplots       
                        plot_power_aws (bool):
                            Plot the relationship between seismic power and AWS values.
                        variable_name (str):
                            The name of the variable being predicted (e.g., 'AWS Wind Speed (km/hr)').
                
                    Outputs:
                        results (list):
                            A list of dictionaries containing information about all the correlation results for each station.
                    """

                    # Setup Result Lists
                    results = []
                    
                    # Bands
                    # Create Bandwidths
                    bands = []
                    for f1 in range(fmin, fmax - f_band_width + 1, step_size):
                        f2 = f1 + f_band_width
                        band = (f1, f2)
                        bands.append(band)
                    # Num of Bands
                    n_bands = len(bands)
                    # Create Band Centers
                    band_centres = []
                    for f1, f2 in bands:
                        band_centre = [(f1 + f2)/2]
                        band_centres.append(band_centre)
                
                    # Loop through stations
                    for station_dict in self.seismic_wind.fft:
                
                        # Setup Variables
                        station = list(station_dict.keys())[0] 
                        EW_power = station_dict[station][0]['EW']
                        NS_power = station_dict[station][0]['NS']
                        Z_power = station_dict[station][0]['Z']
                        freq = station_dict[station][0]['freq']
                        aws_values = station_dict[station][0]['aws_values']
                        wind_direction = station_dict[station][0]['wind_direction']

                        # Polar coords
                        r = np.sqrt(EW_power**2 + NS_power**2 + Z_power**2)
                        theta = np.arctan2(NS_power, EW_power)  
                        # Phi not needed here. AWS doesnt capture phi

                        # Wrap Around Problem
                        # AWS Data
                        dir_radians = np.deg2rad(wind_direction)
                        dir_sin = np.sin(dir_radians)
                        dir_cos = np.cos(dir_radians)

                        # Seismic theta
                        theta_sin = np.sin(theta)
                        theta_cos = np.cos(theta)

                        # Setup results
                        station_results = []
                        
                        # Setup powers
                        X_r = np.zeros((len(aws_values), len(bands)))
                        X_theta_sin = np.zeros((len(aws_values), len(bands)))
                        X_theta_cos = np.zeros((len(aws_values), len(bands)))

                        # Apply bandwidths to data
                        for i, (f1,f2) in enumerate(bands):
                            band_width = (freq >= f1) & (freq < f2)
                            X_r[:, i] = np.log10(r[:, band_width].mean(axis=1) + 1e-20)
                            X_theta_sin[:, i] = (theta_sin[:, band_width].mean(axis=1))
                            X_theta_cos[:, i] = (theta_cos[:, band_width].mean(axis=1))

                        # Direction Stacks
                        X_theta_dir = np.column_stack([X_theta_sin, X_theta_cos])
                        y_aws_dir = np.column_stack([dir_sin, dir_cos])

                        # Minimum Wind Speed Threshold
                        if min_WS is not None:
                            mask_aws = aws_values > min_WS
                            X_r = X_r[mask_aws]
                            X_theta_dir = X_theta_dir[mask_aws]
                            y_aws_dir = y_aws_dir[mask_aws]

                        # Wind Speed Model
                        X_r_train, X_r_test, y_aws_values_train, y_aws_values_test = train_test_split(X_r, aws_values, test_size=0.2, random_state=42)
                        # Wind Direction Model
                        X_theta_dir_train, X_theta_dir_test, y_aws_dir_train, y_aws_dir_test = train_test_split(X_theta_dir, y_aws_dir, test_size=0.2, random_state=42)

                        # Pipeline
                        # Scale, Ridge 
                        var_model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha))])
                        sin_model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha))])
                        cos_model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha))])

                        # CV
                        cv = KFold(n_splits=5, shuffle=True, random_state=42)
                        var_scores = cross_val_score(var_model, X_r_train, y_aws_values_train, cv=cv, scoring='r2')
                        sin_scores = cross_val_score(sin_model, X_theta_dir_train, y_aws_dir_train[:, 0], cv=cv, scoring='r2')
                        cos_scores = cross_val_score(cos_model, X_theta_dir_train, y_aws_dir_train[:, 1], cv=cv, scoring='r2')
                
                        # Fit Model
                        var_model.fit(X_r_train, y_aws_values_train)
                        sin_model.fit(X_theta_dir_train, y_aws_dir_train[:, 0])
                        cos_model.fit(X_theta_dir_train, y_aws_dir_train[:, 1])

                        # Predictions
                        var_pred = var_model.predict(X_r_test)
                        sin_pred = sin_model.predict(X_theta_dir_test)
                        cos_pred = cos_model.predict(X_theta_dir_test)

                        # Round WS Prediction to one decimal so it matches with AWS Measurement
                        var_pred = np.round(var_pred, decimals=1)

                        # Observed
                        y_var_test = y_aws_values_test
                        y_sin_test = y_aws_dir_test[:, 0]
                        y_cos_test = y_aws_dir_test[:, 1]

                        # Predicted direction
                        dir_pred_radian = np.arctan2(sin_pred, cos_pred)
                        dir_pred = np.rad2deg(dir_pred_radian)

                        # Round to nearest 10 degrees
                        dir_pred = np.round(dir_pred, decimals=-1)
                        
                        # Wrap back to 0-360 degrees
                        dir_pred = dir_pred % 360 

                        # Fix angles near end points
                        dir_test_radian = np.arctan2(y_sin_test, y_cos_test)
                        dir_test = np.rad2deg(dir_test_radian) % 360
                        dir_fix = np.abs((dir_pred - dir_test + 180) % 360 - 180)
                
                        # Mean direction
                        dir_mean = np.mean(dir_fix)

                        # rmse
                        var_rmse = np.sqrt(mean_squared_error(y_var_test, var_pred))
                        sin_rmse = np.sqrt(mean_squared_error(y_sin_test, sin_pred))
                        cos_rmse = np.sqrt(mean_squared_error(y_cos_test, cos_pred))
                        dir_rmse = np.sqrt(np.mean(dir_fix**2))
                
                        # R²
                        var_r2 = r2_score(y_var_test, var_pred)
                        sin_r2 = r2_score(y_sin_test, sin_pred)
                        cos_r2 = r2_score(y_cos_test, cos_pred)

                        # Freq Importance
                        var_importance = permutation_importance(var_model, X_r_test, y_var_test, n_repeats=n_repeats, scoring='r2').importances_mean
                        sin_importance = permutation_importance(sin_model, X_theta_dir_test, y_sin_test, n_repeats=n_repeats, scoring='r2').importances_mean
                        cos_importance = permutation_importance(cos_model, X_theta_dir_test, y_cos_test, n_repeats=n_repeats, scoring='r2').importances_mean

                        # Print best result
                        # R²
                        print(f"{variable_name} R²: {var_r2:.4f}")
                        print(f"Sin R²: {sin_r2:.4f}")
                        print(f"Cos R²: {cos_r2:.4f}")

                        # rmse
                        print(f"{variable_name} rmse: {var_rmse:.4f}")
                        print(f"Sin rmse: {sin_rmse:.4f}")
                        print(f"Cos rmse: {cos_rmse:.4f}")
                
                        # Cross Validation R²
                        print(f"{variable_name} cv R²: {var_scores.mean():.4f} +/- {var_scores.std():.4f}")
                        print(f"Sin cv R²: {sin_scores.mean():.4f} +/- {sin_scores.std():.4f}")
                        print(f"Cos cv R²: {cos_scores.mean():.4f} +/- {cos_scores.std():.4f}")
                        results.append(station_results)
                
                        # WD
                        print(f"Wind direction Mean Error: {dir_mean:.2f}°")
                        print(f"Wind direction RMSE: {dir_rmse:.2f}°")

                        # Plot freq importance
                        if plot_importance == True:
                            # Plot
                            fig, ax = plt.subplots(1, 3, figsize=(8, 6))
                            ax[0].plot(band_centres, var_importance)
                            ax[0].set_title(f'R: {variable_name} \n Permutation Importance \n Acrosss Freq Spectrum')

                            ax[1].plot(band_centres, sin_importance[:n_bands])
                            ax[1].set_title(f'Sin: {variable_name} \n Permutation Importance \n Acrosss Freq Spectrum')
                            
                            ax[2].plot(band_centres, cos_importance[n_bands:2*n_bands])
                            ax[2].set_title(f'Cos: {variable_name} \n Permutation Importance \n Acrosss Freq Spectrum')
                            
                            fig.suptitle(f'{station}:')
                            fig.supxlabel('Frequency (Hz)')
                            fig.supylabel('RF Permutation Importance')
                            fig.tight_layout()                                        

                        # Obs vs Pred
                        if plot_results == True:
                            #Plot obs vs pred WS
                            # AWS Variable
                            # Plot
                            plt.figure(figsize=(10, 10))
                            plt.scatter(y_var_test, var_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                
                            # y = x line
                            plt.plot([y_var_test.min(), y_var_test.max()],
                                    [y_var_test.min(), y_var_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            # Make it pretty
                            plt.xlabel(f"Observed {variable_name}", fontsize = 20)
                            plt.ylabel(f"Predicted {variable_name}", fontsize = 20)
                            plt.title(f"{station} {variable_name}", fontsize = 25)
                            plt.xticks(fontsize = 20)
                            plt.yticks(fontsize = 20)
                            plt.legend(fontsize = 20, loc ='upper left')
                            plt.tight_layout()

                            # Plot obs vs pred Wind Direction
                            # sin2theta and cos2theta
                            fig, ax = plt.subplots(2,1, figsize=(10, 10))
                            ax[0].scatter(y_sin_test, sin_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                            ax[1].scatter(y_cos_test, cos_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                            
                            # y = x line
                            ax[0].plot([y_sin_test.min(), y_sin_test.max()],
                                    [y_sin_test.min(), y_sin_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            ax[1].plot([y_cos_test.min(), y_cos_test.max()],
                                    [y_cos_test.min(), y_cos_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            # Make it pretty
                            ax[0].set_xlabel("Observed Wind Direction Sin2Theta", fontsize = 20)
                            ax[0].set_ylabel("Predicted Wind Direction Sin2Theta", fontsize = 20)
                            ax[0].set_title(f"{station} Wind Direction Sin2Theta", fontsize = 25)
                            ax[1].set_xlabel("Observed Wind Direction Cos2Theta", fontsize = 20)
                            ax[1].set_ylabel("Predicted Wind Direction Cos2Theta", fontsize = 20)
                            ax[1].set_title(f"{station} Wind Direction Cos2Theta", fontsize = 25)
                            ax[0].tick_params(axis='both', which='major', labelsize=20)
                            ax[1].tick_params(axis='both', which='major', labelsize=20)
                            fig.tight_layout()
                
                            # Sin and Cos Converted back to angle
                            plt.figure(figsize=(10, 10))
                            plt.scatter(dir_test, dir_pred, alpha=0.7, s = 100, label ='Observed Data > Wind Speed Threshold')
                            
                            # y = x line
                            plt.plot([dir_test.min(), dir_test.max()],
                                    [dir_test.min(), dir_test.max()],
                                    'r--', linewidth = 4, label = 'y = x')
                            # Make it pretty
                            plt.xlabel("Observed Wind Direction (°)", fontsize = 20)
                            plt.ylabel("Predicted Wind Direction (°)", fontsize = 20)
                            plt.title(f"{station} Wind Direction (°)", fontsize = 25)
                            plt.xticks(fontsize = 20)
                            plt.yticks(fontsize = 20)
                            plt.legend(fontsize = 20, loc ='upper left')
                            plt.tight_layout()
                
                        # Polar plots
                        if plot_polar == True:
                            #Create plot
                            plt.figure(figsize=(10,6))
                            ax = plt.subplot(projection='polar')
                            plt.polar()

                            obs = ax.scatter(dir_test_radian, y_var_test, alpha=0.7, s = 60, label = f'Observed {variable_name}', marker='x')
                            pred = ax.scatter(np.deg2rad(dir_pred), var_pred, alpha=0.7, s = 60, label = f'Predicted {variable_name}', marker='^')

                            # Make it pretty
                            ax.set_theta_zero_location('N')
                            ax.set_theta_direction(-1)
                            ax.set_ylabel(f'{variable_name}', labelpad=55, fontsize = 12)
                            ax.set_title(f"{station}: {variable_name} and Wind Direction", pad = 60, fontsize = 25)
                            ax.set_rlabel_position(0)
                            ax.tick_params(labelsize = 12)
                            rmax = ax.get_rmax()

                            cardinals = {"N": 0,           
                            "E": (np.pi / 2),
                            "S": (np.pi),
                            "W": (3 * np.pi / 2)}

                            # Add cardinal direction labels
                            for label, angle in cardinals.items():
                                            ax.text(
                                                angle,
                                                rmax * 1.3,
                                                label,
                                                ha="center",
                                                va="center",
                                                fontsize=15,
                                                fontweight="bold",
                                                clip_on=False)
                            # More pretty
                            plt.legend(loc = 'upper right', bbox_to_anchor = (1.45, 1.2), fontsize = 8) 

                            plt.tight_layout()