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