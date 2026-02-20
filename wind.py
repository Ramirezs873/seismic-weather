from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px


def read_data(path, station_code, id_code):
    
    data_path = Path(path)
    pq_path = data_path / f'HM01X_Data_{station_code}_{id_code}.parquet'
    csv_path = data_path / f'HM01X_Data_{station_code}_{id_code}.txt'

    if pq_path.exists():
        df = pd.read_parquet(pq_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=['datetime'])
    else:
        raise FileNotFoundError(f"No data file found for station {station_code} and id {id_code}")

    return df