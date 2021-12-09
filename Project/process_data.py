import numpy as np
import pandas as pd

from statistics import mode


def get_filename(filename):
    temp = filename.split("/")[-1].strip()
    base_name = temp.split(".")[0].strip()
    return base_name


def normalize(data_frame, column):
    data_frame[column] = (data_frame[column] - data_frame[column].min())/\
                         (data_frame[column].max() - data_frame[column].min())
    return data_frame


def extract_features(file_name, file_type):
    #The following features are given by the dataset
    #Duration, Protocol, SrcAddress, SrcPort, Direction, DestAddress, DestPort, State, srcTypeOfService (sTos), destTypeOfService (dTos), TotalPackets, TotalBytes
    #We want to parse the data and split into time segments
    base_name = get_filename(file_name)
    data_file = pd.read_csv(file_name)
    data_file['StartTime'] = pd.to_numeric(pd.to_datetime(data_file['StartTime']), errors='raise')
    start = data_file['StartTime'][0]
    data_file['Window_lower'] = (data_file['StartTime']-start-120)  # Divide by an additional aggregation window???
    data_file['Window_upper'] = (data_file['StartTime']-start)

    data_file['Label'], labels = pd.factorize(data_file['Label'].str.slice(0, 15))

    time_windows = pd.DataFrame()
    time_window = pd.DataFrame()
    time = None
    loop = window_count = 0
    for loop in range(0, data_file.shape[0], 50):
        print(loop)
        current_samples = data_file[loop: loop + 50]

        if time is None:
            time = current_samples.iloc[0]["StartTime"]

        if current_samples.iloc[-1]['StartTime'] <= time + 20:
            time_window = time_window.append(current_samples)
        else:
            for i in range(current_samples.shape[0], -1, -1):
                if current_samples.iloc[-1]['StartTime'] <= time + 20:
                    break
            time_window = time_window.append(current_samples[:i + 1])
            gb = time_window.groupby('SrcAddr')
            time_windows = time_windows.append(gb.size().to_frame(name='counts').join(gb.agg({'Sport': 'nunique',
                                                                                              'DstAddr':'nunique',
                                                                                              'Dport':'nunique',
                                                                                              'Dur':['sum', 'mean', 'std', 'max', 'median'],
                                                                                              'TotBytes':['sum', 'mean', 'std', 'max', 'median'],
                                                                                              'Label':lambda x: mode(x)})).reset_index().assign(window_id=window_count))
            time_window = pd.DataFrame()
            time_window = time_window.append(current_samples[i + 1:])
            if len(time_window) != 0:
                time = time_window.iloc[0]["StartTime"]
            else:
                time = None
    gb = time_window.groupby('SrcAddr')
    time_windows = time_windows.append(gb.size().to_frame(name='counts').join(gb.agg({'Sport': 'nunique',
                                                                                      'DstAddr': 'nunique',
                                                                                      'Dport': 'nunique',
                                                                                      'Dur': ['sum', 'mean', 'std',
                                                                                              'max', 'median'],
                                                                                      'TotBytes': ['sum', 'mean', 'std',
                                                                                                   'max', 'median'],
                                                                                      'Label': lambda x: mode(
                                                                                          x)})).reset_index().assign(
        window_id=window_count))
    print(time_windows)

    data_columns = []
    for column in time_windows.columns.ravel():
        if isinstance(column, tuple):
            data_columns.append("_".join(column))
        else:
            data_columns.append(column)
    time_windows.columns = data_columns

    time_windows.fillna(-1, inplace=True)

    data_to_normalize = data_columns[:]
    data_to_normalize.remove('ScrAddr')
    data_to_normalize.remove('Label_<lambda>')
    data_to_normalize.remove('window_id')

    normalize(time_windows, data_to_normalize)

    time_windows.drop('SrcAddr', axis=1).to_hdf(f"{base_name}.h5", key="data", mode="w")
    np.save(f"{base_name}_id.npy", time_windows['SrcAddr'])
    np.save(f"{base_name}_labels.npy", labels)
     
            

def main():
    
    extract_features('./datasets/CTU-Malware-Capture-Botnet-1/2013-10-01_capture-win8.netflow.csv',None)

main()
