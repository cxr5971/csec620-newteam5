import numpy as np
import pandas as pd

from scipy import stats


def get_filename(filename):
    """A simple function that grabs the base of the data files name to use for storing data
       :param filename: str: The data file provided for processing
       :returns: The name of the file without the extension for saving data down the line
    """
    temp = filename.split("/")[-1].strip()
    base_name = temp.split(".")[0].strip()
    return base_name


def normalize(data_frame, column):
    """
    A simple function to normalize the data in a column using min max normalization
    :param data_frame: pandas.DataFrame: All the data
    :param column: list(str): A list of the columns to normalize
    :return: The data frame with the normalized data
    """
    data_frame[column] = (data_frame[column] - data_frame[column].min())/\
                         (data_frame[column].max() - data_frame[column].min())
    return data_frame


def entropy(dataframe):
    """
    Calculates the weighted subgroup entropy of a colum
    :param dataframe: pandas.DataFrame: All the data
    :return: The entropy
    """
    if dataframe.shape[0] == 1:
        return 1.0
    else:
        probability = dataframe.value_counts()/dataframe.shape[0]
        h = probability * np.log10(probability)
        return -h.sum()/np.log10(dataframe.shape[0])  # -h.sum = Entropy of the data


def extract_features(file_name, file_type):
    # The following features are given by the dataset
    # Duration, Protocol, SrcAddress, SrcPort, Direction, DestAddress, DestPort, State, srcTypeOfService (sTos),
    # destTypeOfService (dTos), TotalPackets, TotalBytes, and SrcBytes
    # We want to parse the data and split into time segments

    # Step 1: Load the data
    print(f"Reading in data from {file_name}")
    base_name = get_filename(file_name)
    data_file = pd.read_csv(file_name)

    # Step 2: Adjust the time field into something more usable
    data_file['StartTime'] = pd.to_numeric(pd.to_datetime(data_file['StartTime']), errors='raise')

    # Step 3: Shorten labels for ease of use
    data_file['Label'], labels = pd.factorize(data_file['Label'].str.slice(0, 15))

    # Step 4: Aggregate the data packets into time windows
    print("Beginning to aggregate packets into time windows")
    time_windows = pd.DataFrame()
    time_window = pd.DataFrame()
    time = None
    window_count = 0
    sample_count = data_file.shape[0]
    for loop in range(0, sample_count, 50):  # Process data in larger batches to improve performance
        if loop % 100000 == 0:  # Sanity check
            print(f"\tTime window creation {round((loop/sample_count * 100), 2)}%")

        current_samples = data_file[loop: loop + 50]

        # If there is no time  make the first sample the new start of the window
        if time is None:
            time = current_samples.iloc[0]["StartTime"]

        # If the last sample is within the time window add all sample to the window
        if current_samples.iloc[-1]['StartTime'] <= time + 20:
            time_window = time_window.append(current_samples)
        else:
            for i in range(current_samples.shape[0], -1, -1):  # Otherwise find which samples belong in the window
                if current_samples.iloc[-1]['StartTime'] <= time + 20:
                    break
            time_window = time_window.append(current_samples[:i + 1])  # Add them
            gb = time_window.groupby('SrcAddr')  # Aggregate the data
            time_windows = time_windows.append(gb.size().to_frame(
                name='counts').join(gb.agg({'Sport': ['nunique', entropy], 'DstAddr': ['nunique', entropy],
                                            'Dport': ['nunique', entropy], 'Dur': ['sum', 'mean', 'std', 'max', 'median'],
                                            'TotBytes': ['sum', 'mean', 'std', 'max', 'median'],
                                            'SrcBytes': ['sum', 'mean', 'std', 'max', 'median'],
                                            'Label': lambda x: stats.mode(x)[0]}
                                           )).reset_index().assign(window_id=window_count))

            # Reset the window and add any left over samples to it
            time_window = pd.DataFrame()
            time_window = time_window.append(current_samples[i + 1:])
            # If we had left over samples, the first sample becomes the new start time
            if len(time_window) != 0:
                time = time_window.iloc[0]["StartTime"]
            else:
                time = None

    # Aggregate and process the data one more time to save the final window
    gb = time_window.groupby('SrcAddr')
    time_windows = time_windows.append(gb.size().to_frame(
        name='counts').join(gb.agg({'Sport': ['nunique', entropy], 'DstAddr': ['nunique', entropy],
                                    'Dport': ['nunique', entropy], 'Dur': ['sum', 'mean', 'std', 'max', 'median'],
                                    'TotBytes': ['sum', 'mean', 'std', 'max', 'median'],
                                    'SrcBytes': ['sum', 'mean', 'std', 'max', 'median'],
                                    'Label': lambda x: stats.mode(x)[0]}
                                   )).reset_index().assign(window_id=window_count))
    print("Aggregation complete\nCleaning up column names")

    # Step 5: Clean up the column names
    data_columns = []
    for column in time_windows.columns.ravel():
        if isinstance(column, tuple):
            data_columns.append("_".join(column))
        else:
            data_columns.append(column)
    time_windows.columns = data_columns

    print("Columns renamed\nReplacing empty values with -1")
    # Step 6: Replace missing data with -1
    time_windows.fillna(-1, inplace=True)

    # Step 7 : Normalize numerical columns
    data_to_normalize = list(time_windows.columns.values)
    data_to_normalize.remove('SrcAddr')
    data_to_normalize.remove('Label_<lambda>')
    data_to_normalize.remove('window_id')

    print(f"Normalizing columns: {data_to_normalize}")
    normalize(time_windows, data_to_normalize)

    # Step 8: Save data to files
    print("Saving data to disk")
    time_windows.drop('SrcAddr', axis=1).to_hdf(f"{base_name}.h5", key="data", mode="w")
    np.save(f"{base_name}_id.npy", time_windows['SrcAddr'])
    np.save(f"{base_name}_labels.npy", labels)


def main():
    
    extract_features('./datasets/capture20110810.binetflow', None)


if __name__ == '__main__':
    main()
