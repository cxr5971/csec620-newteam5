import pandas as pd








def extract_features(file_name,file_type):
    #The following features are given by the dataset
    #Duration, Protocol, SrcAddress, SrcPort, Direction, DestAddress, DestPort, State, srcTypeOfService (sTos), destTypeOfService (dTos), TotalPackets, TotalBytes
    #We want to parse the data and split into time segments 
    data_file = pd.read_csv(file_name)
    data_file['StartTime'] = pd.to_numeric(pd.to_datetime(data_file['StartTime']),errors='raise')
    start = data_file['StartTime'][0]
    #Need to calculate a time window and assign it as a new 'feature' to each sample



    #line_index = 0
    #with open(file_name, 'r') as data_file:
    #    for line in data_file:
    #        if line_index == 0:
    #            feature_label_dict = {}
    #            feature_labels = line.split(',')
    #            for label in feature_labels:
    #                feature_label_dict[label] = 0
    #            line_index += 1
    #        else:
    #            #Extract the data into time windows, using 2 minutes for now
    #            net_data_entry = line.split(',')
    #           # start_time = 
     
            



def main():
    
    extract_features('./datasets/CTU-Malware-Capture-Botnet-1/2013-10-01_capture-win8.netflow.csv',None)

main()
