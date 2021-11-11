import csv
filename = 'output.csv'
infilename = 'C:\\Users\\Saint\\Downloads\\k_menas_test_41'
fields = ['k', 'threshold', 'tp', 'fn', 'tn', 'fp', 'accuracy', 'tpr', 'fpr', 'f1-score']
with open(infilename, 'r') as infile:
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(fields)
        rows = list()
        k_value = 0
        threshold_value = 0
        for line in infile:
            line = line.strip()
            if line.startswith('{'):
                row = list()
                row.append(k_value)
                row.append(threshold_value)
                line = line.replace('{', '').replace('}', '')
                fields = line.split(', ')
                for field in fields:
                    row.append(field.split(': ')[1])
                rows.append(row)
            else:
                k_value_string, threshold_value_string = line.split(' ')
                k_value = int(k_value_string)
                threshold_value = round(float(threshold_value_string), 2)
        writer.writerows(rows)
        print('done!')



