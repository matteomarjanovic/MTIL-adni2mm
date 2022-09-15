import csv

rows = []
with open('MTIL-adni2mm/lookupcsv/ADNI1_prob_full_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if row[2] != 'mmse':
            row[2] = str(int(row[2])/30)[:5]
        rows.append(row)

with open('MTIL-adni2mm/lookupcsv/ADNI1_prob_full_data_scaled_mmse.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in rows:
        spamwriter.writerow(row)