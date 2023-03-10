import csv
import random

filename = '/home/student/Projects/MTIL-adni2mm/lookupcsv/ADNI1_ADNI2_mmse.csv'

# Read the CSV file into a list of rows
with open(filename, 'r') as file:
    reader = csv.reader(file)
    rows = list(reader)

# Shuffle the rows randomly
random.shuffle(rows)

# Write the shuffled rows back to the same file
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)
