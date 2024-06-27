import os
import csv

data_dir = '/home/ehost/syoon/dataset1000/val'
csv_filename = '/home/ehost/syoon/validation_set_subjects_dirs1000.csv'

# Get subject directories
subject_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# Sort directories numerically
subject_dirs = sorted(subject_dirs, key=lambda x: int(x.replace('subject', '')))

# Create full paths
subject_dirs = [os.path.join(data_dir, subject) for subject in subject_dirs]

# Write to CSV
with open(csv_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    for subject in subject_dirs:
        csvwriter.writerow([subject])

print(f'{csv_filename} The file has been created.')
