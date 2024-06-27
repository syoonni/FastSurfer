import os
import shutil

# Set up a data path
data_dir = "/home/ehost/syoon/data"
subjects = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))], key=lambda x: int(x.replace("subject", "")))

# Specify train and test subject ranges
train_subjects = subjects[0:1000]
val_subjects = subjects[1000:1200]
test_subjects = subjects[-300:]

# Set path
train_dir = os.path.join(data_dir, "/home/ehost/syoon/dataset1000/train")
val_dir = os.path.join(data_dir, "/home/ehost/syoon/dataset1000/val")
test_dir = os.path.join(data_dir, "/home/ehost/syoon/dataset1000/test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

def copy_files(subjects, destination):
    for subject in subjects:
        subject_path = os.path.join(data_dir, subject)
        dest_path = os.path.join(destination, subject)
        shutil.copytree(subject_path, dest_path)

# Copy data
copy_files(train_subjects, train_dir)
copy_files(val_subjects, val_dir)
copy_files(test_subjects, test_dir)

print(f"Train subjects: {len(train_subjects)}")
print(f"Validation subjects: {len(train_subjects)}")
print(f"Test subjects: {len(test_subjects)}")
