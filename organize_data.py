import os
import shutil

# Base directory containing subject folders
base_dir = "/home/ehost/syoon/data"

# List all subject directories
subject_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

for subject in subject_dirs:
    subject_path = os.path.join(base_dir, subject)
    mrimages_dir = os.path.join(subject_path, [d for d in os.listdir(subject_path) if d.startswith('MRimages_') and os.path.isdir(os.path.join(subject_path, d))][0])

    if os.path.exists(mrimages_dir):
        for item in os.listdir(mrimages_dir):
            src_path = os.path.join(mrimages_dir, item)
            dst_path = os.path.join(subject_path, item)
            # Move the file or directory to the subject path
            shutil.move(src_path, dst_path)

        # Remove the now-empty MRimages_{number}.nii.gz directory
        shutil.rmtree(mrimages_dir)

print("Reorganization complete.")

