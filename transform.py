import os
import subprocess

# Paths
input_dir = "/home/ehost/syoon/OASIS3_fastsurfer"
output_dir = "/home/ehost/syoon/data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Find MR image files
origin_images = sorted([f for f in os.listdir(input_dir) if f.startswith('MRimages') and f.endswith('nii.gz')])
label_images = sorted([f for f in os.listdir(input_dir) if f.startswith('Mask') and f.endswith('nii.gz')])

# Loop through each MR image and corresponding mask image
for origin_image, label_image in zip(origin_images, label_images):
    subject_id = os.path.splitext(os.path.splitext(origin_image)[0])[0].split('_')[1]
    input_image = os.path.join(input_dir, origin_image)
    label_image_path = os.path.join(input_dir, label_image)

    # Output directory for the current subject
    subject_output_dir = os.path.join(output_dir, f"subject{subject_id}")
    if not os.path.exists(subject_output_dir):
        os.makedirs(subject_output_dir)
    
    cmd = [
        "python", "/home/ehost/syoon/FastSurfer/FastSurferCNN/run_prediction.py",
        "--t1", input_image,
        "--sd", subject_output_dir,
        "--batch_size", "1",
        "--threads", "1",
        "--device", "cuda",
        "--lut", "/home/ehost/syoon/FastSurfer/FastSurferCNN/config/FastSurfer_ColorLUT.tsv",
        "--brainmask_name", "brainmask.mgz",
        "--asegdkt_segfile", "aparc.DKTatals+aseg.mgz",
        "--conformed_name", "orig.mgz"
    ]

    # Print and run the command
    print(f"Running command for subject {subject_id}: {' '.join(cmd)}")
    subprocess.run(cmd)




