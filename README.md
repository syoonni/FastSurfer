### General

1. Install System Packages
    
    ```bash
    sudo apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    ca-certificates \
    file
    
    ```
    
2. FastSurfer
    
    ```bash
    git clone --branch stable https://github.com/Deep-MI/FastSurfer.git
    cd FastSurfer
    ```
    
3. Python environment
    
    ```bash
    conda env create -f ./env/fastsurfer.yml 
    conda activate fastsurfer
    ```
    
    - add the fastsurfer directory to the python path
    
    ```bash
    export PYTHONPATH="${PYTHONPATH}:$PWD"
    echo "export PYTHONPATH=\"\${PYTHONPATH}:$PWD\"" >> ~/.bashrc\
    ```
    

1. Download Network Checkpoints
    
    ```bash
    python3 FastSurferCNN/download_checkpoints.py --all
    
    ```
    

1. 원본 데이터셋 형식 FastSurfer의 필요형식으로 변환 (
    - `Mask_x.nii.gz` 파일들을 `orig.mgz`, `aparc.DKTatlas+aseg.mgz`, `aseg.auto_noCCseg.mgz` 파일로 변환하고 이동
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/e7b50649-d2d9-4695-878c-45872b4c130e/60679928-2a7c-487f-b91e-dca2184e807c/Untitled.png)
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/e7b50649-d2d9-4695-878c-45872b4c130e/4f6d7f72-f5f3-4f91-a354-e27cb6c5b2d0/Untitled.png)
        
        - dataset : 2678개(24시간 이상 걸림)
        - run_prediction.py 자신의 환경에 맞게 실행하는 코드
        - transform.py
            
            ```python
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
            ```
            
2. dataset 나누기
    - split1.py(train 1000개, test 300개)
        
        ```python
        import os
        import shutil
        
        # Set up a data path
        data_dir = "/home/ehost/syoon/data"
        subjects = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))], key=lambda x: int(x.replace("subject", "")))
        
        # Specify train and test subject ranges
        train_subjects = subjects[0:1000]
        test_subjects = subjects[-300:]
        
        # Set path
        train_dir = os.path.join(data_dir, "/home/ehost/syoon/dataset/train")
        test_dir = os.path.join(data_dir, "/home/ehost/syoon/dataset/test")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        def copy_files(subjects, destination):
            for subject in subjects:
                subject_path = os.path.join(data_dir, subject)
                dest_path = os.path.join(destination, subject)
                shutil.copytree(subject_path, dest_path)
        
        # Copy data
        copy_files(train_subjects, train_dir)
        copy_files(test_subjects, test_dir)
        
        print(f"Train subjects: {len(train_subjects)}")
        print(f"Test subjects: {len(test_subjects)}")
        ```
        
3. dataset directory CSV파일 생성 (여기부터 다시)
    - generate_csv.py
        
        ```python
        import os
        import csv
        
        data_dir = '/home/ehost/syoon/dataset1000/train'
        csv_filename = '/home/ehost/syoon/training_set_subjects_dirs1000.csv'
        
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
        ```
        

### Hdf5-Trainingset Generation

- data 1000개
- generate_hdf5.py 코드 수정한 함수들
    - def _load_volumes()
    
    ```python
        def _load_volumes(self, subject_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple]:
            try:
                print(f"subject_path = {subject_path}")
    
                orig_path = join(subject_path, self.orig_name)
                aseg_path = join(subject_path, self.aparc_name)
                aseg_nocc_path = join(subject_path, self.aparc_nocc) if self.aparc_nocc else None
    
                print(f"Loading original image from {orig_path}")
                orig = nib.load(orig_path)
                orig_data = np.asarray(orig.get_fdata(), dtype=np.uint8)
    
                zoom = orig.header.get_zooms()
    
                print(f"Loaded ground truth segmentation from {aseg_path}")
                aseg = nib.load(aseg_path)
                aseg_data = np.asarray(aseg.get_fdata(), dtype=np.int16)
    
                if aseg_nocc_path:
                    print(f"Loading segmentation without corpus callosum from {aseg_nocc_path}")
                    aseg_nocc = nib.load(aseg_nocc_path)
                    aseg_nocc_data = np.asarray(aseg_nocc.get_fdata(), dtype=np.int16)
                else:
                    aseg_nocc_data = None
    
                return orig_data, aseg_data, aseg_nocc_data, zoom
    
            except Exception as e:
                print(f"Error loading volumes from {subject_path}: {e}")
                raise
    ```
    
    - def create_hdf5_dataset()
    
    ```python
        def create_hdf5_dataset(self, blt: int):
            data_per_size = defaultdict(lambda: defaultdict(list))
            start_d = time.time()
    
            for idx, current_subject in enumerate(self.subject_dirs):
                try:
                    start = time.time()
    
                    print(f"Processing subject {idx + 1}/{len(self.subject_dirs)}: {current_subject}")
    
                    orig, aseg, aseg_nocc, zoom = self._load_volumes(current_subject)
                    size, _, _ = orig.shape
    
                    mapped_aseg, mapped_aseg_sag = map_aparc_aseg2label(
                        aseg,
                        self.labels,
                        self.labels_sag,
                        self.lateralization,
                        aseg_nocc,
                        processing=self.processing,
                    )
    
                    if self.plane == "sagittal":
                        mapped_aseg = mapped_aseg_sag
                        weights = create_weight_mask(
                            mapped_aseg,
                            max_weight=self.max_weight,
                            ctx_thresh=19,
                            max_edge_weight=self.edge_weight,
                            max_hires_weight=self.hires_weight,
                            cortex_mask=self.gm_mask,
                            gradient=self.gradient,
                        )
    
                    else:
                        weights = create_weight_mask(
                            mapped_aseg,
                            max_weight=self.max_weight,
                            ctx_thresh=33,
                            max_edge_weight=self.edge_weight,
                            max_hires_weight=self.hires_weight,
                            cortex_mask=self.gm_mask,
                            gradient=self.gradient,
                        )
    
                    print(f"Created weights for subject {idx + 1}")
    
                    # transform volumes to correct shape
                    [orig, mapped_aseg, weights], zoom = self.transform(self.plane, [orig, mapped_aseg, weights], zoom)
    
                    # Create Thick Slices, filter out blanks
                    orig_thick = get_thick_slices(orig, self.slice_thickness)
    
                    orig, mapped_aseg, weights = filter_blank_slices_thick(
                        orig_thick, mapped_aseg, weights, threshold=blt
                    )
    
                    num_batch = orig.shape[2]
                    orig = np.transpose(orig, (2, 0, 1, 3))
                    mapped_aseg = np.transpose(mapped_aseg, (2, 0, 1))
                    weights = np.transpose(weights, (2, 0, 1))
    
                    data_per_size[f"{size}"]["orig"].extend(orig)
                    data_per_size[f"{size}"]["aseg"].extend(mapped_aseg)
                    data_per_size[f"{size}"]["weight"].extend(weights)
                    data_per_size[f"{size}"]["zoom"].extend((zoom,) * num_batch)
                    sub_name = current_subject.split("/")[-1]
                    data_per_size[f"{size}"]["subject"].append(
                        sub_name.encode("ascii", "ignore")
                    )
    
                    print(f"Processed subject {idx + 1}")
    
                except Exception as e:
                    #LOGGER.info("Volume: {} Failed Reading Data. Error: {}".format(idx, e))
                    print(f"Volume: {idx + 1} Failed Reading Data. Error: {e}")
                    continue
    
            for key, data_dict in data_per_size.items():
                data_per_size[key]["orig"] = np.asarray(data_dict["orig"], dtype=np.uint8)
                data_per_size[key]["aseg"] = np.asarray(data_dict["aseg"], dtype=np.uint8)
                data_per_size[key]["weight"] = np.asarray(data_dict["weight"], dtype=float)
    
            with h5py.File(self.dataset_name, "w") as hf:
                dt = h5py.special_dtype(vlen=str)
                for key, data_dict in data_per_size.items():
                    group = hf.create_group(f"{key}")
                    group.create_dataset("orig_dataset", data=data_dict["orig"])
                    group.create_dataset("aseg_dataset", data=data_dict["aseg"])
                    group.create_dataset("weight_dataset", data=data_dict["weight"])
                    group.create_dataset("zoom_dataset", data=data_dict["zoom"])
                    group.create_dataset("subject", data=data_dict["subject"], dtype=dt)
    
            end_d = time.time() - start_d
            print(f"Successfully written {self.dataset_name} in {end_d:.3f} seconds.")
    ```
    

```bash
cd FastSurfer/FastSurferCNN
```

```bash
python3 generate_hdf5.py \
--hdf5_name /home/ehost/syoon/hdf5/training_set_sagittal.hdf5 \
--data_dir /home/ehost/syoon/dataset1000/train \
--csv_file /home/ehost/syoon/training_set_subjects_dirs1000.csv \
--plane sagittal \
--image_name orig.mgz \
--gt_name aparc.DKTatlas+aseg.mgz \
--gt_nocc mri/aseg.auto_noCCseg.mgz
```

```bash
python3 generate_hdf5.py \
--hdf5_name /home/ehost/syoon/hdf5/validation_set_sagittal.hdf5 \
--data_dir /home/ehost/syoon/dataset1000/val \
--csv_file /home/ehost/syoon/validation_set_subjects_dirs1000.csv \
--plane sagittal \
--image_name orig.mgz \
--gt_name aparc.DKTatlas+aseg.mgz \
--gt_nocc mri/aseg.auto_noCCseg.mgz
```

```bash
python3 generate_hdf5.py \
--hdf5_name /home/ehost/syoon/hdf5/training_set_coronal.hdf5 \
--data_dir /home/ehost/syoon/dataset1000/train \
--csv_file /home/ehost/syoon/training_set_subjects_dirs1000.csv \
--plane coronal \
--image_name orig.mgz \
--gt_name aparc.DKTatlas+aseg.mgz \
--gt_nocc mri/aseg.auto_noCCseg.mgz
```

```bash
python3 generate_hdf5.py \
--hdf5_name /home/ehost/syoon/hdf5/validation_set_coronal.hdf5 \
--data_dir /home/ehost/syoon/dataset1000/val \
--csv_file /home/ehost/syoon/validation_set_subjects_dirs1000.csv \
--plane coronal \
--image_name orig.mgz \
--gt_name aparc.DKTatlas+aseg.mgz \
--gt_nocc mri/aseg.auto_noCCseg.mgz
```

```bash
python3 generate_hdf5.py \
--hdf5_name /home/ehost/syoon/hdf5/training_set_axial.hdf5 \
--data_dir /home/ehost/syoon/dataset1000/train \
--csv_file /home/ehost/syoon/training_set_subjects_dirs1000.csv \
--plane axial \
--image_name orig.mgz \
--gt_name aparc.DKTatlas+aseg.mgz \
--gt_nocc mri/aseg.auto_noCCseg.mgz
```

```bash
python3 generate_hdf5.py \
--hdf5_name /home/ehost/syoon/hdf5/validataion_set_axial.hdf5 \
--data_dir /home/ehost/syoon/dataset1000/val \
--csv_file /home/ehost/syoon/validation_set_subjects_dirs1000.csv \
--plane axial \
--image_name orig.mgz \
--gt_name aparc.DKTatlas+aseg.mgz \
--gt_nocc mri/aseg.auto_noCCseg.mgz
```

optimization.py에서 networks import error 이슈

→ from FastSurferCNN.models.networks import FastSurferCNN, FastSurferVINN

```python
python3 run_model.py \
--cfg config/FastSurferVINN.yaml \
DATA.PATH_HDF5_TRAIN hdf5_sets1000/training_set_coronal.hdf5 \
DATA.PATH_HDF5_VAL hdf5_sets1000/validation_set_coronal.hdf5 \
DATA.PLANE coronal

```

```python
python3 run_model.py \
--cfg config/FastSurferVINN.yaml \
DATA.PATH_HDF5_TRAIN hdf5_sets1000/training_set_axial.hdf5 \
DATA.PATH_HDF5_VAL hdf5_sets1000/validation_set_axial.hdf5 \
DATA.PLANE axial

```

```python
python3 run_model.py \
--cfg config/FastSurferVINN.yaml \
DATA.PATH_HDF5_TRAIN hdf5_sets1000/training_set_sagittal.hdf5 \
DATA.PATH_HDF5_VAL hdf5_sets1000/validation_set_sagittal.hdf5 \
DATA.PLANE sagittal

```