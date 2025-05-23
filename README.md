# Demucs-v2-ps-ts
This repository is based on the [P9-handin repository](https://github.com/madslangl/P9-handin) created by Mads Lang Laursen. 
Code is originally based on Demucs V2. The original readme is included in this folder as README-original.md.
In addition to the original repository, this version includes new models that have been added to extend its functionality.

### Running the code:
To install python dependencies run
```
pip install -r requirements.txt
```

This includes all relevant software, **except** for ffmpeg, sox, and soundstretch which should be installed by:
```cmd
sudo apt install -y ffmpeg sox soundstretch
```

Running the training on a single GPU: 
```cmd
python3 -m demucs -b 16 -e 150 --PITSISDR --repeat 1 --audio_channels 1 --wav <train_and_eval_dataset_path> --musdb <test_dataset_path>
```

Running the training on multiple GPUs:
```cmd 
python3 run.py -b 128 -e 150 --PITSISDR --repeat 1 --audio_channels 1 --wav <train_and_eval_dataset_path> --musdb <test_dataset_path> 
```

Running the inference:
```cmd
python3 -m demucs.separate -n <model_name> <mixture_wav_path>
```

### Models
Trained models are available in the [`Models`](./Models) directory. They are stored as Git LFS objects.

### Datasets
Dataset can be shared upon a request. 

### Singularity & Slurm
The repository contains a demucs.def file for a Singularity container that can be used to run the training. 
Additionally, scripts for training on with Slurm Workload Managers are found in [`Slurm`](./slurm) directory.