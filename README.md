# Demucs-v2-ps-ts
This repository is based on the [P9-handin repository](https://github.com/madslangl/P9-handin) created by Mads Lang Laursen. 

Code is originally based on Demucs V2. The original readme is included in this folder as README-original.md.

In addition to the original repository, this version includes new models that have been added to extend its functionality.

### Running the code:
Please install the Conda environment by,
```cmd
conda env create -f environment.yml
```

This creates a conda environment named demucs2-0-3, which should be activated by:
```cmd
conda activate demucs2-0-3
```
This includes all relevant software, **except** for ffmpeg, sox, and soundstretch which should be installed by:
```cmd
sudo apt install -y ffmpeg sox soundstretch
```

The SI-SDR based model is included in models directory as git LFS object.

The conversation generation script is confidential and non-published.

Metadata is included for some datasets. The code will automatically generate the metadata files, but this takes a while, so for ease of use they are included here.