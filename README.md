# P9-handin
Made by Mads Lang Laursen on a project-oriented stay in a collaboration with Aalborg University and GN Hearing.

Code is based on Demucs V2. Please refer to the report for relevant descriptions, comments, and references. 

### Running the code:
Please install the Conda environment by,
```cmd
conda env create -f environment.yml
```

This creates a conda evnvironment named demucs2-0-3, which should be activated by:
```cmd
conda activate demucs2-0-3
```
This includes all relevant software, **except** for ffmpeg, sox and soundstrech which should be installed by:
```cmd
sudo apt install -y ffmpeg sox soundstretch
```


The SI-SDR based model is included in the link below, together with the preliminary models.

The full data set is available upon request. The validation data set and the subset used for training preliminary models can be found here:

```link
https://www.dropbox.com/scl/fo/d94zy2qs24l97jy9k7n1d/AD5e_EyqEihlY3W-KVVIQQg?rlkey=bnjarmetrn09yernwypy6jdq0&st=7jzj2kqj&dl=0
```

The conversation generation script is unfortunately confidential and non-published. Upon request, I can give a demonstration, but it is not available for publication at the time of writing.
