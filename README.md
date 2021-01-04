# Analysis of fMRI Data Through Deep Collapsed Nonparametric Von Mises-Fisher Mixture Models

## Requirements

To install requirements:

```setup    
conda env create -f environment.yml
conda activate Test
```

## File

    datasets/                      # container of data  
    vae/                           # SVRAE code  
    vmfmm/                         # Co-vMFMM code
    runs/                          # runing result  

## Training

To train the model(s) in the paper, run this command:  

    __params:__  
    -r   # name of runing folder, default is SVAE-CvMFMM  
    -n   # training epoch, default is 50  
    -s   # data set name, default is adhd  
    -v   # version of training, default is 1  

Runing example:
```
python trainSVAE.py
```

Note: the reparametation trick code is taken from (https://github.com/nicola-decao/s-vae-pytorch)


