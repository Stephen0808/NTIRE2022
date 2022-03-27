# NTIRE2022
This repo is for NTIRE2022 Perceptual Image Quality Assessment Challenge Track 2 No-Reference competition.

## Intergration Models
When intergrate different models for better performance, put the code into train_config directory.

The final test output is put into the final_test_output directory.

## Dataset
Before running the codes, you should download the PIPAL datasets for [train](https://drive.google.com/drive/folders/1G4fLeDcq6uQQmYdkjYUHhzyel4Pz81p-?usp=sharing) and [val](https://drive.google.com/drive/folders/1w0wFYHj8iQ8FgA9-YaKZLq7HAtykckXn).Note that although we load the reference images, but we only use the distorted images as input for training and testing.

Change the path in train.py from line 206 to line 209.

## Training
Rename the model in line 237 in train.py. Run the code:
```
$ CUDA_VISIBLE_DEVICES=num_gpu python train.py
```
## Testing
Change the checkpoint path in line 147 and inference images in line 130 and 131 in inference.py.
```
$ CUDA_VISIBLE_DEVICES=num_gpu python inference.py
```