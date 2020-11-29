# perceptNet
Code for paper 'PerceptNet: Learning Perceptual Similarity of Haptic Textures in Presence of Unorderable Triplets', WHC 2019
https://arxiv.org/abs/1905.03302


## Abstract 

Modeling perceptual similarity is essential for improving the sensing capability of machines and robots. We propose a deep metric learning approach where the model is trained only on non-numerical, triplet-based relative comparisons of signals. While existing methods ignore unorderable/ambiguous signals as uninformative, our study demonstrates the value of incorporating such signals into the training process, in order to better encode the nuances and limitations of human perception. The model is trained on relatively little data while generalizing gracefully to unseen signals.


## Dataset

Experiments are performed for three test cases in order of increasing difficulty -
1. Unseen triplets
2. Unseen samples
3. Unseen classes

For each case, the training and test triplets are present for five random train/test splits in the data folder.

## Requirements
The model is implemented in PyTorch. Please install other Python libraries using requirements.txt

`$pip install -r requirements.txt`

## Train Model

Specify the data directory and results directory in train.py

`python3 train.py`
