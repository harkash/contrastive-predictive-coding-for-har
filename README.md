# Contrastive Predictive Coding for Human Activity Recognition
This repo contains the Pytorch implementation of the paper: [Contrastive Predictive Coding for Human Activity Recognition](https://dl.acm.org/doi/10.1145/3463506), accepted at IMWUT 2021.

## Introduction
Feature extraction is crucial for human activity recognition (HAR) using body-worn movement sensors. Recently, learned representations have been used successfully, offering promising alternatives to manually engineered features. Our work focuses on effective use of small amounts of labeled data and the opportunistic exploitation of unlabeled data that are straightforward to collect in mobile and ubiquitous computing scenarios. We hypothesize and demonstrate that explicitly considering the temporality of sensor data at representation level plays an important role for effective HAR in challenging scenarios. We introduce the Contrastive Predictive Coding (CPC) framework to human activity recognition, which captures the temporal structure of sensor data streams. Through a range of experimental evaluations on real-life recognition tasks, we demonstrate its effectiveness for improved HAR. CPC-based pre-training is self-supervised, and the resulting learned representations can be integrated into standard activity chains. It leads to significantly improved recognition performance when only small amounts of labeled training data are available, thereby demonstrating the practical value of our approach. Through a series of experiments, we also develop guidelines to help practitioners adapt and modify the framework towards other mobile and ubiquitous computing scenarios.

## Overview
The Activity Recognition Chain (ARC) comprises of five distinct steps: data collection, pre-processing, windowing, feature extraction, and classification.
In this work, we focus on the fourth step -- which is feature extraction, and accomplish it via self-supervision.

We first split the dataset by participants to form the `train-val-test` splits. 
Of the available participants, 20% are chosen randomly for testing whereas the remaining 80% are further divided randomly into the train and validation splits at a 80:20 ratio.
The train data is normalized to have zero mean and unit variance, which are subsequently applied to the validation and test splits as well to get the pre-processed data.
The processed data for Mobiact can be accessed [~~here~~](https://gatech.box.com/s/3urr2mf4lntf57y2ef64ogpl4ijla87b) [here](https://www.dropbox.com/s/17ftg0368mwsidx/mobiact.mat?dl=0). 
Please download and add it to a new folder called `data`.  
   
The sliding window process is applied to segment one second of data with 50% overlap. 
On these unlabeled windows, we first pre-train the CPC model, and transfer the learned encoder weights to a classifier for activity recognition (or fine-tuning). 
During classification, the learned encoder weights are frozen, and only the MLP classifier is updated via the cross entropy loss utilizing ground truth labels.


## Training
In this repo, the pre-training can be performed using: `python main.py --dataset mobiact` and the trained model weights 
get saved to the `models/<DATE>` folder where `<DATE>` corresponds to the date the training completed. 
By specifying the dataset as `mobiact`, the processed data is utilized based on the location detailed in `arguments.py`. 
The pre-training logs get saved in the `saved_logs/<DATE>` folder and contain the losses as well as the accuracy of the CPC pre-training.

## Activity recognition/classification
Once the model training is complete, the saved weights can be utilized for classification by running: 
```python evaluate_with_classifier.py --dataset mobiact --saved_model <FULL PATH/TO/MODEL.pkl> ```.  
The classification logs get saved to the `saved_logs/<DATE>` folder as well, and contain the losses as well as 
the accuracy and f1-scores for the classification.

## Summary of files in the repository
| File name | Function |
|---|---|
| `arguments.py` | Contains the arguments utilized for the pre-training and classification. |
| `dataset.py` | Dataset and data loaders to load the pre-processed data and apply the sliding window procedure. |
| `evaluate_with_classifier.py` | To train the classifier on the learned encoder weights. |
| `main.py` | Over-arching file for the pre-training. |
| `meter.py` | For logging metrics and losses. |
| `model.py` | Defines the model and the CPC loss. |
| `sliding_window.py` | Efficiently performs the sliding window process on numpy. Taken from from [here](http://www.johnvinyard.com/blog/?p=268). Also utilized in the original [DeepConvLSTM](https://github.com/fjordonez/DeepConvLSTM) implementation. |
| `trainer.py` | Contains methods for the pre-training with CPC.  |
| `utils.py` | Utility functions. |

## License
This software is released under the GNU General Public License v3.0

## Repositories utilized in this project
This project is based on the CPC implementations detailed in the following repositories:
[Contrastive-Predictive-Coding-PyTorch](https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch) and
[emotion_detection_cpc](https://github.com/McHughes288/emotion_detection_cpc).
They were very useful for this project.
