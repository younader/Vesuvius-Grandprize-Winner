# Vesuvius Grandprize Winning Solution
![Vesuvius Challenge GP Solution](pictures/logo.png)

The Repository contains the First Place Vesuvius Grand Prize solution. 
This repository is part of the First Place Grand Prize Submission to the Vesuvius Challenge 2023 from Youssef Nader, Luke Farritor and Julian Schilliger.

<!-- <img align="center" width="60" height="60" src="pictures/ThaumatoAnakalyptor.png">  -->
## Automatic Segmentation <img align="center" width="60" height="60" src="pictures/ThaumatoAnakalyptor.png"> 
Check out the automatic segmentation pipeline ThaumatoAnakalyptor of our winning Grand Prize submission by Julian Schiliger. 
[ThaumatoAnakalyptor](https://github.com/schillij95/ThaumatoAnakalyptor/tree/main) performs in full 3D and is also capable of segmenting in very mushy and twisted scroll regions.

# Ink Detection Overview<img align="center" width="60" height="60" src="pictures/logo.png"> :
Our final canconical model was a timesformer small architecture with divided space-time attention. 
The dataset underwent expansion and cleaning rounds to increase accuracy of the labels and become as accurate as possible, approximately 15 rounds were performed between the first letters and final solution. 
Our solution also consisted of 2 other architectures, Resnet3D-101 with pretrained weights, I3D with non-local block and maxpooling. 

Our implementation uses `torch`, `torch-lightning`,the [`timesformer-pytorch`](https://github.com/lucidrains/TimeSformer-pytorch) and [`3D-ResNets-PyTorch`](https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py). 


# 🚀 Get Started

I recommend using a docker image like `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel` for your development environment. Kaggle/Colab images should work fine as well. 

To install this project, run:

```bash
pip install -r requirements
#to download the segments from the server
./download.sh
#propagates the inklabels into the respective segment folders for training
python prepare.py
```
You can find the weights of the canonical timesformer uploaded [here](https://drive.google.com/drive/folders/1rn3GMOvtJRMBHOxVhWFVSY6IVI6xUnYp?usp=sharing)
# Example Inference

To run inference of timesformer:

```bash
python inference_timesformer.py --segment_id 20231210121321 20231221180251 --segment_path $(pwd)/train_scrolls --model_path timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt
```

The optional parameter ```--out_path``` can be used to specify the output path of the predictions.