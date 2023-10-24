# <center>Source-free Domain Adaptation via Avatar Prototype Generation and Adaptation</center>
This repository provides the official implementation for "**Source-free Domain Adaptation via Avatar Prototype Generation and Adaptation**". (IJCAI2021)

# Paper
[Source-free Domain Adaptation via Avatar Prototype Generation and Adaptation](https://arxiv.org/abs/2106.15326)
![CPGA](./results/archi.png "An overview of CPGA")
To handle source-free domain adaptation task, we propose a Contrastive Prototype Generation and Adaptation (CPGA) method. Specifically, CPGA consists of two stage: (1) **Prototype generation**: under the guidance of the fixed classifier, a generator ![](http://latex.codecogs.com/svg.latex?G_{g}) is trained to generate avatar feature prototypes via ![](http://latex.codecogs.com/svg.latex?\mathcal{L}_{ce}) and ![](http://latex.codecogs.com/svg.latex?\mathcal{L}_{con}^{p}). (2) **Protype adaptation**: in each training batch, we use the learned prototype generator to generate one prototype for each class. Based on the generated prototypes and pseudo labels obtained by clustering, we align each pseudo-labeled target feature to the corresponding class prototype by training a domain-invariant feature extractor via ![](http://latex.codecogs.com/svg.latex?\mathcal{L}_{con}^{w}), ![](http://latex.codecogs.com/svg.latex?\mathcal{L}_{elr}) and ![](http://latex.codecogs.com/svg.latex?\mathcal{L}_{nc}). Note that the classifier ![](http://latex.codecogs.com/svg.latex?C_{y}) is fixed during the whole training phase.

# Getting Started
## Installation
- Clone this repository:
```
git clone https://github.com/SCUT-AILab/CPGA.git
cd CPGA
```

- Install the requirements by runing the following command:
```
pip install -r requirements.txt
```

## Data Preparation
- The `.pkl` files of data list and its corresponding labels have been put in the directory `./data`.

<!-- - Download the Pneumonia and COVID-19 dataset and put the data in this repo.
    - Link: [datasets](https://drive.google.com/open?id=1FcXIYJBtfvc1dN54R4cad9cuKVzS8WOb) -->
- Please manually download the [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) benchmark from the official websites and put it in the directory `./dataset`.

## Training
- First, to obtain the pre-trained model on the source domain:
```
python train_source --gpu 0 --data_root ./dataset/VISDA-C/train --label_file ./data/visda_synthesis_9_1_split.pkl
```

- Second, to train CPGA on the target domain (please assign a source-trained model path):
```
python main --gpu 0,1 --max_epoch 1400 --source_model_path ./model_source/synthesis_resnet101_best.pkl --data_path ./dataset/VISDA-C/validation --label_file ./data/visda_real_train.pkl
```


## Testing 
To test CPGA on the target domain using the trained model (please assign a trained model path)
```
python test --gpu 0 --model_path ./model_VISDA-C/best.pkl --data_path ./dataset/VISDA-C/validation --label_file ./data/visda_real_train.pkl
```
<!-- 提供模型 -->

## Results
Classification accuracies (%) on the large-scale **VisDA** dataset (ResNet-101).
  
|  Method   | Plane   |Bicycle   |Bus   |Car   |Horse   |Knife   |Mcycl   |Person   |Plant   |Sktbrd   |Train   |Truck   |Per-class | Pre-training |
|  :----:  | :----:  | :----:  |:----:  |:----:  |:----:  |:----:  |:----:  |:----:  |:----:  |:----:  |:----:  |:----:  |:----:  |:----:  |
| ResNet-101  |55.1|53.3|61.9|59.1|80.6|17.9|79.7|31.2|81.0|26.5|73.5|8.5| 52.4 |- |
| PrDA  |86.9|81.7|84.6|63.9|93.1|91.4|86.6|71.9|84.5|58.2|74.5|42.7| 76.6 |- |
| SHOT  |92.6|81.1|80.1|58.5|89.7|86.1|81.5|77.8|89.5|84.9|84.3|49.3| 79.6 |- |
| MA  |94.8|73.4|68.8|74.8|93.1|95.4|88.6|84.7|89.1|84.7|83.5|48.1| 81.6 |- |
| BAIT  |93.7|83.2|84.5|65.0|92.9|95.4|88.1|80.8|90.0|89.0|84.0|45.3| 82.7 |- |
| CPGA (ours)  |95.6|89.0|75.4|64.9|91.7|97.5|89.7|83.8|93.9|93.4|87.7|69.0| 86.0 |[Download](https://drive.google.com/file/d/1LyRdK_CxHjY8QRklG65xErl7P4Itv7hM/view?usp=sharing) |


# Visualization
We have released the code in the directory `./visualization`. 


# Citation
If you find our work useful in your research, please cite the following paper:
```
@inproceedings{Qiu2021CPGA,
  title={Source-free Domain Adaptation via Avatar Prototype Generation and Adaptation},
  author={Zhen Qiu and Yifan Zhang and Hongbin Lin and Shuaicheng Niu and Yanxia Liu and Qing Du and Mingkui Tan},
  booktitle={International Joint Conference on Artificial Intelligence},
  year={2021}
}
```

# Contact
For any question, please file an issue or contact
```
Zhen Qiu: qiuzhenscut@gmail.com
Hongbin Lin: sehongbinlin@mail.scut.edu.cn
```
