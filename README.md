This is implementation for the reviewd paper: Class Relationship Embedded Learning for Source-Free Unsupervised Domain Adaptation, paper ID: 979.


## Project structure

The project structure is presented as follows

```
├──basicda  
├──configs
| ├──_base_
| ├────cls_datasets
| ├────cls_models
| ├──sfda_class_relation
├──data
├──runs
├──CRCo
| ├──hooks
| ├──loaders
| ├──models
| ├──trainers
├──experiments
| ├──sfda_class_relation_train.sh
| ├──sfda_source_only_train.sh
├──train.py
```

**basicda**: basic framework for domain adaptation tasks

**config**: training configs files for different experiments

**data**: contain dataset images and labels

**runs**: automatically created which stores checkpoints, tensorboard and text logging files

**CRCo**: source code of our method, contains hooks (evaluation hooks), loaders (train and test loaders), models (definition of models), trainers (training and testing process)

**experiments**: training scripts

**train.py**: entrance to training process

Below are the structure under **data**.

```
│officehome/
├──Art/
│  ├── Alarm_Clock
│  │   ├── 00001.jpg
│  │   ├── 00002.jpg
│  │   ├── ......
│  ├── Backpack
│  │   ├── 00001.jpg
│  │   ├── 00002.jpg
│  │   ├── ......
│  ├── ......
├──Clipart/
│  ├── Alarm_Clock
│  │   ├── 00001.jpg
│  │   ├── 00002.jpg
│  │   ├── ......
│  ├── Backpack
│  │   ├── 00001.jpg
│  │   ├── 00002.jpg
│  │   ├── ......
│  ├── ......
│txt/
├──officehome/
│  ├── labeled_source_images_Art.txt
```

## Core files

1. Model definition:  

   ./CRCo/models/sfda_simplified_contrastive_model.py

2. Training process: 

   ./CRCo/trainers/trainer_sfda_class_relation.py


## 
## Training and testing scripts

1. prepare dataset following the **data** folder structure
2. go to the project home
```
cd ${HOME}/PycharmProjects/class-relation-learning
```
3. training source domain model
```
CUDA_VISIBLE_DEVICES=0 bash ./experiments/sfda_source_only_train.sh my_exp ./configs/sfda_class_relation/class_relation_officehome_AaD_AC.py
```
You can skip the step by using our provided models in this link[https://drive.google.com/drive/folders/1xNHfjZUCUKql3H26jaAuoNy7XYiS12Qv?usp=share_link],

4. train target domain model
```
CUDA_VISIBLE_DEVICES=0,1 bash ./experiments/sfda_class_relation_train.sh my_exp ./configs/sfda_class_relation/class_relation_officehome_AaD_AC.py
```
You can view different metrics and testing results during training process in the txt file or through tensorboard.




