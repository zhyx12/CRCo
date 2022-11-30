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
```

**basicda**: basic framework for domain adaptation tasks

**config**: training configs files for different experiments

**data**: contain dataset images and labels

**runs**: automatically created which stores checkpoints, tensorboard and text logging files

**CRCo**: source code of our method, contains hooks (evaluation hooks), loaders (train and test loaders), models (definition of models), trainers (training and testing process)

**experiments**: training scripts

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

## Training scripts

```
cd ${HOME}/PycharmProjects/class-relation-learning
CUDA_VISIBLE_DEVICES=0,1 bash ./sfda_class_relation_train.sh exp ./configs/sfda_class_relation/class_relation_officehome_AaD_AC.py
```





