This is implementation for the paper: Class Relationship Embedded Learning for Source-Free Unsupervised Domain
Adaptation.


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

**configs**: training configs files for different experiments

**data**: contain dataset images and labels

**runs**: automatically created which stores checkpoints, tensorboard and text logging files

**CRCo**: source code of our method, contains hooks (evaluation hooks), loaders (train and test loaders), models (
definition of models), trainers (training and testing process)

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

## Steps to reproduce the results
Here we take Office-Home A->C under the close-set source-free unsupervised Domain adaptation setting as an example.

1. Some preparations. 
   - Make a folder '/home/username/PycharmProjects', replace the username by your name under /home.
   - Download the "code.zip" here. Unzip it under '/home/username/PycharmProjects', and rename this folder by 'class-relation-learning' (needed in [sfda_source_only_train.sh](experiments%2Fsfda_source_only_train.sh) and [sfda_class_relation_train.sh](experiments%2Fsfda_class_relation_train.sh))
   - Prepare dataset following the **data** folder structure.
2. Go to the project home

```
cd /home/username/PycharmProjects/class-relation-learning
```

3. Training source domain model using the following script. Here single GPU is enough. In the following scripts,
   "my_exp" is an experiment name specified by yourself. The .py file stores the configurations for training and testing.  

```
CUDA_VISIBLE_DEVICES=0 bash ./experiments/sfda_source_only_train.sh my_exp ./configs/sfda_class_relation/sfda_officehome_source_A.py
```

After trained done, you can find the source pre-trained model in the **runs** folder. You should keep the path of model in mind for the target domain training. Or you can skip the step by using our provided [models](https://drive.google.com/drive/folders/1xNHfjZUCUKql3H26jaAuoNy7XYiS12Qv?usp=share_link).

4. Configure the source model path (absolute path) in configs/sfda_class_relation/class_relation_officehome_AaD_AC.py configuration file. Specifically, append the absolute path in "pretrained_model" key which looks as follows:   

```
control = dict(
  save_interval=500,
  max_save_num=1,
  seed=2023,
  pretrained_model =  /home/***/source_model_Art.pth,
)
```

5. Train target domain model using the following script. Since we have three views for each image (one weak and two strong views), we need **Two**
   GPUs with 11G memory for training. For larger models like ResNet101 or ViT-B, larger GPU memory is necessary.

```
CUDA_VISIBLE_DEVICES=0,1 bash ./experiments/sfda_class_relation_train.sh my_exp ./configs/sfda_class_relation/class_relation_officehome_AaD_AC.py
```

The results of all training metrics and testing accuracy is stored in the **runs** folder. You can view them from the
txt log file or through tensorboard.



