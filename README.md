# Diabetic Retinopathy Detection

Diabetic retinopathy (DR) is an illness which causes damages to retina.
The goal of this project is to build a generic framework to classify nonreferable (NRDR) and referable (RDR) diabetic retinopathy based on fundus images with deep learning algorithms.

## Table of contents
* Technologies
* Dataset
* Usage
* Authors

## Technologies
Project is created with:
- Python 3.6
- Tensorflow 2.0.0

## Dataset
Indian Diabetic Retinopathy Image Dataset (IDRID)

## Usage
Manipulate confin.gin to switch from different modes. And use```main.py``` to start the program.

### Tuning Mode
Under this mode, no checkpoint will be saved and nothing will be visualized. And grid search will be executed.
```
main.tuning = True
main.hparams = {'HP_BS': 8,
                'HP_LR': 1e-3} # these hyperparameters will not be executed.
main.num_epoch = 1 #the epochs you want to run#
```

### Non-tuning Mode
Under this mode, you will have control of the hyperparameter for a single run. Checkpoints will be saved for every 5 epochs. If there is previous checkpoint, it'll be restored automatically. A random image in test set will be visualized by Grad-CAM.
```
main.tuning = False
main.hparams = {'HP_BS': 8,
                'HP_LR': 1e-3} #batch size and learning rate you want to use#
main.num_epoch = 1 #epochs you want to run#
```

## Authors
- Cao Shijia, scarlettcao95@gmail.com
- Zhong Liangyu, @LEGO999, zhonglyu00@gmail.com
