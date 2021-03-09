# Use Xception For Face Anti-spoofing

* This repository contains some scripts to train [Xception](https://arxiv.org/pdf/1610.02357.pdf) introduced by François Chollet, the founder of Keras.
* Also support other classification projects (via Xception)

## Environments

I have tested the scripts in the following environment.

* Microsoft Windows 10 Pro Insider Preview
* RTX3090 (24GB)
* Nvidia Driver Version: 465.42
* CUDA Version: 11.3  
* python 3.6.12
* numpy 1.19.5
* scipy 1.5.4
* h5py 3.1.0
* Keras 2.4.3
* tf-nightly 2.5.0.dev20210131 (for RTX 30 Series)
* tensorflow-gpu 1.15.0 (for other Nvidia GPU)
* Flask 1.1.2

## Training Dataset

### Dataset Structure

Make sure your dataset structure is like following structure.
```bash
Dataset-AntiDF
├─Fake
└─Real
```

### Several related datasets or spoofing methods

```
DFFD: Diverse Fake Face Dataset (Contains most of the pictures of the following data sets)
Hao Dang, A. (2020). On the Detection of Digital Face Manipulation. In In Proceeding of IEEE Computer Vision and Pattern Recognition (CVPR 2020).

Large-scale CelebFaces Attributes (CelebA) Dataset 
Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep Learning Face Attributes in the Wild. In Proceedings of International Conference on Computer Vision (ICCV).

FaceForensics++
Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, & Matthias Nie\ssner (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. In ICCV 2019.

PGGN
Tero Karras, Timo Aila, Samuli Laine, & Jaakko Lehtinen (2017). Progressive Growing of GANs for Improved Quality, Stability, and Variation. CoRR.

StarGAN
Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, & Jaegul Choo (2018). StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

StyleGAN
Tero Karras, Samuli Laine, & Timo Aila (2018). A Style-Based Generator Architecture for Generative Adversarial Networks. CoRR.
```

## Label File: classes.txt

Create a text file where all the class names are listed line by line. This can be easily done with the below command.

```bash
ls Dataset-AntiDF > classes.txt
```

## Train on the model

* `<dataset_root>`: Path to the directory where all the training data stored. **required**
* `<classes>`: Path to a txt file where all the class names are listed line by line. **required**
* `<result_root>`: Path to the directory where all the result data will be saved. **required**
* `[epochs_pre]`: The number of epochs during the first training stage (default: 5).
* `[epochs_fine]`: The number of epochs during the second training stage (default: 50).
* `[batch_size_pre]`: Batch size during the first training stage (default: 32).
* `[batch_size_fine]`: Batch size during the second training stage (default: 16).
* `[lr_pre]`: Learning rate during the first training stage (default:1e-3).
* `[lr_fine]`: Learning rate during the second training stage (default:1e-4).
* `[snapshot_period_pre]`: Snapshot period during the first training stage (default:1). At the every spedified epochs, a serialized model file will be saved under <result_root>.
* `[snapshot_period_fine]`: Snapshot period during the second training stage (default:1).

### command

```bash
python fine_tune.py <dataset_root> <classes> <result_root> [epochs_pre] [epochs_fine] [batch_size_pre] [batch_size_fine] [lr_pre] [lr_fine] [snapshot_period_pre] [snapshot_period_fine]
python fine_tune.py D:\Dataset-AntiDF classes.txt result-balanced-4w_5_50_180_16_1e-3_1e-4_2/ --epochs_pre 5 --epochs_fine 50 --batch_size_pre 180 --batch_size_fine 16 --lr_pre 1e-3 --lr_fine 1e-4
```

### `fine_tune.py`

* Xception's weights are initialized with the ones pre-trained on the ImageNet dataset (officialy provided by the keras team).
* In the first training stage, only the top classifier of the model is trained for 5 epochs.
* In the second training stage, the whole model is trained for 50 epochs with a lower learning rate.
* All the result data (serialized model files and figures) are to be saved under `result/`


## Inference via the model

* `<model>`: Path to a serialized model file. **required**
* `<classes>`: Path to a txt file where all the class names are listed line by line. **required**
* `<image>`: Path to an image file that you would like to classify. **required**

### command

```bash
python inference.py <model> <classes> <image>
python inference.py result-balanced-4w_5_50_180_16_1e-3_1e-4_1/model_fine_final.h5 classes.txt images/faceapp/F_FAP1_00334-2.png
```
### result

```bash
......
2021-03-08 22:50:04.006047: I tensorflow/stream_executor/cuda/cuda_blas.cc:1838] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
Top 1 ====================
Class name: Real
Probability: 100.00%
Top 2 ====================
Class name: Fake
Probability: 0.00%
```

##  inference service