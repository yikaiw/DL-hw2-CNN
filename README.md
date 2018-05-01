# CNN
TensorFlow implementation of CNN.

## Environment
* TensorFlow-gpu 1.3.0
* Python 2.7.12

## Data preparing
First, download two datasets:

* Each dataset contains 65 categories of real-world images or clip-art images.
* [Dataset download link](https://pan.baidu.com/s/10cT-PIYP2QExZGYEfS6ovw).
* Directory structure: './dset{1, 2}/train/label{0, 1, ..., 64}/xxx.jpg'.

Second, modify the direction of dataset in main.py.

Third, write the dataset to tfrecords:

```
python build_data.py
```

## Training
Quick train:

```
python main.py
```

Or continue training from a pre-trained model:

```
python main.py --pre_trained 20180117-1030
```

## Results
See training details and transfered images in TensorBoard:

```
tensorboard --logdir checkpoints/${datetime}
```
