# CNN
TensorFlow implementation of CNN.

## Environment
* TensorFlow-gpu 1.3
* Python 3.5

## Data preparing
First, download two datasets:

* Each dataset contains 65 categories of real-world images or clip-art images.
* [Dataset download link](https://pan.baidu.com/s/10cT-PIYP2QExZGYEfS6ovw).
* Directory structure: './dset{1, 2}/train/label{0, 1, ..., 64}/xxx.jpg'.

Second, build new folder data/ and put the folders dset1 and dset2 into data/.

Third, write dset1 and dset2 to tfrecords:

```
python tfrecord_writer.py
```

## Training
Quick train (default for dset1):

```
python main.py
```

Or train with selected dataset (dset1 or dset2):

```
python main.py --dataset dset2
```

Or continue training from a pre-trained model:

```
python main.py --dataset dset2 --pre_trained 20180117-1030
```

## Results
See training details and transfered images in TensorBoard:

```
tensorboard --logdir checkpoints/${datetime}
```
