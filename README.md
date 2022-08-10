# Distortion_newarch
new architecture


## Data
The data directory has the following structure:
```
<data_root>/
  train/
    train_class/
      train_image1.jpg
      train_image2.jpg
      ...
  val/
    val_class/
      val_image1.jpg
      val_image2.jpg
      ...
```

```train_class``` and ```val_class``` folders are so that we can use the standard torchvision data loaders without change.

## Running Command 
train from scratch (not pretraining)
```
python train.py  --batch_size 32 --dataset /flickr --pretrain_iter 0 --necst_iter 10000
```
train from scratch (pretraining)
```
python train.py  --batch_size 32 --dataset /flickr --pretrain_iter 5000
```
load pretrained checkpoint
```
python train.py  --batch_size 32 --dataset /flickr --resume_pretrain /checkpoint/pretrain500.pyt
```
load trained checkpoint
```
python train.py  --batch_size 32 --dataset /flickr --resume /checkpoint/train500.pyt (auto skipping necst pretraining)
```

