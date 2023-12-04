# HAFE

It's the code for the paper [HAFE: A Hierarchical Awareness and Feature Enhancement Network for Scene Text Recognition](https://doi.org/10.1016/j.knosys.2023.111178), Knowledge-Based Systems 2024.
### Install the enviroment
```bash
pip install -r requirements.txt
```
Please convert your own dataset to **LMDB** format by create_dataset.py. (Borrowed from https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py, provided by [Baoguang Shi](https://github.com/bgshih))

There are converted [Synth90K](http://www.robots.ox.ac.uk/~vgg/data/text/) and [SynthText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/) LMDB dataset by [luyang-NWPU](https://github.com/luyang-NWPU): [[Here]](https://pan.baidu.com/s/1C42j5EoDy1fTtDE8gwwndw),  password: tw3x


### Training
```bash
sh ./train.sh
```

### Testing
```bash
sh ./val.sh
```

### Recognize a image
```bash
python  pre_img.py  YOUR/MODEL/PATH  YOUR/IMAGE/PATH
```

### Citation
```
@article{Ethereal2023HAFE,
  title={HAFE: A Hierarchical Awareness and Feature Enhancement Network for Scene Text Recognition},
  author={Kai HE, Jinlong TANG, Zikang LIU , Ziqi YANG},
  journal={Knowledge-Based Systems},
  year={2024},
  publisher={Elsevier}
}
```
### Acknowledgment
This code is based on [HGA-STR](https://github.com/luyang-NWPU/HGA-STR) by [luyang-NWPU](https://github.com/luyang-NWPU). Thanks for your contribution.
