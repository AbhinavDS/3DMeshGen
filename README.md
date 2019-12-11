# Mesh Reconstruction for Varying Topology from Single Image

Our goal is to develop a deep learning architecture that learns the topology and produces athree  dimensional  triangular  mesh,  just  froma  single  image.   In  this  work,  we  draw  parallels  between  the  three  dimensional  and  its two dimensional counterpart.  After, reducingthe problem to two dimension,  we propose adeep  learning  architecture  that  learns  to  produce multiple polygons.  Our current methodis based on graph convolutions for learning deformation  along  with  reinforcement  learningfor learning topology.  We also perform ablation studies on different reinforcement learning algorithms for our model.

<!-- [Project Report](docs/report.pdf) -->


## Requirements
* python3
* pytorch==1.3.1
* tensorboardX
* torchtestcase
* gym
* [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)


Note: Work with actor critic branch

## Dataset Creation
The following command generates the train, val and test splits for the 2D polygons dataset.
```
src/$ cd data/data_generator/
src/$ ./create_complete_data.sh
src/$ cd ../../
```

## Training
Run following command from src folder.

```
src/$ python3 main.py  --mode train --expt_res_base_dir ../../results --expt_name rl_pixel2mesh --train_dir ../../data/train/ --val_dir ../../data/val --test_dir ../../data/test/ --suffix complete --learning_rate_decay_every 10000  -n 50000 --display_every 10
```

<!-- ## Testing
```
src/$ python3 main.py  --mode eval --expt_res_base_dir ../../results --expt_name rl_pixel2mesh --train_dir ../../data/train --val_dir ../../data/val --test_dir ../../data/test --suffix complete --display_every 10
```
 -->
## Miscellaneous
* Intermediate results can be seen during training in the results directory to track the progress.