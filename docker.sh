sudo docker run --rm -it --init --runtime=nvidia --ipc=host --network=host --volume=$PWD:/app -e NVIDIA_VISIBLE_DEVICES=0 abhinavds/pygeo /bin/bash

# TRAIN
# python3 main.py  --mode train --expt_res_base_dir ../../results --expt_name check --train_dir data/unittest_data --val_dir data/unittest_data --suffix unittest