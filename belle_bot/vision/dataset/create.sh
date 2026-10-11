

export PYTHONPATH=.


#python3 belle_bot/vision/encoder/dataset/create_dataset.py \
#  --path "/run/media/belle/Houston/datasets/vision-encoder/v1/temp-%06d.tar" \
#  --config.subset "train"


#python3 belle_bot/vision/encoder/dataset/create_dataset.py \
#  --path "/run/media/belle/Houston/datasets/vision-encoder/v1/test-%06d.tar" \
#  --config.subset "test"


# run the downsampler

# delete the temp dir

# run funciton to create the dataset again but this time add in maps and downsample from frequency
# todo in future put the masks in when doing the next step or preprocessing, where we insert all other data
python3 belle_bot/vision/encoder/dataset/create_dataset.py \
  --subset "train" \
  --mask.count 10000 \
  --path "train-%06d.tar" \
  --mask.size 448 \
  --path "/run/media/belle/Houston/datasets/vision-encoder/v1/train-%06d.tar" \
  --frequency_map "frequency_map"
