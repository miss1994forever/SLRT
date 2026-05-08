# Online CSLR Framework
Code for our proposed online continuous sign language recognition framework


## Data Preparation
For datasets and keypoint inputs, please check [../README.md](../README.md).

### Segment Isolated Signs
We use a pre-trained CSLR model, TwoStream-SLR, to segment continuous sign videos into a set of isolated sign clips.
The pre-trained model checkpoints can be downloaded [here](https://github.com/FangyunWei/SLRT/blob/main/TwoStreamNetwork/docs/TwoStream-SLR.md)
After downloading, put them into the folder ``../CTC_fusion/results``
Then run
```
python gen_segment.py
```

### Sign Augmentation
Since the segmented signs are pseudo ground-truths and their boundaries may not be accurate, we further augment these segmented signs by running
```
python sign_augment.py
```
We provide processed meta data for [Phoenix-2014T](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EtgOb0-NAWBHssQdx4zKj_IB7IA4mGk4Wuz5nRx0D8h5Bg?e=GqJYSp) and [CSL-Daily](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/Eu-Q1K-DlW1ChO2JjNBWXKsBN3otZ88z_RKXN9hEr5g9iA?e=uS6gbq).

## Training
```
source /root/miniconda3/etc/profile.d/conda.sh
conda activate slrt_legacy
export OMP_NUM_THREADS=1

config_file='configs/phoenix-2014t_ISLR.yaml'
python -m torch.distributed.run --nproc_per_node 1 --master_port 29999 training.py --config=${config_file}
```
We provide model checkpoints for [Phoenix-2014T](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EidJXFxpyaNPho5SKtVHEJ8BHex8Gq62koL-RrNnqtF1PA?e=IGGpxU) and [CSL-Daily](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EhS5B3p9i3FNu5OpqFy3WyABkMMGg1VbAzMJrxjuFVOg6Q?e=c7OK0Z).

## Testing (online inference)
```
source /root/miniconda3/etc/profile.d/conda.sh
conda activate slrt_legacy
export OMP_NUM_THREADS=1

python -m torch.distributed.run --nproc_per_node 1 --master_port 29997 prediction_slide.py --config=configs/slide_phoenix-2014t.yaml --save_fea 0
```
This is the exact Online test command verified in this workspace after the layout cleanup.

The flag `--save_fea` is optional and is mainly for exporting features to boost an offline model with the well-optimized online model. Use `--save_fea 0` for the first repeatable smoke test.