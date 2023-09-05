# PVDD
This is an official implementation of the paper "Towards Real-World Video Denosing: A Practical Video Denosing Dataset and Network". [[PDF](https://arxiv.org/pdf/2207.01356.pdf)]

## Code

### Dependencies
* Python 3.6
* PyTorch >= 1.1.0
* numpy
* cv2
* skimage
* DCNv2
* easydict
* yaml
### Quick Start
Clone this github repo.
```bash
git clone https://github.com/Marinyyt/PVDD.git
cd PVDD
```

#### Training
1. Download [PVDD](## Dataset)|[CRVD](https://mega.nz/file/Hx8TgLQY#0MoZSqdrQ_HgIc4OP6_jmwAwupNctPc7ZilXLV_FAQ0)|[DAVIS](https://pan.baidu.com/s/1P6-ei5mKIKxEo1z4YRLWIQ?pwd=ss56) dataset and unpack them to any place you want.
2. Run ```train.py``` using the corresponding yaml files. (Please change the ```data_path``` argument in yaml files and noise-level file path in Dataset class.)
```bash
# PVDD sRGB 
python train.py --config /USER_PATH/PVDD/configs/PVDD_pvdd0815_02_charbo_bs1_pvdd_model.yaml --save_path /USER_SAVE_PATH
python train.py --config /USER_PATH/PVDD/configs/PVDD_pvdd0815_02_level_charbo_bs1_pvdd_model.yaml --save_path /USER_SAVE_PATH

# PVDD RAW
python train.py --config /USER_PATH/PVDD/configs/PVDD_pvdd0815_02_charbo_bs1_pvdd_raw_model.yaml --save_path /USER_SAVE_PATH
python train.py --config /USER_PATH/PVDD/configs/PVDD_pvdd0815_02_level_charbo_bs1_pvdd_raw_model.yaml --save_path /USER_SAVE_PATH

# CRVD sRGB
python train.py --config /USER_PATH/PVDD/configs/PVDD_pvdd0815_charbo_bs1_crvd_model.yaml --save_path /USER_SAVE_PATH

# DAVIS sRGB
python train.py --config /USER_PATH/PVDD/configs/PVDD_pvdd0815_charbo_bs1_davis_model.yaml --save_path /USER_SAVE_PATH
```
3. You can find the results and logs in ```save_path```.


#### Testing
1. Download our pre-trained models and unpack them to any place you want or use your pre-trained models.
3. Run.
```bash
# PVDD
python test_video_pvdd_server.py --model_file /USER_MODEL_CKPT_PATH --save_path /USER_SAVE_PATH --test_path /USER_TEST_DATA_PATH --num_frame 5
python test_video_pvdd_level_server.py --model_file /USER_MODEL_CKPT_PATH --save_path /USER_SAVE_PATH --test_path /USER_TEST_DATA_PATH --num_frame 5

python test_video_pvdd_raw_server.py --model_file /USER_MODEL_CKPT_PATH --save_path /USER_SAVE_PATH --test_path /USER_TEST_DATA_PATH --num_frame 5
python test_video_pvdd_level_raw_server.py --model_file /USER_MODEL_CKPT_PATH --save_path /USER_SAVE_PATH --test_path /USER_TEST_DATA_PATH --num_frame 5

# DAVIS
python test_video_davis_server.py --model_file /USER_MODEL_CKPT_PATH --save_path /USER_SAVE_PATH --test_path /USER_TEST_DATA_PATH --num_frame 5

# CRVD
python test_video_crvd_server.py --model_file /USER_MODEL_CKPT_PATH --save_path /USER_SAVE_PATH --test_path /USER_TEST_DATA_PATH --num_frame 5
```


### pre-trained models
[Google Drive](https://drive.google.com/drive/folders/1qEmupCR4JcaPNky3B5ldRN88t8K6CGaG) | [Baidu Drive](https://pan.baidu.com/s/1lO4OKMWBWqd4DrZG1QRWNw?pwd=s9bq)
## Dataset
Please download PVDD from Google Drive or Baidu Drive.
|     | sRGB  | raw  |
|  ----  | ----  | ---- |
| Training Dataset  | [Google Drive](https://drive.google.com/drive/folders/1rMbZqd84S1Py6buhNH6suPDnyFJjITLe?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1qiX52NPDixHwLyPKzFUHXQ?pwd=a5nt)| [Google Drive](https://drive.google.com/drive/folders/1oT68UZwR9pByINBZam_1NrciFVwdhtj8?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1qiX52NPDixHwLyPKzFUHXQ?pwd=a5nt) |
| Testing Dataset  | [Google Drive](https://drive.google.com/drive/folders/1TRSlPo1CiBPunJVC1NQmV5oLcLLo0laU?usp=sharing), [Baiidu Drive](https://pan.baidu.com/s/1W_K6odlhCHtm8zK0eZ-25g?pwd=pid1) | [Google Drive](https://drive.google.com/drive/folders/1n1wdKLIUfRNoykEPT6A6X-NsIJnkF76i?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1W_K6odlhCHtm8zK0eZ-25g?pwd=pid1) |

## Citation
If you find our work useful in your research or publication, please cite:
```
@article{xu2022pvdd,
  title={Pvdd: A practical video denoising dataset with real-world dynamic scenes},
  author={Xu, Xiaogang and Yu, Yitong and Jiang, Nianjuan and Lu, Jiangbo and Yu, Bei and Jia, Jiaya},
  journal={arXiv preprint arXiv:2207.01356},
  year={2022}
}
```
