# AVCLNet
AVCLNet: Multimodal Multi-Speaker Tracking Network Using Audio-Visual Contrastive Learning

## Requirements
- Python 3.8, PyTorch 2.0, opencv-python, numpy, scipy, matplotlib,PyQt ,Py3nvml 


## Datasets & Data Preparation
The model is evaluated on two primary benchmarks: AV16.3 and CAV3D. Both provide synchronized, well-calibrated audio-visual streams with identity-level annotations.

* AV16.3 Dataset
Source: Available at > http://www.glat.info/ma/av16.3/.

Setup: 16-channel audio (16kHz) from two 8-element circular microphone arrays (0.8m apart) and 25Hz video (288×360).

Preparation:

Download camera parameters: cam.mat and rigid010203.mat.

Training Sets: seq18, seq19, seq35, seq40 (approx. 13,450 sample pairs).

Testing Sets: seq24, seq25, seq30, seq45.

* CAV3D Dataset (CAV3D-MOT)
Source: Available at > https://speechtek.fbk.eu/cav3d-dataset/.

Setup: 8-channel audio (96kHz) and 15Hz video (768×1024). This dataset is more challenging due to mutual occlusion, silence periods, and speakers entering/exiting the FOV.

Preparation:

Training Sets: 3 sequences from CAV3D-MOT (approx. 11,935 sample pairs).

Testing Sets: Remaining 2 sequences.

## Preprocessing Pipeline
Use the scripts in `tools/` to prepare the data:

Audio Preprocessing:

`tools/prepareAudio.py`: Synchronize audio signals.

`tools/prepare_gccphat.py`: Generate GCC-PHAT features.

Sample Collection:

`tools/prepareSample.py` & `tools/prepareAuSample.py`: Align image frames with corresponding audio samples to create pair-wise data for contrastive learning.


## Descriptions
## Training
Configure the sample paths: `models/my_dataset.py`.

Execute the training script: Bash `python train.py`

Perform tracking on the test sequences by running: Bash `python tracking/test.py`

* Audio Localization (AO):

Algorithm: `python GCF/GCF_extract_stGCF.py`

Evaluation: `python GCF/stGCF.py`

* Visual Localization (VO):
  
Algorithm: Uses a pre-trained SiamFC for feature extraction.

Evaluation: `python visualnet/VO.py`


## Citation
If you find this work useful for your research, please cite our paper:

@article{li2026avclnet,
  title={AVCLNet: Multimodal Multispeaker Tracking Network Using Audio-Visual Contrastive Learning},
  author={Li, Yihan and Li, Yidi and Xu, Zhenhuan and Guo, Hao and Liu, Mengyuan and Wan, Weiwei},
  journal={CAAI Transactions on Intelligence Technology},
  volume={11},
  number={1},
  pages={238--255},
  year={2025},
  publisher={Wiley Online Library},
  doi={10.1049/cit2.70092}
}
