Title: Frequency-Aware Inference-Time Feature Refinement for Zero-Shot Anomaly Detection
Zero-shot anomaly detection leveraging vision-language models provides a viable paradigm for privacy-sensitive scenarios. However, pretrained networks prioritize the global coherence of visual representations, which suppresses latent high- and low-frequency cues that deviate from the dominant patterns, resulting in the omission of critical anomaly-related information. In this paper, we propose Frequency-Aware Inference-Time Feature Refinement (FA-ITFR), a training-free method that strengthens the sensitivity of intermediate and output-layer features to frequency variations, restoring richer visual representations. Frequency-Aware Global-Local Interaction Attention (F-GLIA), constructed during feature extraction to capture latent multi-band Fourier patterns in the global space, propagates frequency-enhanced signals to the local semantic space via an information flow mechanism, and selectively reinforces anomaly-sensitive components, thereby enhancing anomaly perception. To enhance the fidelity of output-layer features, we propose a dual-branch framework combining Multi-Resolution Fusion (MRF) and Dual-Wavelet Refinement (DWR). MRF projects local features across multiple spatial resolutions to compensate for fine-grained structural cues attenuated in single-scale processing, while DWR employs a dual-wavelet basis to disentangle features into high-frequency detail and low-frequency structural components, utilizing cross-frequency fusion to emphasize anomaly-relevant subbands.

This repository provides code that combines the backbone network with FA-ITFR, designed for easy reproduction and evaluation without requiring large amounts of training data.
For required packages, please refer to requirements.txt.
Dataset: For dataset and environment information, please refer to AnomalyCLIP, link: https://github.com/zqhang/AnomalyCLIP 
After downloading the mvtec AD dataset, please unzip the json_file folder and execute the mvtec.py file to obtain the dataset JSON file required by the model.
Generate the dataset JSON: Take MVTec AD for example (With multiple anomaly categories)
Structure of MVTec Folder:

mvtec/
│
├── meta.json
│
├── bottle/
│   ├── ground_truth/
│   │   ├── broken_large/
│   │   │   └── 000_mask.png
|   |   |   └── ...
│   │   └── ...
│   └── test/
│       ├── broken_large/
│       │   └── 000.png
|       |   └── ...
│       └── ...
│   
└── ...


We provide a script for the data in the json_file folder. Select the script and execute it. The generated JSON file stores all the information required for FA-ITFR.
python mvtec.py

Additionally, a cache file package is required, which we have publicly released at the link https://pan.baidu.com/s/1nUXk6_CwcVwWuUjyc9Tugw. The extraction code is 0417. After downloading, please place it in the project directory and unzip it for use.

After preparing the dataset, please unzip all publicly available compressed files, with the models located in the `models` package. Taking the MVTEC AD dataset as an example, execute the file `test.py`. Inside the file, under "metrics", you can select either "image-level" or "pixel-level" to obtain the abnormal image classification results and abnormal region segmentation results, respectively. The results will be automatically stored in the `result` file.
Run FA-ITFR: python test.py
