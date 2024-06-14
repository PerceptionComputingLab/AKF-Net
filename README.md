# AEPformer

AEPformer: Asynchronous Enhancement Pattern guided disentangled Transformer for pituitary gland and pituitary microadenoma segmentation in DCE-MRI

# Requirements

- python==3.9
- batchgenerators==0.25
- monai==1.3.1
- numpy 
- scikit-learn==1.5.0
- scipy==1.13.1
- SimpleITK==2.3.1
- tensorboard==2.17.0
- torch==1.12.1
- torchvision==0.13.1
- tqdm==4.66.4

You can install these packages by executing the following command:

```
pip install -r requirements.txt
```

# Dataset

This work uses a private dataset. Due to some factors, only part of the data is given here to test the operation of the code in `dataset/sample_data`.

# Training

- **Step 1.** In the `main_train.py` file, modify the statement `os.environ['CUDA_VISIBLE_DEVICES']` to select the GPU you want to use. For example, set `os.environ['CUDA_VISIBLE_DEVICES']='0'` to use the first GPU.
- **Step 2.** In the `config.py` file, modify the key named `dataset_path` to specify the data path
- **Step 3.** Set the training parameters in the `config.py` file
- **Step 4.** Execute the command to perform training
```
python main_train.py
```

# Inference

- **Step 1.** The breakpoints of the model training will be saved in the `runs` directory. Select the absolute path of the model breakpoint to be inferred and copy it to the `checkpoint_path` field in the `config.py` file.
- **Step 2.** In the `main_test.py` file, modify the statement `os.environ['CUDA_VISIBLE_DEVICES']` to select the GPU you want to use.
- **Step 3.** Execute the command to perform inference
```
python main_test.py
```

