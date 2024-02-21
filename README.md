# Document Scanner: Semantic Segmentation

<!-- #### Dataset and Trained Model Download Links

1. [Resized final dataset](https://www.dropbox.com/s/rk37cuwtav5j1s7/document_dataset_resized.zip?dl=1)
2. [Model - MobileNetV3-Large backend](https://www.dropbox.com/s/4znmfi5ew1u5z9y/model_mbv3_iou_mix_2C049.pth?dl=1)
3. [Model - Resnet50 backend](https://www.dropbox.com/s/kotc40uz6bhvpel/model_r50_iou_mix_2C020.pth?dl=1)

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/scl/fo/w3i08lmjnd6ba3td89a3p/h?dl=1&rlkey=unuq45366j21xctj9ovt9ehd3) -->

### Setup
- To create `environment`:
```bash
conda create -n env python=3.8
conda activate env
pip install -r requirements.txt
pip install nvidia-pyindex
pip install onnx-graphsurgeon
```
### Create dataset

- Download raw scroped dataset and background

- To gen full mask
```bash
python3 main.py --scenario generate_doc_set
```

- To resize
```bash
python3 main.py --scenario resizer \
-s DOCUMENTS/CHOSEN/images \
-d DOCUMENTS/CHOSEN/resized_images 
-x 640
```

```bash
python3 main.py --scenraio resizer \
-s DOCUMENTS/CHOSEN/masks \
-d DOCUMENTS/CHOSEN/resized_masks \
-x 640
```

- To gen data (config factor to change the ratio of image size and background size)
```bash
python3 main.py --scenario create_dataset
```

### Train
- To train:
```bash
python3 main.py --scenario train \
--backbone_model "mbv3" \
--data_dir "document_dataset_resized" \
--batch_size 32 \
--num_epoch 100
```

### Inference
- To infer:
```bash
python3 main.py --scenario infer \
--data_infer "./test_images/IMG_20220721_162757.jpg" \
--image_size 384 \
--checkpoint_path "/home/aittgp/vutt/workspace/Document-Scanner/model_repository/res50_15k.pth"
```
### Export model
- To export model to onnx
```bash
python3 export_onnx.py
```

- To export onnx to tflite
```bash
onnx2tf -i path/to/model.onnx -o saved_model
```
<!-- 
### Document Scanner Application

<img src = 'app_images/app_demo.png'> -->


### Simple Demo
- To run streamlit app:
```bash
streamlit run app.py --server.port 8080
```