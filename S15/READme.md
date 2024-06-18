Here is a `README.md` for your project based on the uploaded PDF about YOLOv9:

---

# YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information

## Introduction

YOLOv9 is an advanced real-time object detection system combining the concepts of Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN). The project addresses critical issues in deep learning such as information bottlenecks and data loss during feature extraction in deep networks. It offers significant improvements in parameter utilization and accuracy over existing state-of-the-art models.

## Key Features

- **Programmable Gradient Information (PGI):** Enhances gradient reliability and optimizes network weight updates.
- **Generalized Efficient Layer Aggregation Network (GELAN):** Provides a flexible and lightweight architecture with superior parameter utilization.
- **Superior Performance:** Outperforms other real-time object detection models in terms of accuracy and parameter efficiency.
- **Versatile Use:** Suitable for both lightweight and large-scale models.

## Installation

To install the required dependencies, use the following commands:

```bash
# Clone the repository
git clone https://github.com/WongKinYiu/yolov9.git
cd yolov9

# Install required Python packages
pip install -r requirements.txt
```

## Usage

To train YOLOv9 on the MS COCO dataset or your custom dataset:

1. **Prepare the dataset:**
   - Download the by anotation 

2. **Train the model:**
   ```bash
   python train.py --data coco.yaml --cfg yolov9.yaml --weights yolov9.pt --batch-size 16
   ```

3. **Evaluate the model:**
   ```bash
   python test.py --data coco.yaml --weights runs/train/exp/weights/best.pt --batch-size 16
   ```

4. **Inference:**
   ```bash
   python detect.py --weights runs/train/exp/weights/best.pt --source data/images
   ```

## Architecture Overview

### Programmable Gradient Information (PGI)
- **Auxiliary Reversible Branch:** Generates reliable gradients and updates network parameters without additional inference costs.
- **Multi-Level Auxiliary Information:** Aggregates gradients from different prediction heads to retain complete information for various target tasks.

### Generalized Efficient Layer Aggregation Network (GELAN)
- **Modular Design:** Supports various computational blocks, allowing customization for different inference devices.
- **Lightweight and Fast:** Achieves high parameter efficiency and inference speed using conventional convolution operators.

## Experimental Results

YOLOv9 achieves state-of-the-art performance on the MS COCO dataset with significant improvements in accuracy and efficiency over previous models like YOLOv7 and YOLOv8. For detailed results, refer to the experimental section in the [paper](https://arxiv.org/abs/2402.13616).

| Model       | # Params | FLOPs | AP<sub>50:95</sub> | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
|-------------|----------|-------|-------------------|----------------|----------------|----------------|----------------|----------------|
| YOLOv9-S    | 7.1M     | 26.4G | 46.8%             | 63.4%          | 50.7%          | 26.6%          | 56.0%          | 64.5%          |
| YOLOv9-M    | 20.0M    | 76.3G | 51.4%             | 68.1%          | 56.1%          | 33.6%          | 57.0%          | 68.0%          |
| YOLOv9-C    | 25.3M    | 102.1G| 53.0%             | 70.2%          | 57.8%          | 36.2%          | 58.5%          | 69.3%          |
| YOLOv9-E    | 57.3M    | 189.0G| 55.6%             | 72.8%          | 60.6%          | 40.2%          | 61.0%          | 71.4%          |

## Citation

If you use this code or models in your research, please cite the original paper:

```
@article{Wang2024YOLOv9,
  title={YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information},
  author={Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao},
  journal={arXiv preprint arXiv:2402.13616},
  year={2024}
}
```

## Acknowledgements

We thank the National Center for High-performance Computing (NCHC) for providing computational and storage resources.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


   ![alt text](image3.png)

4.Traning steps

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      88/99      12.9G     0.3369     0.2965      0.929         25        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        134        191      0.884      0.928      0.946      0.895
   

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      89/99      12.9G     0.3211      0.294     0.9137         23        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        134        191      0.909      0.893      0.941      0.882
   

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      90/99      12.9G     0.3215     0.2872     0.9162         20        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        134        191      0.931      0.911      0.951      0.898
   

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      91/99      12.9G     0.3139     0.2806      0.906         22        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        134        191      0.931      0.926      0.953      0.901
   

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      92/99      12.9G     0.3083     0.2685     0.9045         22        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        134        191      0.945      0.892      0.958      0.902
   

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      93/99      12.9G     0.3084       0.27     0.9052         16        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        134        191      0.946      0.874       0.95      0.898
   

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      94/99      12.9G      0.294     0.2464     0.8971         22        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        134        191      0.952      0.871      0.942      0.891
   

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      95/99      12.9G     0.2977      0.244     0.8969         16        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        134        191      0.939      0.873       0.94      0.893
   

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      96/99      12.9G     0.2887     0.2444     0.8925         16        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        134        191      0.923      0.884      0.946      0.898
   

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      97/99      12.9G     0.2822     0.2319     0.8897         22        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        134        191      0.923      0.884      0.946      0.897
   

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      98/99      12.9G     0.2892     0.2422     0.8928         22        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        134        191      0.901      0.911      0.951      0.901
   

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      99/99      12.9G     0.2816     0.2307     0.8869         23        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        134        191      0.947       0.88      0.949      0.899
   



## Gradio App

A Gradio app was developed to provide an easy-to-use interface for testing the YOLOv9 model. The app allows users to upload images and get predictions on the presence and location of manholes.

![alt text](images/image1.png)

## Deployment on Hugging Face

The Gradio app was deployed on Hugging Face Spaces, making it accessible to a wider audience. The deployment process involved:

1. **Preparing the App**: Ensuring the Gradio app is fully functional locally.
2. **Creating a Hugging Face Space**: Setting up a new space on Hugging Face.
3. **Uploading the App**: Uploading the Gradio app and its dependencies to the Hugging Face Space.

### **[YOLOv9 Demo on Hugging Face Spaces](https://huggingface.co/spaces/sagar007/YOLOV9)**


## Usage

To use the deployed Gradio app:

1. Visit the Hugging Face Space: [Gradio Manhole Detection App](#).
2. Upload an image containing a manhole.
3. Get predictions on the presence and location of the manhole.

## Requirements

- Python 3.7+
- Gradio
- YOLOv9
- AWS SageMaker Studio (for training)

## Installation

To run the Gradio app locally:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/manhole-detection-yolov9.git
    cd manhole-detection-yolov9
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Gradio app:
    ```bash
    python app.py
    ```

## Acknowledgments

- [YOLOv9](#) for the object detection model.
- [Gradio](https://gradio.app/) for the interactive interface.
- [Hugging Face](https://huggingface.co/) for providing the deployment platform.
- [Amazon SageMaker Studio](https://aws.amazon.com/sagemaker/studio/) for model training.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
