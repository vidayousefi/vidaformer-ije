# VidaFormer

Official repository for paper: VidaFormer: A hybrid transformer for image steganography

## Abstract:
Image steganography has advanced significantly with the integration of deep learning, particularly through the use of convolutional neural networks (CNNs) for extracting complex features. While CNNs have long been a cornerstone of image analysis due to their efficiency in capturing local dependencies, the emergence of transformer models has revolutionized the field by achieving superior accuracy and effectively capturing global relationships within data. However, transformers face challenges related to computational cost and memory usage, particularly when processing high-resolution images. To address these limitations, we propose a hybrid architecture that uses convolutional layers in high-resolution stages and transformer blocks in low-resolution stages. This design balances efficiency and performance by leveraging convolutional layers to capture short-range dependencies and transformers to model long-range relationships. Additionally, by replacing standard convolutions with CoordConv layers, we enhance the model's spatial awareness and feature localization capabilities. We introduce VidaFormer, an innovative steganography framework that integrates deep learning with these architectural enhancements to embed arbitrary binary data into images while maintaining high visual quality in the resulting stego images. VidaFormer outperforms state-of-the-art models, achieving an effective capacity of 4.89 bits-per-pixel on the DIV2K dataset, a 1.7% improvement over the previous best model. This demonstrates VidaFormer's superior performance and scalability for modern image steganography applications

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/mrtkhosravi/VidaFormer.git
    cd VidaFormer
    ```
2. Install dependencies:
    ```bash
    conda create -n vidaformer python==3.11 pip
    conda activate vidaformer
    pip install -r requirements.txt
    ```
3. Training:
    ```bash
    python train.py --train_data="PATH_TO_TRAIN_FOLDER" --val_data="PATH_TO_VAL_FOLDER"
    ```
4. Additional Parrameters:

    "--epochs"          default=80      
    "--data_depth"      default=6       
    "--resolution"      default=256     
    "--lr"              default=1e-4    
    "--decay"           default=5e-6    
    "--dropout"         default=0.03    
    "--mse"             default=1e-3    
    "--coordconv"       default=True
    "--transformer"     default=True
    "--base_channels"   default=48


## License
This project is licensed under the [MIT License].

```
MIT License

Copyright (c) 2024 [Your Name or Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
