# Official repoitory for paper: VidaFormer: A hybrid transformer for image steganography

# Abstract:
Image steganography has advanced significantly with the integration of deep learning, particularly through the use of convolutional neural networks (CNNs) for extracting complex features. While CNNs have long been a cornerstone of image analysis due to their efficiency in capturing local dependencies, the emergence of transformer models has revolutionized the field by achieving superior accuracy and effectively capturing global relationships within data. Transformers encounter issues with computational expense and memory consumption, especially when handling high-resolution images. To address these limitations, we propose a hybrid architecture that uses convolutional layers in high-resolution stages and transformer blocks in low-resolution stages. This design balances efficiency and performance by leveraging convolutional layers to capture short-range dependencies and transformers to model long-range relationships. Additionally, by replacing standard convolutions with CoordConv layers, we enhance the model's spatial awareness and feature localization capabilities. We introduce VidaFormer, an innovative steganography framework that integrates deep learning with these architectural enhancements to embed arbitrary binary data into images while maintaining high visual quality in the resulting stego images. VidaFormer outperforms state-of-the-art models, achieving an effective capacity of 4.89 bits-per-pixel on the DIV2K dataset, a 1.7% improvement over the previous best model. This demonstrates VidaFormer's superior performance and scalability for modern image steganography applications.

## Citation

If you find this repository or the VidaFormer model useful in your research, please consider citing our work:

**Vida Yousefi Ramandi, Mansoor Fateh, Mohsen Rezvani.** "VidaFormer: A hybrid transformer for image steganography."

### BibTeX
```bibtex
@misc{yousefi2025vidaformer,
  title = {VidaFormer: A hybrid transformer for image steganography},
  {Yousefi Ramandi, Vida and Fateh, Mansoor and Rezvani, Mohsen},
  year = {2025},
  note = {Code available at \url{https://github.com/vidayousefi/vidaformer-ije}}
}
