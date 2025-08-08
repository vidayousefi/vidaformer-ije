import io
import random

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from vformer.misc.utils import quantize_image


class Attacks:
    def __init__(self, noise: float, crop: int, jpeg: bool):
        self.noise = [noise * i for i in range(1, 5)] if noise > 0 else []
        self.jpeg = [90, 70, 50] if jpeg else []
        self.crop = [crop * i for i in range(1, 5)] if crop > 0 else []

    # ============================= JPEG =======================================

    def jpeg_attack_val(self, stego, decoder, quantize):
        jpeg_decoded = dict()
        for i, quality in enumerate(self.jpeg):
            jpeg_stego = self._jpeg_compression_attack(stego, quality)
            if quantize:
                jpeg_stego = quantize_image(jpeg_stego)
            jpeg_decoded[i + 1] = decoder(jpeg_stego)
        return jpeg_decoded

    def _jpeg_compression_attack(self, tensor, quality=75):
        dev = tensor.device
        n = tensor.size(0)
        result_list = []
        for i in range(n):
            t = tensor[i]
            mn, mx = t.min().item(), t.max().item()
            if mn != mx:
                t = (t - mn) / (mx - mn)
            # Ensure tensor is in the range [0, 1]
            t = t.clamp(0, 1)

            # Convert the tensor to a PIL image
            to_pil = transforms.ToPILImage()
            image = to_pil(t.cpu())

            # Save the image to a bytes buffer with JPEG compression
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)

            # Load the image back from the buffer
            buffer.seek(0)
            jpeg_image = Image.open(buffer)

            # Convert the PIL image back to a tensor
            to_tensor = transforms.ToTensor()
            jpeg_tensor = to_tensor(jpeg_image).to(dev)
            if mx != mn:
                jpeg_tensor = mn + jpeg_tensor * (mx - mn)
            result_list.append(jpeg_tensor)

        return torch.stack(result_list)

    # =============================== NOISE ====================================

    def noise_attack_train(self, stego, decoder):
        if random.random() > 0.5:
            noise = random.choice(self.noise)
            noise_stego = self._salt_and_pepper_noise(stego, noise)
            return decoder(noise_stego)
        else:
            return decoder(stego)

    def noise_attack_val(self, stego, decoder, quantize):
        noise_decoded = dict()
        for i, noise in enumerate(self.noise):
            noise_stego = self._salt_and_pepper_noise(stego, noise)
            if quantize:
                noise_stego = quantize_image(noise_stego)
            noise_decoded[i + 1] = decoder(noise_stego)
        return noise_decoded

    # ================================== SALT AND PEPPER ==================================

    def _salt_and_pepper_noise(self, tensor, prob):
        salt_prob = prob / 2.0
        pepper_prob = prob / 2.0

        mn, mx = tensor.min().item(), tensor.max().item()
        if mn != mx:
            tensor = (tensor - mn) / (mx - mn)
        t = tensor.clone()

        salt_mask = torch.rand_like(t) < salt_prob
        salt_values = t + torch.rand_like(t) * (1.0 - t)
        t[salt_mask] = salt_values[salt_mask]

        pepper_mask = torch.rand_like(t) < pepper_prob
        pepper_values = t - torch.rand_like(t) * t
        t[pepper_mask] = pepper_values[pepper_mask]

        if mx != mn:
            t = mn + t * (mx - mn)

        return t

    # ================================= CROP ===================================

    def crop_attack_val(self, stego, decoder, quantize):
        crop_decoded = dict()
        for i, crop in enumerate(self.crop):
            crop_stego = self._crop(stego, crop)
            if quantize:
                crop_stego = quantize_image(crop_stego)
            crop_decoded[i + 1] = decoder(crop_stego)
        return crop_decoded

    def _crop(self, stego, crop):
        c = stego[:, :, crop:-crop, crop:-crop]
        return F.pad(c, (crop, crop, crop, crop), mode="constant", value=0)
