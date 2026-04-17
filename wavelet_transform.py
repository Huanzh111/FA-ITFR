import pywt
import torch
import torch.nn.functional as F
import numpy as np


def apply_wavelet_transform(tensor):
    wavelet1 = 'haar'
    wavelet2 = 'sym2'
    batch_size, height, width = tensor.shape
    transformed_low_features_1 = []
    transformed_low_features_2 = []
    wavelet1 = pywt.Wavelet(wavelet1)
    wavelet2 = pywt.Wavelet(wavelet2)
    for i in range(batch_size):
        img = tensor[i].cpu().numpy()
        coeffs1 = pywt.dwt2(img, wavelet1)
        cA1, (cH1, cV1, cD1) = coeffs1
        High1 = np.stack([cH1, cV1, cD1], axis=-1)
        High1 = np.mean(High1, axis=-1)
        low_frequency1 = cA1 * 0.5 + High1 * 0.5
        transformed_low_features_1.append(low_frequency1)

        coeffs2 = pywt.dwt2(img, wavelet2)
        cA2, (cH2, cV2, cD2) = coeffs2
        High2 = np.stack([cH2, cV2], axis=-1)
        High2 = np.sum(High2, axis=-1)
        low_frequency2 = cA2 * 0.5 + High2 * 0.5
        transformed_low_features_2.append(low_frequency2)
    transformed_features_low_1 = torch.tensor(np.array(transformed_low_features_1), dtype=tensor.dtype, device=tensor.device)
    transformed_features_low_1 = F.interpolate(transformed_features_low_1.unsqueeze(1), size=(height, width), mode='bilinear', align_corners=False)
    transformed_features_low_1 = transformed_features_low_1.squeeze(1)
    transformed_features_low_2 = torch.tensor(np.array(transformed_low_features_2), dtype=tensor.dtype,
                                              device=tensor.device)
    transformed_features_low_2 = F.interpolate(transformed_features_low_2.unsqueeze(1), size=(height, width),
                                               mode='bilinear', align_corners=False)
    transformed_features_low_2 = transformed_features_low_2.squeeze(1)
    output_features = transformed_features_low_1 * 0.4 + transformed_features_low_2 * 0.6
    return output_features