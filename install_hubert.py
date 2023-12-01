 # SETUP
large_quant_model = False  # Use the larger pretrained model
device = 'cuda'  # 'cuda', 'cpu', 'cuda:0', 0, -1, torch.device('cuda')

import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from bark_hubert_quantizer.hubert_manager import HuBERTManager
from bark_hubert_quantizer.pre_kmeans_hubert import CustomHubert
from bark_hubert_quantizer.customtokenizer import CustomTokenizer

model = ('quantifier_V1_hubert_base_ls960_23.pth', 'tokenizer_large.pth') if large_quant_model else (
    'quantifier_hubert_base_ls960_14.pth', 'tokenizer.pth')

print('Loading HuBERT...')
hubert_model = CustomHubert(
    HuBERTManager.make_sure_hubert_installed(), device=device)
print('Loading Quantizer...')
quant_model = CustomTokenizer.load_from_checkpoint(
    HuBERTManager.make_sure_tokenizer_installed(model=model[0], local_file=model[1]), device)
print('Loading Encodec...')
encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(6.0)
encodec_model.to(device)

print('Downloaded and loaded models!')