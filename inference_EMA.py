import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write
import IPython.display as ipd
import glob
import torch
from pydub import AudioSegment
from torch.utils.data import DataLoader
from text import text_to_sequence, cmudict
from text.symbols import symbols
import utils_data
import re
from num2words import num2words
from kaldiio import WriteHelper
import os
from tqdm import tqdm
from text import text_to_sequence, convert_text
from model import GradTTSWithEmo
import utils_data as utils
from attrdict import AttrDict
from models import Generator as HiFiGAN


HIFIGAN_CONFIG = './configs/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'


if __name__ == '__main__':
    hps, args = utils.get_hparams_decode()
    device = torch.device('cpu' if not torch.cuda.is_available() else "cuda")
    ckpt = utils_data.latest_checkpoint_path(hps.model_dir, "EMA_grad_*.pt")
    print(ckpt)
    model = GradTTSWithEmo(**hps.model).to(device)
    logger = utils_data.get_logger(hps.model_dir, "inference.log")
    utils_data.load_checkpoint(ckpt, model, None)
    _ = model.cuda().eval()

    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    emos = sorted(["angry", "surprise", "fear", "happy", "neutral", "sad"])
    speakers = ['M1', 'F1', 'M2']

    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]

    replace_nums = []
    for i in texts:
        replace_nums.append(i.split('|', 1))

    nums2word = [re.sub('(\d+)', lambda m: num2words(m.group(), lang='kz'), sentence) for sentence in np.array(replace_nums)[:, 0]]
    # Speakers id.
    # M1 = 0
    # F1 = 1
    # M2 = 2
    text2speech = []
    for i, j in zip(nums2word, np.array(replace_nums)[:, 1]):
        text2speech.append(f'{i}|{j}')

    for i, line in enumerate(text2speech):
        emo_i = int(line.split('|')[1])
        control_spk_id = int(line.split('|')[2])
        control_emo_id = emos.index(emos[emo_i])
        text = line.split('|')[0]
        with torch.no_grad():
            ### define emotion
            emo = torch.LongTensor([control_emo_id]).to(device)
            sid = torch.LongTensor([control_spk_id]).to(device)
            text_padded, text_len = convert_text(text)
            y_enc, y_dec, attn = model.forward(text_padded, text_len, 
                                        n_timesteps=args.timesteps, 
                                        temperature=args.noise,
                                        stoc=args.stoc, spk=sid,emo=emo, length_scale=1.,
                                        classifier_free_guidance=args.guidance)
        res = y_dec.squeeze().cpu().numpy()
        x = torch.from_numpy(res).cuda().unsqueeze(0)
        y_g_hat = vocoder(x)
        audio = y_g_hat.squeeze()
        audio = audio * 32768.0
        audio = audio.detach().cpu().numpy().astype('int16')
        audio = AudioSegment(audio.data, frame_rate=22050, sample_width=2, channels=1)
        audio.export(f'{args.generated_path}/{emos[emo_i]}_{speakers[int(line.split("|")[2])]}.wav', format="wav")