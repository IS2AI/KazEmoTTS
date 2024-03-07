import kaldiio
import os
import librosa
from tqdm import tqdm
import glob
import json 
from shutil import copyfile
import pandas as pd
import argparse
from text import _clean_text, symbols
from num2words import num2words
import re
from melspec import mel_spectrogram
import torchaudio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, help='path to the emotional dataset')
    args = parser.parse_args()
    dataset_path = args.data
    filelists_path = 'filelists/all_spks/'
    feats_scp_file = filelists_path + 'feats.scp'
    feats_ark_file = filelists_path + 'feats.ark'


    spks = ['1263201035', '805570882', '399172782']
    train_files = []
    eval_files = []
    for spk in spks:
        train_files += glob.glob(dataset_path + spk + "/train/*.wav")
        eval_files += glob.glob(dataset_path + spk + "/eval/*.wav")

    os.makedirs(filelists_path, exist_ok=True)

    with open(filelists_path + 'train_utts.txt', 'w', encoding='utf-8') as f:
        for wav_path in train_files:
            wav_name = os.path.splitext(os.path.basename(wav_path))[0]
            f.write(wav_name + '\n')
    with open(filelists_path + 'eval_utts.txt', 'w', encoding='utf-8') as f:
        for wav_path in eval_files:
            wav_name = os.path.splitext(os.path.basename(wav_path))[0]
            f.write(wav_name + '\n')

    with open(feats_scp_file, 'w') as feats_scp, \
        kaldiio.WriteHelper(f'ark,scp:{feats_ark_file},{feats_scp_file}') as writer:
        for root, dirs, files in os.walk(dataset_path):
            for file in tqdm(files):
                if file.endswith('.wav'):
                    # Get the file name and relative path to the root folder
                    wav_path = os.path.join(root, file)
                    rel_path = os.path.relpath(wav_path, dataset_path)
                    wav_name = os.path.splitext(os.path.basename(wav_path))[0]
                    signal, rate = torchaudio.load(wav_path)
                    spec = mel_spectrogram(signal, 1024, 80, 22050, 256,
                              1024, 0, 8000, center=False).squeeze()
                    # Write the features to feats.ark and feats.scp
                    writer[wav_name] = spec
    

    emotions = [os.path.basename(x).split("_")[1] for x in glob.glob(dataset_path + '/**/**/*')]
    emotions = sorted(set(emotions))

    utt2spk = {}
    utt2emo = {}
    wavs = glob.glob(dataset_path + '**/**/*.wav')
    for wav_path in tqdm(wavs):
        wav_name = os.path.splitext(os.path.basename(wav_path))[0]
        emotion =  emotions.index(wav_name.split("_")[1])
        if wav_path.split('/')[-3] == '1263201035':
            spk = 0 ## labels should start with 0
        elif wav_path.split('/')[-3] == '805570882':
            spk = 1
        else:
            spk = 2
        utt2spk[wav_name] = str(spk)
        utt2emo[wav_name] = str(emotion)
    utt2spk = dict(sorted(utt2spk.items()))
    utt2emo = dict(sorted(utt2emo.items()))

    with open(filelists_path + 'utt2emo.json', 'w') as fp:
        json.dump(utt2emo, fp,  indent=4)
    with open(filelists_path + 'utt2spk.json', 'w') as fp:
        json.dump(utt2spk, fp,  indent=4) 
    
    txt_files = sorted(glob.glob(dataset_path + '/**/**/*.txt'))
    count = 0
    txt = []
    basenames = []
    utt2text = {}
    flag = False
    with open(filelists_path + 'text', 'w', encoding='utf-8') as write:
        for txt_path in txt_files:
            basename = os.path.basename(txt_path).replace('.txt', '')
            with open(txt_path, 'r', encoding='utf-8') as f:
                txt.append(_clean_text(f.read().strip("\n"), cleaner_names=["kazakh_cleaners"]).replace("'", ""))
                basenames.append(basename) 
    output_string = [re.sub('(\d+)', lambda m: num2words(m.group(), lang='kz'), sentence) for sentence in txt]
    cleaned_txt = []
    for t in output_string:
        cleaned_txt.append(''.join([s for s in t if s in symbols]))               
    utt2text = {basenames[i]: cleaned_txt[i] for i in range(len(cleaned_txt))}
    utt2text = dict(sorted(utt2text.items()))

    vocab = set()
    with open(filelists_path + '/text', 'w', encoding='utf-8') as f:
        for x, y in utt2text.items():
            for c in y: vocab.add(c)
            f.write(x + ' ' +  y + '\n')
