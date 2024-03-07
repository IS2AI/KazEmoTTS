<h1 align="center">KazEmoTTS <br> ⌨️ 😐 😠 🙂 😞 😱 😮 🗣</h1>

<p align="center">
  <a href="https://github.com/IS2AI/Kazakh_Emotional_TTS/stargazers">
    <img src="https://img.shields.io/github/stars/IS2AI/Kazakh_Emotional_TTS.svg?colorA=orange&colorB=orange&logo=github"
         alt="GitHub stars">
  </a>
  <a href="https://github.com/IS2AI/Kazakh_Emotional_TTS/issues">
    <img src="https://img.shields.io/github/issues/IS2AI/Kazakh_Emotional_TTS.svg"
         alt="GitHub issues">
  </a>
  <a href="https://issai.nu.edu.kz">
    <img src="https://img.shields.io/static/v1?label=ISSAI&amp;message=official site&amp;color=blue&amp"
         alt="ISSAI Official Website">
  </a> 
</p>

<p align = "center">This repository provides a <a href="some_cloud_link">dataset</a> and a text-to-speech (TTS) model for the paper <br><a href = "link_to_be_added"><b>KazEmoTTS:
A Dataset for Kazakh Emotional Text-to-Speech Synthesis</b></a></p>

You are able to listen for samples here: [Demo_link](https://anonimous4849.github.io)

<a name = "stats"><h2>Dataset Statistics 📊</h2></a>

<table align = "center">
<thead align = "center">
  <tr>
    <th rowspan="3">Emotion</th>
    <th rowspan="3"># recordings</th>
    <th colspan="4">Narrator F1</th>
    <th colspan="4">Narrator M1</th>
    <th colspan="4">Narrator M2</th>
  </tr>
  <tr></tr>
  <tr>
    <th>Total (h)</th>
    <th>Mean (s)</th>
    <th>Min (s)</th>
    <th>Max (s)</th>
    <th>Total (h)</th>
    <th>Mean (s)</th>
    <th>Min (s)</th>
    <th>Max (s)</th>
    <th>Total (h)</th>
    <th>Mean (s)</th>
    <th>Min (s)</th>
    <th>Max (s)</th>
  </tr>
</thead>
<tbody align = "center">
  <tr>
    <td>neutral</td>
    <td>9,385</td>
    <td>5.85</td>
    <td>5.03</td>
    <td>1.03</td>
    <td>15.51</td>
    <td>4.54</td>
    <td>4.77</td>
    <td>0.84</td>
    <td>16.18</td>
    <td>2.30</td>
    <td>4.69</td>
    <td>1.02</td>
    <td>15.81</td>
  </tr>
 <tr></tr>
  <tr>
    <td>angry</td>
    <td>9,059</td>
    <td>5.44</td>
    <td>4.78</td>
    <td>1.11</td>
    <td>14.09</td>
    <td>4.27</td>
    <td>4.75</td>
    <td>0.93</td>
    <td>17.03</td>
    <td>2.31</td>
    <td>4.81</td>
    <td>1.02</td>
    <td>15.67</td>
  </tr>
  <tr></tr>
  <tr>
    <td>happy</td>
    <td>9,059</td>
    <td>5.77</td>
    <td>5.09</td>
    <td>1.07</td>
    <td>15.33</td>
    <td>4.43</td>
    <td>4.85</td>
    <td>0.98</td>
    <td>15.56</td>
    <td>2.23</td>
    <td>4.74</td>
    <td>1.09</td>
    <td>15.25</td>
  </tr>
  <tr></tr>
  <tr>
    <td>sad</td>
    <td>8,980</td>
    <td>5.60</td>
    <td>5.04</td>
    <td>1.11</td>
    <td>15.21</td>
    <td>4.62</td>
    <td>5.13</td>
    <td>0.72</td>
    <td>18.00</td>
    <td>2.65</td>
    <td>5.52</td>
    <td>1.16</td>
    <td>18.16</td>
  </tr>
  <tr></tr>
  <tr>
    <td>scared</td>
    <td>9,098</td>
    <td>5.66</td>
    <td>4.96</td>
    <td>1.00</td>
    <td>15.67</td>
    <td>4.13</td>
    <td>4.51</td>
    <td>0.65</td>
    <td>16.11</td>
    <td>2.34</td>
    <td>4.96</td>
    <td>1.07</td>
    <td>14.49</td>
  </tr>
  <tr></tr>
  <tr>
    <td>surprised</td>
    <td>9,179</td>
    <td>5.91</td>
    <td>5.09</td>
    <td>1.09</td>
    <td>14.56</td>
    <td>4.52</td>
    <td>4.92</td>
    <td>0.81</td>
    <td>17.67</td>
    <td>2.28</td>
    <td>4.87</td>
    <td>1.04</td>
    <td>15.81</td>
  </tr>
</tbody>
</table>

<table align = "center">
<thead align = "center">
  <tr>
    <th>Narrator</th>
    <th># recordings</th>
    <th>Duration (h)</th>
  </tr>
</thead>
<tbody align = "center">
  <tr>
    <td>F1</td>
    <td>24,656</td>
    <td>34.23</td>
  </tr>
  <tr></tr>
  <tr>
    <td>M1</td>
    <td>19,802</td>
    <td>26.51</td>
  </tr>
  <tr></tr>
  <tr>
    <td>M2</td>
    <td>10,302</td>
    <td>14.11</td>
  </tr>
  <tr></tr>
  <tr>
    <td><b>Total</b></td>
    <td><b>54,760</b></td>
    <td><b>74.85</b></td>
  </tr>
</tbody>
</table>


## Installation 🛠️

First, you need to build `monotonic_align` code:

```bash
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

**Note**:Python version is 3.9.13

## Pre-processing data for training 🧹

You need to download [KazEmo](https://drive.google.com/file/d/1jHzzqS58Te8xR1VqBl4dcpOCitsESi62/view?usp=share_link) corpus and customize it as in `filelists/all_spk` by running `data_preparation.py`:
```shell
python data_preparation.py -d provide a directory of the KazEmo corpus
```

## Training stage 🏋️‍♂️
To start the training, you need to provide a path to the model configurations `configs/train_grad.json` and a directory for checkpoints `logs/train_logs` to specify your GPU.

```shell
CUDA_VISIBLE_DEVICES=YOUR_GPU_ID
python train_EMA.py -c <configs/train_grad.json> -m <checkpoint>
```

## Inference 🧠

### Pre-training stage 🏃
If you want to use pre-trained model you need to download [checkpoints](https://drive.google.com/file/d/1yfIOoVZEiFflh9494Ul6bUmktYvdM7XM/view?usp=share_link) for TTS model and vocoder.

To run inference you need:

Create a text file with sentences you want to synthesize like `filelists/inference_generated.txt`.

Specify `txt` file as follows: `text|emotion id|speaker id`.

Change path to the HiFi-Gan checkpoint in `inference_EMA.py`.

Apply classifier guidance level to 100 `-g`.

```shell
python inference_EMA.py -c <config> -m <checkpoint> -t <number-of-timesteps> -g <guidance-level> -f <path-for-text> -r <path-to-save-audios>
```

## Citation 🎓

```bibtex
@misc{abilbekov2024kazemotts,
      title={{KazEmoTTS: A Dataset for Kazakh Emotional Text-to-Speech Synthesis}}, 
      author={Adal Abilbekov, Saida Mussakhojayeva, Rustem Yeshpanov, Huseyin Atakan Varol},
      year={2024},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={}
}
```

## References

* HiFi-GAN vocoder, official github repository: [link](https://github.com/jik876/hifi-gan).
* Monotonic Alignment Search algorithm is used for unsupervised duration modelling, official github repository: [link](https://github.com/jaywalnut310/glow-tts).
* GradTTS text2speech model, official github repository: [link](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)
