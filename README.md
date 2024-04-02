<h1 align="center">KazEmoTTS <br> ⌨️ 😐 😠 🙂 😞 😱 😮 🗣</h1>

<p align="center">
  <a href="https://github.com/IS2AI/KazEmoTTS/stargazers">
    <img src="https://img.shields.io/github/stars/IS2AI/KazEmoTTS"
         alt="GitHub stars">
  </a>
  <a href="https://github.com/IS2AI/KazEmoTTS/issues">
    <img src="https://img.shields.io/github/issues/IS2AI/KazEmoTTS.svg"
         alt="GitHub issues">
  </a>
  <a href="https://issai.nu.edu.kz">
    <img src="https://img.shields.io/static/v1?label=ISSAI&amp;message=official site&amp;color=blue&amp"
         alt="ISSAI Official Website">
  </a> 
</p>

<p align = "center">This repository provides a <a href="https://docs.google.com/forms/d/e/1FAIpQLSeTg88cvRbZkR5Go1p0IkQxFnOJv2KL6j2WVcsa6ut4XzQp5g/viewform">dataset</a> and a text-to-speech (TTS) model for the paper <br><a href = "link_to_be_added"><b>KazEmoTTS:
A Dataset for Kazakh Emotional Text-to-Speech Synthesis</b></a></p>

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

<h2 align = "justify">Installation 🛠️</h2>

<p align = "justify">First, you need to build the <a href = 'https://github.com/jaywalnut310/glow-tts'><tt>monotonic_align</tt></a> code:</p>

```bash
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

**Note**: Python version is 3.9.13

<h2 align = "justify">Pre-Processing Data for Training 🧹</h2>

<p align = "justify">You need to download the <a href = "https://docs.google.com/forms/d/e/1FAIpQLSeTg88cvRbZkR5Go1p0IkQxFnOJv2KL6j2WVcsa6ut4XzQp5g/viewform">KazEmoTTS</a> dataset and customize it, as in <tt>filelists/all_spk</tt>, by executing <tt>data_preparation.py</tt>:</p>

```shell
python data_preparation.py -d <path_to_KazEmoTTS_dataset>
```

<h2 align = "justify">Training Stage 🏋️‍♂️</h2>

<p align = "justify">To initiate the training process, you must specify the path to the model configurations, which can be found in <tt>configs/train_grad.json</tt>, and designate a directory for checkpoints, typically located at <tt>logs/train_logs</tt>, to specify the GPU you will be using.</p>

```shell
CUDA_VISIBLE_DEVICES=YOUR_GPU_ID
python train_EMA.py -c <configs/train_grad.json> -m <checkpoint>
```

<h2 align = "justify">Inference 🧠</h2>

<h3 align = "justify">Pre-Training Stage 🏃</h3>

<p align = "justify">If you intend to utilize a pre-trained model, you will need to download the necessary checkpoints <a href = "https://issai.nu.edu.kz/wp-content/uploads/2024/03/pt_10000.zip">TTS</a>, <a href = "https://issai.nu.edu.kz/wp-content/uploads/2024/03/pre_trained_hf.zip">vocoder</a> for both the TTS model based on <a href = "https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS">GradTTS</a> and <a href = "https://github.com/jik876/hifi-gan">HiFi-GAN</a>.</p>

To conduct inference, follow these steps:

- Create a text file containing the sentences you wish to synthesize, such as `filelists/inference_generated.txt`.
- Specify the `txt` file format as follows: `text|emotion id|speaker id`.
- Adjust the path to the HiFi-Gan checkpoint in `inference_EMA.py`.
- Set the classifier guidance level to 100 using the `-g` parameter.

```shell
python inference_EMA.py -c <config> -m <checkpoint> -t <number-of-timesteps> -g <guidance-level> -f <path-for-text> -r <path-to-save-audios>
```

<h2 align = "justify">Synthesized samples 🔈</h2>
<p align = "justify">You can listen to some synthesized samples <a href = "https://anonimous4849.github.io">here</a>.</p>

<h2 align = "justify">Citation 🎓</h2>

<p align = "justify">We kindly urge you, if you incorporate our dataset and/or model into your work, to cite our paper as a gesture of recognition for its valuable contribution. The act of referencing the relevant sources not only upholds academic honesty but also ensures proper acknowledgement of the authors' efforts. Your citation in your research significantly contributes to the continuous progress and evolution of the scholarly realm. Your endorsement and acknowledgement of our endeavours are genuinely appreciated.

```bibtex
@misc{abilbekov2024kazemotts,
      title={KazEmoTTS: A Dataset for Kazakh Emotional Text-to-Speech Synthesis}, 
      author={Adal Abilbekov and Saida Mussakhojayeva and Rustem Yeshpanov and Huseyin Atakan Varol},
      year={2024},
      eprint={2404.01033},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
