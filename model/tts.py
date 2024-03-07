import math
import random

import torch

from model import monotonic_align
from model.base import BaseModule
from model.text_encoder import TextEncoder
from model.diffusion import Diffusion
from model.utils import sequence_mask, generate_path, duration_loss, fix_len_compatibility


class GradTTSWithEmo(BaseModule):
    def __init__(self, n_vocab=148, n_spks=1,n_emos=5, spk_emb_dim=64,
                 n_enc_channels=192, filter_channels=768, filter_channels_dp=256,
                 n_heads=2, n_enc_layers=6, enc_kernel=3, enc_dropout=0.1, window_size=4,
                 n_feats=80, dec_dim=64, beta_min=0.05, beta_max=20.0, pe_scale=1000,
                 use_classifier_free=False, dummy_spk_rate=0.5,
                 **kwargs):
        super(GradTTSWithEmo, self).__init__()
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.n_emos = n_emos
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        self.use_classifier_free = use_classifier_free

        # if n_spks > 1:
        self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.emo_emb = torch.nn.Embedding(n_emos, spk_emb_dim)
        self.merge_spk_emo = torch.nn.Sequential(
            torch.nn.Linear(spk_emb_dim*2, spk_emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(spk_emb_dim, spk_emb_dim)
        )
        self.encoder = TextEncoder(n_vocab, n_feats, n_enc_channels, 
                                   filter_channels, filter_channels_dp, n_heads, 
                                   n_enc_layers, enc_kernel, enc_dropout, window_size,
                                   spk_emb_dim=spk_emb_dim, n_spks=n_spks)
        self.decoder = Diffusion(n_feats, dec_dim, spk_emb_dim, beta_min, beta_max, pe_scale)

        if self.use_classifier_free:
            self.dummy_xv = torch.nn.Parameter(torch.randn(size=(spk_emb_dim, )))
            self.dummy_rate = dummy_spk_rate
            print(f"Using classifier free with rate {self.dummy_rate}")

    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, temperature=1.0, stoc=False, spk=None, emo=None,
                length_scale=1.0,  classifier_free_guidance=1., force_dur=None):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        # Get speaker embedding
        spk = self.spk_emb(spk)
        emo = self.emo_emb(emo)
        
        if self.use_classifier_free:
            emo = emo / torch.sqrt(torch.sum(emo**2, dim=1, keepdim=True))  # unit norm
        
        spk_merged = self.merge_spk_emo(torch.cat([spk, emo], dim=-1))
        
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk_merged)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        if force_dur is not None:
            w_ceil = force_dur.unsqueeze(1)  # [1, 1, Ltext]
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # print(z)
        # Generate sample by performing reverse dynamics
        
        unit_dummy_emo = self.dummy_xv / torch.sqrt(torch.sum(self.dummy_xv**2)) if self.use_classifier_free else None
        dummy_spk = self.merge_spk_emo(torch.cat([spk, unit_dummy_emo.unsqueeze(0).repeat(len(spk), 1)], dim=-1)) if self.use_classifier_free else None

        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk_merged,
                                       use_classifier_free=self.use_classifier_free,
                                       classifier_free_guidance=classifier_free_guidance,
                                       dummy_spk=dummy_spk)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def classifier_guidance_decode(self, x, x_lengths, n_timesteps, temperature=1.0, stoc=False, spk=None, emo=None,
                                   length_scale=1.0, classifier_func=None, guidance=1.0, classifier_type='conformer'):
        x, x_lengths = self.relocate_input([x, x_lengths])

        # Get speaker embedding
        spk = self.spk_emb(spk)
        dummy_emo = self.emo_emb(torch.zeros_like(emo).long())  # this is for feeding the text encoder.

        spk_merged = self.merge_spk_emo(torch.cat([spk, dummy_emo], dim=-1))

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk_merged)

        w = torch.exp(logw) * x_mask
        # print("w shape is ", w.shape)
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        if classifier_type == 'CNN' or classifier_type == 'CNN-with-time' :
            y_max_length = max(y_max_length, 180)  # NOTE: added for CNN classifier
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics

        decoder_outputs = self.decoder.classifier_decode(z, y_mask, mu_y, n_timesteps, stoc, spk_merged,
                                                         classifier_func, guidance,
                                                         control_emo=emo, classifier_type=classifier_type)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]
        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def classifier_guidance_decode_DPS(self, x, x_lengths, n_timesteps, temperature=1.0, stoc=False, spk=None, emo=None,
                                   length_scale=1.0, classifier_func=None, guidance=1.0, classifier_type='conformer'):
        x, x_lengths = self.relocate_input([x, x_lengths])

        # Get speaker embedding
        spk = self.spk_emb(spk)
        dummy_emo = self.emo_emb(torch.zeros_like(emo).long())  # this is for feeding the text encoder.

        spk_merged = self.merge_spk_emo(torch.cat([spk, dummy_emo], dim=-1))

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk_merged)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        if classifier_type == 'CNN' or classifier_type == 'CNN-with-time' :
            y_max_length = max(y_max_length, 180)  # NOTE: added for CNN classifier
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics

        decoder_outputs = self.decoder.classifier_decode_DPS(z, y_mask, mu_y, n_timesteps, stoc, spk_merged,
                                                         classifier_func, guidance,
                                                         control_emo=emo, classifier_type=classifier_type)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]
        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def classifier_guidance_decode_two_mixture(self, x, x_lengths, n_timesteps, temperature=1.0, stoc=False, spk=None, emo1=None, emo2=None, emo1_weight=None,
                                   length_scale=1.0, classifier_func=None, guidance=1.0, classifier_type='conformer'):
        x, x_lengths = self.relocate_input([x, x_lengths])

        # Get speaker embedding
        spk = self.spk_emb(spk)
        dummy_emo = self.emo_emb(torch.zeros_like(emo1).long())  # this is for feeding the text encoder.

        spk_merged = self.merge_spk_emo(torch.cat([spk, dummy_emo], dim=-1))

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk_merged)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        if classifier_type == 'CNN' or classifier_type == 'CNN-with-time' :
            y_max_length = max(y_max_length, 180)  # NOTE: added for CNN classifier
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics

        decoder_outputs = self.decoder.classifier_decode_mixture(z, y_mask, mu_y, n_timesteps, stoc, spk_merged,
                                                         classifier_func, guidance,
                                                         control_emo1=emo1, control_emo2=emo2, emo1_weight=emo1_weight, classifier_type=classifier_type)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]
        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def classifier_guidance_decode_two_mixture_DPS(self, x, x_lengths, n_timesteps, temperature=1.0, stoc=False, spk=None, emo1=None, emo2=None, emo1_weight=None,
                                   length_scale=1.0, classifier_func=None, guidance=1.0, classifier_type='conformer'):
        x, x_lengths = self.relocate_input([x, x_lengths])

        # Get speaker embedding
        spk = self.spk_emb(spk)
        dummy_emo = self.emo_emb(torch.zeros_like(emo1).long())  # this is for feeding the text encoder.

        spk_merged = self.merge_spk_emo(torch.cat([spk, dummy_emo], dim=-1))

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk_merged)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        if classifier_type == 'CNN' or classifier_type == 'CNN-with-time' :
            y_max_length = max(y_max_length, 180)  # NOTE: added for CNN classifier
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics

        decoder_outputs = self.decoder.classifier_decode_mixture_DPS(z, y_mask, mu_y, n_timesteps, stoc, spk_merged,
                                                         classifier_func, guidance,
                                                         control_emo1=emo1, control_emo2=emo2, emo1_weight=emo1_weight, classifier_type=classifier_type)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]
        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def compute_loss(self, x, x_lengths, y, y_lengths, spk=None, emo=None, out_size=None, use_gt_dur=False, durs=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
            use_gt_dur: bool
            durs: gt duration
        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])  # y: B, 80, L

        spk = self.spk_emb(spk)
        emo = self.emo_emb(emo)  # [B, D]
        if self.use_classifier_free:
            emo = emo / torch.sqrt(torch.sum(emo ** 2, dim=1, keepdim=True))  # unit norm
            use_dummy_per_sample = torch.distributions.Binomial(1, torch.tensor(
                [self.dummy_rate] * len(emo))).sample().bool()  # [b, ] True/False where True accords to rate
            emo[use_dummy_per_sample] = (self.dummy_xv / torch.sqrt(
                torch.sum(self.dummy_xv ** 2)))  # substitute with dummy xv(unit norm too)
        
        spk = self.merge_spk_emo(torch.cat([spk, emo], dim=-1))  # [B, D]

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        if use_gt_dur:
            attn = generate_path(durs, attn_mask.squeeze(1)).detach()
        else:
            with torch.no_grad():
                const = -0.5 * math.log(2 * math.pi) * self.n_feats
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
                y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
                mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + const
                # it's actually the log likelihood of y given the Gaussian with (mu_x, I)

                attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
                attn = attn.detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)
        # print(attn.shape)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            clip_size = min(out_size, y_max_length)  # when out_size > max length, do not actually perform clipping
            clip_size = -fix_len_compatibility(-clip_size)  # this is to ensure dividable
            max_offset = (y_lengths - clip_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)
            
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], clip_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, clip_size, dtype=y.dtype, device=y.device)
            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = clip_size + (y_lengths[i] - clip_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)
            
            attn = attn_cut  # attn -> [B, text_length, cut_length]. It does not begin from top left corner
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))  # here mu_x is not cut.
        mu_y = mu_y.transpose(1, 2)  # B, 80, cut_length

        # Compute loss of score-based decoder
        # print(y.shape, y_mask.shape, mu_y.shape)
        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, spk)
        
        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        
        return dur_loss, prior_loss, diff_loss


class GradTTSXvector(BaseModule):
    def __init__(self, n_vocab=148, spk_emb_dim=64,
                 n_enc_channels=192, filter_channels=768, filter_channels_dp=256,
                 n_heads=2, n_enc_layers=6, enc_kernel=3, enc_dropout=0.1, window_size=4,
                 n_feats=80, dec_dim=64, beta_min=0.05, beta_max=20.0, pe_scale=1000, xvector_dim=512, **kwargs):
        super(GradTTSXvector, self).__init__()
        self.n_vocab = n_vocab
        # self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        self.xvector_proj = torch.nn.Linear(xvector_dim, spk_emb_dim)
        self.encoder = TextEncoder(n_vocab, n_feats, n_enc_channels,
                                   filter_channels, filter_channels_dp, n_heads,
                                   n_enc_layers, enc_kernel, enc_dropout, window_size,
                                   spk_emb_dim=spk_emb_dim, n_spks=999)  # NOTE: not important `n_spk`
        self.decoder = Diffusion(n_feats, dec_dim, spk_emb_dim, beta_min, beta_max, pe_scale)

    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, temperature=1.0, stoc=False, spk=None, length_scale=1.0):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
            spk: actually the xvectors
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        spk = self.xvector_proj(spk)  # NOTE: use x-vectors instead of speaker embedding

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def compute_loss(self, x, x_lengths, y, y_lengths, spk=None, out_size=None, use_gt_dur=False, durs=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotonic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
            spk: xvector
            use_gt_dur: bool
            durs: gt duration
        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])

        spk = self.xvector_proj(spk)  # NOTE: use x-vectors instead of speaker embedding

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        if not use_gt_dur:
            with torch.no_grad():
                const = -0.5 * math.log(2 * math.pi) * self.n_feats
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
                y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
                mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + const

                attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
                attn = attn.detach()
        else:
            with torch.no_grad():
                attn = generate_path(durs, attn_mask.squeeze(1)).detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # print(attn.shape)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)

            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)
            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, spk)

        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

        return dur_loss, prior_loss, diff_loss
