import math
import torch
from einops import rearrange

from model.base import BaseModule


class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)  # kernel=3, stride=2, padding=1.

    def forward(self, x):
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3, 
                                         padding=1), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                               dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(BaseModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)  # NOTE: 1x1 conv
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)            

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GradLogPEstimator2d(BaseModule):
    def __init__(self, dim, dim_mults=(1, 2, 4), groups=8, spk_emb_dim=64, n_feats=80, pe_scale=1000):
        super(GradLogPEstimator2d, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale
        
        self.spk_mlp = torch.nn.Sequential(torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4), Mish(),
                                           torch.nn.Linear(spk_emb_dim * 4, n_feats))
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), Mish(),
                                       torch.nn.Linear(dim * 4, dim))

        dims = [3, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                       ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                       Residual(Rezero(LinearAttention(dim_out))),
                       Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                     ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                     ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                     Residual(Rezero(LinearAttention(dim_in))),
                     Upsample(dim_in)]))
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, x, mask, mu, t, spk=None):
        # x, mu: [B, 80, L], t: [B, ], mask: [B, 1, L]
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)
        
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)  # [B, 64]

        s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        x = torch.stack([mu, x, s], 1)  # [B, 3, 80, L]
        mask = mask.unsqueeze(1)  # [B, 1, 1, L]

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)  # [B, 64, 80, L]
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)


def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        noise = beta_init*t + 0.5*(beta_term - beta_init)*(t**2)
    else:
        noise = beta_init + (beta_term - beta_init)*t
    return noise


class Diffusion(BaseModule):
    def __init__(self, n_feats, dim, spk_emb_dim=64,
                 beta_min=0.05, beta_max=20, pe_scale=1000):
        super(Diffusion, self).__init__()
        self.n_feats = n_feats
        self.dim = dim
        # self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        
        self.estimator = GradLogPEstimator2d(dim,
                                             spk_emb_dim=spk_emb_dim,
                                             pe_scale=pe_scale,
                                             n_feats=n_feats)

    def forward_diffusion(self, x0, mask, mu, t):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)  # it is actually the integral of beta
        mean = x0*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None,
                          use_classifier_free=False,
                          classifier_free_guidance=3.0,
                          dummy_spk=None):  # emo need to be merged by spk

        # looks like a plain Euler-Maruyama method
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max, 
                                cumulative=False)

            if not use_classifier_free:
                if stoc:  # adds stochastic term
                    dxt_det = 0.5 * (mu - xt) - self.estimator(xt, mask, mu, t, spk)
                    dxt_det = dxt_det * noise_t * h
                    dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                           requires_grad=False)
                    dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                    dxt = dxt_det + dxt_stoc
                else:
                    dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t, spk))
                    dxt = dxt * noise_t * h
                xt = (xt - dxt) * mask
            else:
                if stoc:  # adds stochastic term
                    score_estimate = (1 + classifier_free_guidance) * self.estimator(xt, mask, mu, t, spk) \
                                     - classifier_free_guidance * self.estimator(xt, mask, mu, t, dummy_spk)
                    dxt_det = 0.5 * (mu - xt) - score_estimate
                    dxt_det = dxt_det * noise_t * h
                    dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                           requires_grad=False)
                    dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                    dxt = dxt_det + dxt_stoc
                else:
                    score_estimate = (1 + classifier_free_guidance) * self.estimator(xt, mask, mu, t, spk) \
                                     - classifier_free_guidance * self.estimator(xt, mask, mu, t, dummy_spk)
                    dxt = 0.5 * (mu - xt - score_estimate)
                    dxt = dxt * noise_t * h
                xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def forward(self, z, mask, mu, n_timesteps, stoc=False, spk=None,
                use_classifier_free=False,
                classifier_free_guidance=3.0,
                dummy_spk=None
                ):
        return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc, spk, use_classifier_free, classifier_free_guidance, dummy_spk)

    def loss_t(self, x0, mask, mu, t, spk=None):
        xt, z = self.forward_diffusion(x0, mask, mu, t)  # z is sampled from N(0, I)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        noise_estimation = self.estimator(xt, mask, mu, t, spk)
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))  # multiply by lambda which is set to be variance
        # actually multiplied by sqrt(lambda), but not lambda
        # NOTE: here use a trick to put lambda into L2 norm so that don't divide z with std.
        loss = torch.sum((noise_estimation + z)**2) / (torch.sum(mask)*self.n_feats)
        return loss, xt

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, mu, t, spk)

    def classifier_decode(self, z, mask, mu, n_timesteps, stoc=False, spk=None, classifier_func=None, guidance=1.0, control_emo=None, classifier_type="conformer"):
        # control_emo should be [B, ] tensor
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype,
                                                   device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max,
                                cumulative=False)
            # =========== classifier part ==============
            xt = xt.detach()
            xt.requires_grad_(True)
            if classifier_type == 'CNN-with-time':
                logits = classifier_func(xt.transpose(1, 2), mu.transpose(1, 2), (mask == 1.0).squeeze(1), t=t)
            else:
                logits = classifier_func(xt.transpose(1, 2), mu.transpose(1, 2), (mask == 1.0).squeeze(1))

            if classifier_type == 'conformer':  # [B, C]
                probs = torch.log_softmax(logits, dim=-1)  # [B, C]
            elif classifier_type == 'CNN' or classifier_type == 'CNN-with-time' :
                probs_every_place = torch.softmax(logits, dim=-1)  # [B, T', C]
                probs_mean = torch.mean(probs_every_place, dim=1)  # [B, C]
                probs = torch.log(probs_mean)
            else:
                raise NotImplementedError

            control_emo_probs = probs[torch.arange(len(control_emo)).to(control_emo.device), control_emo]
            control_emo_probs.sum().backward(retain_graph=True)
            # NOTE: sum is to treat all the components as the same weight.
            xt_grad = xt.grad
            # ==========================================

            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (mu - xt) - self.estimator(xt, mask, mu, t, spk) - guidance * xt_grad
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t, spk) - guidance * xt_grad)
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    def classifier_decode_DPS(self, z, mask, mu, n_timesteps, stoc=False, spk=None, classifier_func=None, guidance=1.0, control_emo=None, classifier_type="conformer"):
        # control_emo should be [B, ] tensor
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max, cumulative=False)
            beta_integral_t = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
            bar_alpha_t = math.exp(-beta_integral_t)

            # =========== classifier part ==============
            xt = xt.detach()
            xt.requires_grad_(True)
            score_estimate = self.estimator(xt, mask, mu, t, spk)
            x0_hat = (xt + (1-bar_alpha_t) * score_estimate) / math.sqrt(bar_alpha_t)

            if classifier_type == 'CNN-with-time':
                raise NotImplementedError
            else:
                logits = classifier_func(x0_hat.transpose(1, 2), mu.transpose(1, 2), (mask == 1.0).squeeze(1))
            if classifier_type == 'conformer':  # [B, C]
                probs = torch.log_softmax(logits, dim=-1)  # [B, C]
            elif classifier_type == 'CNN':
                probs_every_place = torch.softmax(logits, dim=-1)  # [B, T', C]
                probs_mean = torch.mean(probs_every_place, dim=1)  # [B, C]

                probs_mean = probs_mean + 10E-10
                # NOTE: at the first few steps, x0 may be very large. Then the classifier output logits will also have extreme value range.
                #

                probs = torch.log(probs_mean)
            else:
                raise NotImplementedError

            control_emo_probs = probs[torch.arange(len(control_emo)).to(control_emo.device), control_emo]
            control_emo_probs.sum().backward(retain_graph=True)
            # NOTE: sum is to treat all the components as the same weight.
            xt_grad = xt.grad
            # ==========================================

            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (mu - xt) - score_estimate - guidance * xt_grad
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device, requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - score_estimate - guidance * xt_grad)
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    def classifier_decode_mixture(self, z, mask, mu, n_timesteps, stoc=False, spk=None, classifier_func=None, guidance=1.0, control_emo1=None,control_emo2=None, emo1_weight=None, classifier_type="conformer"):
        # control_emo should be [B, ] tensor
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype,
                                                   device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max,
                                cumulative=False)
            # =========== classifier part ==============
            xt = xt.detach()
            xt.requires_grad_(True)
            if classifier_type == 'CNN-with-time':
                logits = classifier_func(xt.transpose(1, 2), mu.transpose(1, 2), (mask == 1.0).squeeze(1), t=t)
            else:
                logits = classifier_func(xt.transpose(1, 2), mu.transpose(1, 2), (mask == 1.0).squeeze(1))

            if classifier_type == 'conformer':  # [B, C]
                probs = torch.log_softmax(logits, dim=-1)  # [B, C]
            elif classifier_type == 'CNN' or classifier_type == 'CNN-with-time' :
                probs_every_place = torch.softmax(logits, dim=-1)  # [B, T', C]
                probs_mean = torch.mean(probs_every_place, dim=1)  # [B, C]
                probs = torch.log(probs_mean)
            else:
                raise NotImplementedError

            control_emo_probs1 = probs[torch.arange(len(control_emo1)).to(control_emo1.device), control_emo1]
            control_emo_probs2 = probs[torch.arange(len(control_emo2)).to(control_emo2.device), control_emo2]
            control_emo_probs = control_emo_probs1 * emo1_weight + control_emo_probs2 * (1-emo1_weight)  # interpolate

            control_emo_probs.sum().backward(retain_graph=True)
            # NOTE: sum is to treat all the components as the same weight.
            xt_grad = xt.grad
            # ==========================================

            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (mu - xt) - self.estimator(xt, mask, mu, t, spk) - guidance * xt_grad
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t, spk) - guidance * xt_grad)
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    def classifier_decode_mixture_DPS(self, z, mask, mu, n_timesteps, stoc=False, spk=None, classifier_func=None, guidance=1.0, control_emo1=None,control_emo2=None, emo1_weight=None, classifier_type="conformer"):
        # control_emo should be [B, ] tensor
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype,
                                                   device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max,
                                cumulative=False)
            beta_integral_t = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
            bar_alpha_t = math.exp(-beta_integral_t)
            # =========== classifier part ==============
            xt = xt.detach()
            xt.requires_grad_(True)
            score_estimate = self.estimator(xt, mask, mu, t, spk)
            x0_hat = (xt + (1 - bar_alpha_t) * score_estimate) / math.sqrt(bar_alpha_t)

            if classifier_type == 'CNN-with-time':
                raise NotImplementedError
            else:
                logits = classifier_func(x0_hat.transpose(1, 2), mu.transpose(1, 2), (mask == 1.0).squeeze(1))

            if classifier_type == 'conformer':  # [B, C]
                probs = torch.log_softmax(logits, dim=-1)  # [B, C]
            elif classifier_type == 'CNN' or classifier_type == 'CNN-with-time' :
                probs_every_place = torch.softmax(logits, dim=-1)  # [B, T', C]
                probs_mean = torch.mean(probs_every_place, dim=1)  # [B, C]
                probs_mean = probs_mean + 10E-10

                probs = torch.log(probs_mean)
            else:
                raise NotImplementedError

            control_emo_probs1 = probs[torch.arange(len(control_emo1)).to(control_emo1.device), control_emo1]
            control_emo_probs2 = probs[torch.arange(len(control_emo2)).to(control_emo2.device), control_emo2]
            control_emo_probs = control_emo_probs1 * emo1_weight + control_emo_probs2 * (1-emo1_weight)  # interpolate

            control_emo_probs.sum().backward(retain_graph=True)
            # NOTE: sum is to treat all the components as the same weight.
            xt_grad = xt.grad
            # ==========================================

            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (mu - xt) - score_estimate - guidance * xt_grad
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - score_estimate - guidance * xt_grad)
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt
