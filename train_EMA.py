import numpy as np
from tqdm import tqdm

from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import data_collate
import data_loader
from utils_data import plot_tensor, save_plot
from model.utils import fix_len_compatibility
from text.symbols import symbols
import utils_data as utils


class ModelEmaV2(torch.nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        self.model_state_dict = deepcopy(model.state_dict())
        self.decay = decay
        self.device = device  # perform ema on different device from model if set

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.model_state_dict.values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model_state_dict


if __name__ == "__main__":
    hps = utils.get_hparams()
    logger_text = utils.get_logger(hps.model_dir)
    logger_text.info(hps)

    out_size = fix_len_compatibility(2 * hps.data.sampling_rate // hps.data.hop_length)  # NOTE: 2-sec of mel-spec

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(hps.train.seed)
    np.random.seed(hps.train.seed)

    print('Initializing logger...')
    log_dir = hps.model_dir
    logger = SummaryWriter(log_dir=log_dir)

    train_dataset, collate, model = utils.get_correct_class(hps)
    test_dataset, _, _ = utils.get_correct_class(hps, train=False)

    print('Initializing data loaders...')

    batch_collate = collate
    loader = DataLoader(dataset=train_dataset, batch_size=hps.train.batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=4, shuffle=False)  # NOTE: if on server, worker can be 4

    print('Initializing model...')
    model = model(**hps.model).to(device)
    print('Number of encoder + duration predictor parameters: %.2fm' % (model.encoder.nparams / 1e6))
    print('Number of decoder parameters: %.2fm' % (model.decoder.nparams / 1e6))
    print('Total parameters: %.2fm' % (model.nparams / 1e6))

    use_gt_dur = getattr(hps.train, "use_gt_dur", False)
    if use_gt_dur:
        print("++++++++++++++> Using ground truth duration for training")

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hps.train.learning_rate)

    print('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=hps.train.test_size)
    for i, item in enumerate(test_batch):
        mel = item['mel']
        logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')

    try:
        model, optimizer, learning_rate, epoch_logged = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "grad_*.pt"), model, optimizer)
        epoch_start = epoch_logged + 1
        print(f"Loaded checkpoint from {epoch_logged} epoch, resuming training.")
        global_step = epoch_logged * (len(train_dataset)/hps.train.batch_size)
    except:
        print(f"Cannot find trained checkpoint, begin to train from scratch")
        epoch_start = 1
        global_step = 0
        learning_rate = hps.train.learning_rate

    ema_model = ModelEmaV2(model, decay=0.9999)  # It's necessary that we put this after loading model.

    print('Start training...')
    used_items = set()
    iteration = global_step
    for epoch in range(epoch_start, hps.train.n_epochs + 1):
        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        with tqdm(loader, total=len(train_dataset) // hps.train.batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x, x_lengths = batch['text_padded'].to(device), \
                               batch['input_lengths'].to(device)
                y, y_lengths = batch['mel_padded'].to(device), \
                               batch['output_lengths'].to(device)
                if hps.xvector:
                    spk = batch['xvector'].to(device)
                else:
                    spk = batch['spk_ids'].to(torch.long).to(device)
                emo = batch['emo_ids'].to(torch.long).to(device)

                dur_loss, prior_loss, diff_loss = model.compute_loss(x, x_lengths,
                                                                     y, y_lengths,
                                                                     spk=spk,
                                                                     emo=emo,
                                                                     out_size=out_size,
                                                                     use_gt_dur=use_gt_dur,
                                                                     durs=batch['dur_padded'].to(device) if use_gt_dur else None)
                loss = sum([dur_loss, prior_loss, diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),
                                                               max_norm=1)
                optimizer.step()
                ema_model.update(model)

                logger.add_scalar('training/duration_loss', dur_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/prior_loss', prior_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                  global_step=iteration)

                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())

                if batch_idx % 5 == 0:
                    msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}'
                    progress_bar.set_description(msg)

                iteration += 1

        log_msg = 'Epoch %d: duration loss = %.3f ' % (epoch, float(np.mean(dur_losses)))
        log_msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        log_msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(log_msg)

        if epoch % hps.train.save_every > 0:
            continue

        model.eval()
        print('Synthesis...')

        with torch.no_grad():
            for i, item in enumerate(test_batch):
                if item['utt'] + "/truth" not in used_items:
                    used_items.add(item['utt'] + "/truth")
                x = item['text'].to(torch.long).unsqueeze(0).to(device)
                if not hps.xvector:
                    spk = item['spk_ids']
                    spk = torch.LongTensor([spk]).to(device)
                else:
                    spk = item["xvector"]
                    spk = spk.unsqueeze(0).to(device)
                emo = item['emo_ids']
                emo = torch.LongTensor([emo]).to(device)
                
                x_lengths = torch.LongTensor([x.shape[-1]]).to(device)
                
                y_enc, y_dec, attn = model(x, x_lengths, spk=spk, emo=emo, n_timesteps=10)           
                logger.add_image(f'image_{i}/generated_enc',
                                 plot_tensor(y_enc.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/generated_dec',
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/alignment',
                                 plot_tensor(attn.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                save_plot(y_enc.squeeze().cpu(),
                          f'{log_dir}/generated_enc_{i}.png')
                save_plot(y_dec.squeeze().cpu(),
                          f'{log_dir}/generated_dec_{i}.png')
                save_plot(attn.squeeze().cpu(),
                          f'{log_dir}/alignment_{i}.png')

        ckpt = model.state_dict()

        utils.save_checkpoint(ema_model, optimizer, learning_rate, epoch, checkpoint_path=f"{log_dir}/EMA_grad_{epoch}.pt")
        utils.save_checkpoint(model, optimizer, learning_rate, epoch, checkpoint_path=f"{log_dir}/grad_{epoch}.pt")

