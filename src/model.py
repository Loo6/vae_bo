import torch
from torch import Tensor, nn
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
import pytorch_lightning as pl
from src.loss_vae import loss_fn
from torch import optim
from sklearn.metrics import r2_score

def to_var(x,gpu_exist, volatile=False):
    
    if gpu_exist == True:
        x = x.cuda()
    else:
        x = x
    x = Variable(x, volatile= volatile)
    
    return x

def batch2tensor(batch,Args):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = to_var(v,Args['gpu_exist'])
        if k == 'Effect':
            batch[k] = to_var(v.type(torch.FloatTensor))
    return batch

class Block(nn.Module):
    """
    residual block
    """

    def __init__(
        self, in_features: int, out_features: int, hidden_features: int = None
    ):
        super().__init__()

        if hidden_features is None:
            hidden_features = out_features

        self.block = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Mish(),
            nn.Linear(in_features, hidden_features),
            nn.Mish(),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(0.1)
        )

        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor):
        return self.block(x) + self.skip(x)


class Predictor(nn.Module):
    def __init__(self, latent_size: int, hidden_sizes: list[int] = None):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [192, 128, 64]

        layers = [Block(latent_size, hidden_sizes[0])]
        # preact normalization
        layers.extend(
            Block(hidden_sizes[i - 1], hidden_sizes[i])
            for i in range(1, len(hidden_sizes))
        )
        layers.append(nn.Linear(hidden_sizes[-1], 5))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.mlp(x)

class Model(pl.LightningModule):

    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, max_sequence_length, num_layers=1, prop_hidden_size=None, 
                predict_prop = False, bidirectional=False, gpu_exist = True, NLL = None):

        super().__init__()
        #self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.NLL = NLL
        self.learning_rate = 0.001
        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.gpu_exist  =  gpu_exist

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.predict_prop = predict_prop
        self.hidden_size = hidden_size
        

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout = nn.Dropout(p=word_dropout)
        if self.predict_prop:
            self.predictor = Predictor(latent_size=latent_size, hidden_sizes=prop_hidden_size)
        
        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def forward(self, input_sequence, length):

        batch_size = input_sequence.size(0)
        # print('batch size: ', batch_size)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]
        # print('input_sequence: ', input_sequence.shape)
        #print('length:', length)
        #print('sorted_lengths:', sorted_lengths)
        #print('sorted_idx:', sorted_idx)
        #print('input_sequence:', input_sequence)

        # ENCODER
        input_embedding = self.embedding(input_sequence)
        # print('input_embedding: ', input_embedding.shape)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)
        # print('hidden: ', hidden.shape)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # print('hidden: ', hidden.shape)

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        #z = to_var(torch.randn([batch_size, self.latent_size]))
        z = to_var(torch.normal(mean=torch.zeros([batch_size, self.latent_size]), std=0.01*torch.ones([batch_size, self.latent_size])),self.gpu_exist)
        z = z * std + mean

        # DECODER
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        input_embedding = self.word_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
       
        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        #print('reversed_idx:', reversed_idx)

        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)
        z = z[reversed_idx]

        # print('mean: ', mean.shape)
        # print(reversed_idx)

        # if batch_size == 1:
        #     return mean
        # else:
        #     mean = mean[reversed_idx] 
        #     return mean

        if batch_size > 1:
            mean = mean[reversed_idx]
            logv = logv[reversed_idx]

        # mean = mean[reversed_idx] 
        # print('returning mean: ', mean)
        
        # logv = logv[reversed_idx] 
        #prediction = 0
        if self.predict_prop:
            prediction = self.predictor(z)
            return logp, mean, logv, z, prediction
        else:
            return logp, mean, logv, z

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        step = self.global_step
        batch_size = batch['input'].size(0)
        logp, mean, logv, z, prediction = self(batch['input'], batch['length'])
        NLL_loss, KL_loss, KL_weight, prop_pred_loss = loss_fn(self.NLL, logp, batch['target'], batch['length'], mean, logv,
            'logistic', step, 2500, 0.0025,predict_prop = True, prop = batch['prop'], pred = prediction)
        loss = (NLL_loss + KL_weight * KL_loss + prop_pred_loss) / batch_size
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("NLL_loss", NLL_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("KL_loss", KL_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("KL_loss_weight", KL_weight, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("prop_pred_loss", prop_pred_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        step = self.global_step
        batch_size = batch['input'].size(0)
        logp, mean, logv, z, prediction = self(batch['input'], batch['length'])
        NLL_loss, KL_loss, KL_weight, prop_pred_loss = loss_fn(self.NLL, logp, batch['target'], batch['length'], mean, logv,
            'logistic', step, 2500, 0.0025,predict_prop = True, prop = batch['prop'], pred = prediction)
        loss = (NLL_loss + KL_weight * KL_loss + prop_pred_loss) / batch_size
        self.log("val_loss", loss, logger=True, sync_dist=True)
        self.log("val_NLL_loss", NLL_loss, logger=True, sync_dist=True)
        self.log("val_KL_loss", KL_loss, logger=True, sync_dist=True)
        self.log("val_KL_loss_weight", KL_weight, logger=True, sync_dist=True)
        self.log("val_prop_pred_loss", prop_pred_loss, logger=True, sync_dist=True)
        
        hlgap_r2 = r2_score(prediction[0, :].cpu().numpy().flatten(), batch['prop'][0, :].cpu().numpy().flatten())
        ip_r2 = r2_score(prediction[1, :].cpu().numpy().flatten(), batch['prop'][1, :].cpu().numpy().flatten())
        ea_r2 = r2_score(prediction[2, :].cpu().numpy().flatten(), batch['prop'][2, :].cpu().numpy().flatten())
        dipole_r2 = r2_score(prediction[3, :].cpu().numpy().flatten(), batch['prop'][3, :].cpu().numpy().flatten())
        opticalgap_r2 = r2_score(prediction[4, :].cpu().numpy().flatten(), batch['prop'][4, :].cpu().numpy().flatten())
        self.log("HLGap_r2", hlgap_r2, prog_bar=True, logger=True, sync_dist=True)
        self.log("IP_r2", ip_r2, prog_bar=True, logger=True, sync_dist=True)
        self.log("EA_r2", ea_r2, prog_bar=True, logger=True, sync_dist=True)
        self.log("Dipole_r2", dipole_r2, prog_bar=True, logger=True, sync_dist=True)
        self.log("OpticalGap_r2", opticalgap_r2, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def inference(self, z=None):
        pad_idx = 0
        sos_idx = self.sos_idx
        eos_idx = self.eos_idx
        input_sequence = sos_idx
        batch_size = z.size(0)
        generation = []
        hidden = self.latent2hidden(z)
        #print('hidden >>>> ', hidden.size())
        t = 0
        hidden = hidden.unsqueeze(0)
        while (t < self.max_sequence_length and input_sequence != eos_idx):
            #hidden = hidden.unsqueeze(0)
            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(sos_idx).long(),self.gpu_exist)
                #input_sequence = torch.Tensor(batch_size).fill_(sos_idx).long()
            #print(input_sequence.shape)
            input_sequence = input_sequence.unsqueeze(1)
            
            input_embedding = self.embedding(input_sequence)
            
            output, hidden = self.decoder_rnn(input_embedding, hidden)
            hidden = hidden
          
            logits = self.outputs2vocab(output)
           
            input_sequence = self._sample(logits)
            input_sequence =  input_sequence.unsqueeze(0)
           
            generation.append(input_sequence[0].cpu().tolist())

            t = t + 1
        
        return generation#input_sequence#generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to