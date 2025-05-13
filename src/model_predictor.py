import torch
from torch import Tensor, nn
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
import pytorch_lightning as pl
from src.loss_vae import loss_fn_noweight
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
        layers.append(nn.Linear(hidden_sizes[-1], 4))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.mlp(x)

class CoVWeightingLoss(nn.modules.Module):

    """
        Wrapper of the BaseLoss which weighs the losses to the Cov-Weighting method,
        where the statistics are maintained through Welford's algorithm. But now for 32 losses.
    """

    def __init__(self):
        super(CoVWeightingLoss, self).__init__()
        # self.n = 4
        # self.train = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Record the weights.
        self.num_losses = 4

        # How to compute the mean statistics: Full mean or decaying mean.
        self.mean_decay = False
        # self.mean_decay_param = args.mean_decay_param

        self.current_iter = -1
        self.alphas = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)

        # Initialize all running statistics at 0.
        self.running_mean_L = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_mean_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_S_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_std_l = None

    def forward(self, prop_loss_hlgap, prop_loss_ip, prop_loss_ea, prop_loss_opticalgap):
        # Retrieve the unweighted losses.
        unweighted_losses = torch.stack((prop_loss_hlgap, prop_loss_ip, prop_loss_ea, prop_loss_opticalgap))
        # Put the losses in a list. Just for computing the weights.
        L = unweighted_losses.clone().detach().requires_grad_(False).to(self.device)

        # If we are doing validation, we would like to return an unweighted loss be able
        # to see if we do not overfit on the training set.
        if not self.train:
            return torch.sum(L)

        # Increase the current iteration parameter.
        self.current_iter += 1
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L.clone() if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0

        # If we are in the first iteration set alphas to all 1/32
        if self.current_iter <= 1:
            self.alphas = torch.ones((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device) / self.num_losses
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / torch.sum(ls)

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        elif self.current_iter > 0 and self.mean_decay:
            mean_param = self.mean_decay_param
        else:
            mean_param = (1. - 1 / (self.current_iter + 1))

        # 2. Update the statistics for l
        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L.clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L

        # Get the weighted losses and perform a standard back-pass.
        weighted_losses = [self.alphas[i] * unweighted_losses[i] for i in range(len(unweighted_losses))]
        loss = sum(weighted_losses)
        return {
            "loss": loss
        }

class Predictor_Model(pl.LightningModule):

    def __init__(self, vae_model, prop_hidden_size=None, gpu_exist = True):

        super().__init__()
        #self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.vae = vae_model
        self.batch_size = 128
        self.learning_rate = 0.001
        self.gpu_exist  =  gpu_exist
        self.cwloss = CoVWeightingLoss()

        self.predictor = Predictor(latent_size=192, hidden_sizes=prop_hidden_size)

    def forward(self, z):

        prediction = self.predictor(z)

        return prediction

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        logp, mean, logv, z = self.vae.forward(batch['input'], batch['length'])
        pred = self(z)
        prop = batch['prop']
        MSE = torch.nn.MSELoss()
        hlgap_pred_loss = 0
        ip_pred_loss = 0
        ea_pred_loss = 0
        dipole_pred_loss = 0
        opticalgap_pred_loss = 0
        for pr_idx in range(self.batch_size):
                #print(pr_idx)
                hlgap_pred_loss += MSE(pred[pr_idx][0].double(), prop[pr_idx][0].double())
                ip_pred_loss += MSE(pred[pr_idx][1].double(), prop[pr_idx][1].double())
                ea_pred_loss += MSE(pred[pr_idx][2].double(), prop[pr_idx][2].double())
                opticalgap_pred_loss += MSE(pred[pr_idx][3].double(), prop[pr_idx][3].double())
        loss = hlgap_pred_loss / self.batch_size + ip_pred_loss / self.batch_size + ea_pred_loss / self.batch_size + opticalgap_pred_loss / self.batch_size
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logp, mean, logv, z = self.vae.forward(batch['input'], batch['length'])
        pred = self(z)
        prop = batch['prop']
        MSE = torch.nn.MSELoss()
        hlgap_pred_loss = 0
        ip_pred_loss = 0
        ea_pred_loss = 0
        dipole_pred_loss = 0
        opticalgap_pred_loss = 0
        for pr_idx in range(self.batch_size):
                #print(pr_idx)
                hlgap_pred_loss += MSE(pred[pr_idx][0].double(), prop[pr_idx][0].double())
                ip_pred_loss += MSE(pred[pr_idx][1].double(), prop[pr_idx][1].double())
                ea_pred_loss += MSE(pred[pr_idx][2].double(), prop[pr_idx][2].double())
                opticalgap_pred_loss += MSE(pred[pr_idx][3].double(), prop[pr_idx][3].double())
        loss = hlgap_pred_loss / self.batch_size + ip_pred_loss / self.batch_size + ea_pred_loss / self.batch_size + opticalgap_pred_loss / self.batch_size
        self.log("val_loss", loss, logger=True, sync_dist=True)
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