import torch 
from src.util import kl_anneal_function

def loss_fn(NLL, logp, target, length, mean, logv, anneal_function, step, k0, x0, predict_prop = False, prop = None, pred = None):
        #print(len(logp), len(target)) 
        # (30, 30) - batch size
        #print(logp.shape, target.shape) 
        # torch.Size([30, ?, 50]) torch.Size([30, 74])

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))
        
        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)
        

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k0, x0)

        if predict_prop:
                MSE = torch.nn.MSELoss()
                prop_pred_loss = 0
                for pr_idx in range(len(prop[0])):
                        #print(pr_idx)
                        prop_pred_loss += MSE(pred[:, pr_idx].double(), prop[:, pr_idx].double())
                #print(pred.double(), prop.view(-1, 1).double())
                #prop_pred_loss = MSE(pred.double(), prop.view(-1, 1).double())
                return NLL_loss, KL_loss, KL_weight, prop_pred_loss
        else:
                return NLL_loss, KL_loss, KL_weight

def loss_fn_noweight(NLL, logp, target, length, mean, logv, predict_prop = False, prop = None, pred = None):
        #print(len(logp), len(target)) 
        # (30, 30) - batch size
        #print(logp.shape, target.shape) 
        # torch.Size([30, ?, 50]) torch.Size([30, 74])
        batch_size = len(prop[0])
        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))
        
        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)
        

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())

        if predict_prop:
                MSE = torch.nn.MSELoss()
                hlgap_pred_loss = 0
                ip_pred_loss = 0
                ea_pred_loss = 0
                opticalgap_pred_loss = 0
                for pr_idx in range(batch_size):
                        #print(pr_idx)
                        hlgap_pred_loss += MSE(pred[:, pr_idx][0].double(), prop[:, pr_idx][0].double())
                        ip_pred_loss += MSE(pred[:, pr_idx][1].double(), prop[:, pr_idx][1].double())
                        ea_pred_loss += MSE(pred[:, pr_idx][2].double(), prop[:, pr_idx][2].double())
                        opticalgap_pred_loss += MSE(pred[:, pr_idx][3].double(), prop[:, pr_idx][3].double())
                #print(pred.double(), prop.view(-1, 1).double())
                #prop_pred_loss = MSE(pred.double(), prop.view(-1, 1).double())
                return NLL_loss / batch_size, KL_loss / batch_size, hlgap_pred_loss / batch_size, ip_pred_loss / batch_size, ea_pred_loss / batch_size, opticalgap_pred_loss / batch_size
        else:
                return NLL_loss / batch_size, KL_loss / batch_size