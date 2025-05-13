import copy
import numpy as np
import torch
from torch import nn

class SoftMax(nn.Module):
    def __init__(self, is_image=False):
        super(SoftMax, self).__init__()
        self.is_image = is_image
        # The SoftMax module expects a decoder to be set if forward is used.
        # We will primarily use jacobian_full, which directly works on logits.

    def forward(self, x):
        """Compute softmax probabilities (and constant) stepwise for sequence logits."""
        seq_len = x.shape[1]
        out_vec = []
        for i in range(seq_len):
            # Softmax for each position in sequence
            probs = torch.softmax(self.decoder(x[:, i, :]), dim=-1)
            constant = torch.sum(torch.exp(self.decoder(x[:, i, :])), dim=-1)
            out_vec.append(torch.cat([probs, constant], dim=-1))
        return torch.stack(out_vec, dim=1)

    def jacobian(self, x):
        """
        Computes the Jacobian of the softmax function (for each sequence position) with respect to input x.
        Returns a tensor of shape (batch, sequence*vocab, sequence*vocab).
        """
        if self.is_image:
            # For image (binary classification per pixel case)
            s = torch.softmax(x, dim=2)[..., 0]  # shape: (batch, pixels)
            jacobian = torch.diag_embed(s * (1 - s))
        else:
            s = torch.softmax(x, dim=-1)  # shape: (batch, seq_len, vocab)
            batch_size, seq_length, vocab_size = x.shape
            jacobian = torch.zeros(batch_size, seq_length * vocab_size, seq_length * vocab_size,
                                   device=x.device, dtype=x.dtype)
            for b in range(batch_size):
                for t in range(seq_length):
                    s_t = s[b, t, :]             # softmax output at position t
                    diag_s = torch.diag(s_t)     # diag of s_t
                    s_outer = torch.outer(s_t, s_t)  # outer product s_t * s_t^T
                    jacobian_t = diag_s - s_outer     # Jacobian for position t (vocab×vocab)
                    start = t * vocab_size
                    end = (t + 1) * vocab_size
                    jacobian[b, start:end, start:end] = jacobian_t
        return jacobian

    def jacobian_full(self, x):
        """
        Computes the full Jacobian matrix of the softmax function *and* the normalizing constant c
        with respect to input x. Returns tensor of shape (batch, sequence*(vocab+1), sequence*vocab).
        """
        batch_size, seq_length, vocab_size = x.shape
        output_size = vocab_size + 1  # vocab plus one constant per position
        # Initialize the Jacobian tensor
        jacobian = torch.zeros(batch_size, seq_length * output_size, seq_length * vocab_size,
                               device=x.device, dtype=x.dtype)
        # Compute softmax probabilities and normalization constants
        logits = x  # (batch, seq_len, vocab)
        s = torch.softmax(logits, dim=-1)  # softmax outputs
        c = torch.sum(torch.exp(logits), dim=-1, keepdim=True)  # normalization constants
        # Fill Jacobian blocks for each sequence position
        for b in range(batch_size):
            for t in range(seq_length):
                logits_t = logits[b, t, :]
                s_t = s[b, t, :]
                # Jacobian of softmax probabilities at position t (vocab x vocab)
                diag_s = torch.diag(s_t)
                s_outer = torch.outer(s_t, s_t)
                jacobian_probs = diag_s - s_outer
                # Jacobian of normalization constant c w.r.t logits (1 x vocab)
                exp_logits = torch.exp(logits_t)
                jacobian_c = -exp_logits / (exp_logits.sum() ** 2)  # shape: (vocab_size,)
                # Combine into a single block (output_size x vocab)
                jacobian_block = torch.zeros(output_size, vocab_size, device=x.device, dtype=x.dtype)
                jacobian_block[:vocab_size, :] = jacobian_probs
                jacobian_block[vocab_size, :] = jacobian_c
                # Place this block in the big Jacobian matrix
                start_row = t * output_size
                end_row = (t + 1) * output_size
                start_col = t * vocab_size
                end_col = (t + 1) * vocab_size
                jacobian[b, start_row:end_row, start_col:end_col] = jacobian_block
        return jacobian

def net_derivative(x, net):
    """
    Finite-difference approximation of the decoder's output derivatives w.r.t. latent vector x.
    Returns a Jacobian tensor J of shape (batch, latent_dim, seq_len, vocab).
    """
    # If using CUDA, ensure inputs and network are on GPU
    if torch.cuda.is_available():
        x = x.to(device="cuda")
        net = net.to(device="cuda")
    eps = 1e-4
    batch_size, latent_dim = x.shape
    # Base prediction for x
    pred_x = net(x)
    # Repeat base prediction for each latent dimension perturbation
    pred_x = pred_x.repeat(latent_dim, 1, 1)  # shape: (latent_dim * batch, seq_len, vocab)
    dxs_list = []
    for i in range(latent_dim):
        dx = torch.zeros_like(x)
        dx[:, i] = eps
        dxs_list.append(x.clone() + dx)
    del x  # free memory
    dxs = torch.cat(dxs_list, dim=0)          # all perturbed inputs (latent_dim * batch, latent_dim)
    _batch_size = 200
    n_batches = int(np.ceil(dxs.shape[0] / _batch_size))
    dys_vec_batches = []
    # Compute output differences in batches for memory efficiency
    for batch_idx in range(n_batches):
        start_idx = batch_idx * _batch_size
        end_idx = (batch_idx + 1) * _batch_size
        dxs_batch = dxs[start_idx:end_idx]
        dys_batch = net(dxs_batch) - pred_x[start_idx:end_idx]
        dys_vec_batches.append(dys_batch)
    dys_vec = torch.cat(dys_vec_batches, dim=0)  # shape: (latent_dim * batch, seq_len, vocab)
    J = dys_vec / eps  # finite difference quotient
    # Reshape J to (batch, latent_dim, seq_len, vocab)
    shift = np.arange(0, latent_dim * batch_size, batch_size)
    J = torch.stack([J[j + shift, ...] for j in range(batch_size)], dim=0)
    J = J.detach().cpu()  # return on CPU (to save GPU memory; will be sent to GPU later in LES forward)
    return J

class LESModelWrapper(nn.Module):
    """
    Wrapper to provide a .decoder interface for the VAE model.
    This ensures compatibility with the LES module expecting model.decoder.
    """
    def __init__(self, vae_model):
        super(LESModelWrapper, self).__init__()
        # Define a decoder submodule that calls the VAE's decode_logits
        class Decoder(nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae
            def forward(self, z):
                # Use the VAE's decode_logits to get logits for the entire sequence
                return self.vae.decode_logits(z)
        self.decoder = Decoder(vae_model)

def _safe_les(les_tensor: torch.Tensor,
                bad_val: float = 50.0) -> torch.Tensor:
    """
    把 les_tensor 中的 nan / inf 统一替换成 +bad_val (正数)。
    保证返回值 finite；bad_val 越大 → 惩罚越重。
    """
    les_tensor = torch.nan_to_num(
        les_tensor, nan=bad_val, posinf=bad_val, neginf=bad_val
    )
    return les_tensor

class LES(nn.Module):
    def __init__(self, model, polarity=False):
        """
        LES penalty module. Deep-copies the given model (with model.decoder defined) for internal use.
        If polarity=False, uses full LES (entropy) penalty; if True, uses Jacobian norms without softmax.
        """
        super(LES, self).__init__()
        self.model = copy.deepcopy(model).train()
        # Ensure decoder is in training mode (needed for gradients) 
        self.model.decoder = self.model.decoder.train()
        # Store device from model's decoder parameters
        self.device = next(self.model.decoder.parameters()).device
        self.dtype = next(self.model.decoder.parameters()).dtype
        self.polarity = polarity

    def forward(self, x, a_omega=None):
        # 1) 先确定当前要工作的 device / dtype
        if torch.cuda.is_available():
            x = x.to(device="cuda")
            self.model = self.model.to(device="cuda")
            self.model.decoder = self.model.decoder.to(device="cuda")
        device_cur = x.device          # 运行时设备
        dtype_cur  = x.dtype           # 运行时 dtype
    
        # 2) 计算 decoder logits
        pre_sm = self.model.decoder(x)   # (B, L, V)
        seq_len = pre_sm.shape[1]
    
        # 3) a_omega (Jacobian of decoder w.r.t latent)
        if a_omega is None:
            with torch.no_grad():
                a_omega = net_derivative(x.clone().cpu(),   # CPU 里算 Jacobian，可节省 GPU 显存
                                        self.model.decoder.cpu())  # 保证 derivative 阶段不占 GPU
        # a_omega 形状: (B, L, latent_dim, V)
        a_omega = torch.cat([a_omega[:, :, i, :] for i in range(seq_len)], dim=-1)
        a_omega = a_omega.to(dtype=dtype_cur, device=device_cur).transpose(1, 2)
        #               (B, latent_dim, L*V)
    
        # 4) Softmax Jacobian
        if not self.polarity:
            sm = SoftMax(is_image=False)
            softmax_J = sm.jacobian_full(pre_sm).to(dtype=dtype_cur, device=device_cur)
            # (B, L*(V+1), L*V)
            c = torch.bmm(softmax_J, a_omega)    # 设备一致 ✅
        else:
            c = a_omega                          # polarity=True 直接用原 Jacobian
    
        # 5) C^T C 及奇异值
        cov = torch.bmm(c.transpose(1, 2), c)    # (B, latent_dim, latent_dim)
        svals = torch.linalg.svdvals(cov)        # (B, latent_dim)
        penalty = -0.5 * torch.log(svals.clamp_min(1e-8)).sum(dim=1)
        penalty = _safe_les(penalty, bad_val=-1500.0)
        return penalty          # (B,)

