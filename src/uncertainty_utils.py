import torch
import torch.nn as nn

def _enable_dropout(module: nn.Module):
    """递归启用 Dropout 层（推理阶段做 MC Dropout）。"""
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.train()

class PredictorUncertaintyWrapper(nn.Module):
    """
    封装已有 predictor，返回 (mean, std)。
    mode='mc_dropout' 仅需一张模型；如要做深度集成，在 __init__
    里传入 ensemble=[model1, model2, ...] 并把 mode 设为 'ensemble'。
    """
    def __init__(self, predictor, *, mode="mc_dropout", n_samples=20,
                 ensemble=None):
        super().__init__()
        self.mode, self.n_samples = mode, n_samples
        self.predictor, self.ensemble = predictor, ensemble
        if mode == "mc_dropout":
            _enable_dropout(self.predictor)     # 关键一步

    @torch.no_grad()
    def forward(self, z: torch.Tensor):
        if self.mode == "mc_dropout":
            preds = torch.stack([self.predictor(z) for _ in range(self.n_samples)])
            return preds.mean(0), preds.std(0)
        elif self.mode == "ensemble":
            if self.ensemble is None:
                raise ValueError("ensemble=[] 不能为空")
            preds = torch.stack([m(z) for m in self.ensemble])
            return preds.mean(0), preds.std(0)
        else:
            raise ValueError(f"未知 mode {self.mode}")
