import torch
import torch.nn.functional as F

from util.util import normalize_disparity


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _ensure_4d(x):
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x


def _resize_mask(mask, size):
    mask = _ensure_4d(mask.float())
    if mask.shape[-2:] != tuple(size):
        mask = F.interpolate(mask, size=size, mode='nearest')
    return mask >= 0.5


def _resize_like(x, size, mode='bilinear'):
    x = _ensure_4d(x)
    if x.shape[-2:] == tuple(size):
        return x
    if mode == 'nearest':
        return F.interpolate(x.float(), size=size, mode='nearest')
    return F.interpolate(x.float(), size=size, mode=mode, align_corners=True)


def reliability_aware_prior_consistency_loss(aux, valid, cfg, beta=0.9, eps=1e-6):
    """RAPC exactly following the RMSI-Stereo paper.

    e_t = SmoothL1(N_sg(d^(t+1)) - N(d_M))
    ebar_t = sum_x sg(W_t) e_t / (sum_x sg(W_t) + eps)
    L_RAPC = sum_t omega_t ebar_t / sum_t omega_t

    ``mode`` is only for the paper's loss ablation:
      - global: W_t = 1
      - uncertainty: W_t = U_t
      - support: W_t = S_t = U_t C_t (full RAPC)
    """
    preds = aux.get('rapc/predictions', []) if isinstance(aux, dict) else []
    supports = aux.get('rapc/support', []) if isinstance(aux, dict) else []
    uncertainties = aux.get('rapc/uncertainty', []) if isinstance(aux, dict) else []
    mono = aux.get('rapc/mono_prior', None) if isinstance(aux, dict) else None

    if not preds or mono is None:
        if torch.is_tensor(mono):
            return mono.sum() * 0.0, {}
        return torch.tensor(0.0), {}

    mode = str(_cfg_get(cfg, 'mode', 'support')).lower()
    if mode not in {'global', 'uncertainty', 'support'}:
        raise ValueError(f'rapc.mode must be global, uncertainty, or support, got {mode}')

    iteration_gamma = float(_cfg_get(cfg, 'iteration_gamma', beta))
    terms = []
    weights = []
    support_means = []

    for t, pred in enumerate(preds):
        size = pred.shape[-2:]
        mono_t = _resize_like(mono, size, mode='bilinear').to(dtype=pred.dtype, device=pred.device)
        valid_t = _resize_mask(valid, size).to(device=pred.device)
        valid_t = valid_t & torch.isfinite(pred) & torch.isfinite(mono_t)

        pred_n, _, _ = normalize_disparity(pred, detach_stats=True)
        mono_n, _, _ = normalize_disparity(mono_t.detach(), detach_stats=False)
        per_pixel = F.smooth_l1_loss(pred_n.float(), mono_n.float(), reduction='none')

        if mode == 'global':
            reliability = torch.ones_like(per_pixel)
        elif mode == 'uncertainty':
            reliability = _resize_like(uncertainties[t], size, mode='bilinear')
        else:
            reliability = _resize_like(supports[t], size, mode='bilinear')

        reliability = reliability.detach().float().clamp(0.0, 1.0)
        reliability = reliability * valid_t.float()
        denom = reliability.sum()
        ebar = (per_pixel * reliability).sum() / (denom + eps)

        omega = iteration_gamma ** (len(preds) - 1 - t)
        terms.append(ebar * omega)
        weights.append(omega)
        support_means.append(reliability.sum() / valid_t.float().sum().clamp_min(1.0))

    loss = torch.stack(terms).sum() / max(sum(weights), eps)
    stats = {
        'rapc/loss_raw': loss.detach(),
        'rapc/weight_mean': torch.stack(support_means).mean().detach(),
    }
    return loss, stats


def sequence_loss(
    init_disp,
    disp_pred,
    disp_gt,
    valid,
    max_disp,
    loss_gamma=0.9,
    aux=None,
    rapc_cfg=None,
    current_step=0,
):
    """RMSI-Stereo objective: L_stereo + lambda_p L_RAPC."""
    del current_step  # RAPC in the paper has no schedule/warm-up term.

    if max_disp:
        valid_mask = (valid >= 0.5) & (disp_gt < max_disp)
    else:
        valid_mask = valid >= 0.5
    valid_mask = valid_mask.bool() & torch.isfinite(disp_gt)

    if valid_mask.sum().item() == 0:
        zero = disp_gt.sum() * 0.0
        return zero, {
            'train/EPE': zero.detach(),
            'train/1px': zero.detach(),
            'train/3px': zero.detach(),
            'train/5px': zero.detach(),
        }

    # rho(d_init - d_gt)
    disp_loss = F.smooth_l1_loss(init_disp[valid_mask], disp_gt[valid_mask], reduction='mean')

    # sum_t beta^(T-t) ||d^(t)-d_gt||_1. With 16 training iterations this
    # matches the paper directly; the adjusted exponent preserves the original
    # PromptStereo weighting if a different number of predictions is requested.
    n_prediction = len(disp_pred)
    adjusted_gamma = loss_gamma ** (15 / max(n_prediction - 1, 1))
    for i, pred in enumerate(disp_pred):
        i_weight = adjusted_gamma ** (n_prediction - i - 1)
        i_loss = torch.abs(pred - disp_gt)
        disp_loss = disp_loss + i_weight * i_loss[valid_mask].mean()

    metric = {}

    rapc_enabled = bool(_cfg_get(rapc_cfg, 'enabled', True))
    if rapc_enabled and aux is not None:
        rapc_raw, rapc_stats = reliability_aware_prior_consistency_loss(
            aux, valid_mask, rapc_cfg, beta=loss_gamma
        )
        lambda_p = float(_cfg_get(rapc_cfg, 'lambda_p', 0.01))
        rapc_term = lambda_p * rapc_raw.to(dtype=disp_loss.dtype, device=disp_loss.device)
        disp_loss = disp_loss + rapc_term
        metric.update(rapc_stats)
        metric['rapc/lambda_p'] = torch.as_tensor(lambda_p, device=disp_gt.device).detach()
        metric['rapc/loss_term'] = rapc_term.detach()

    if aux is not None and isinstance(aux, dict):
        for k, v in aux.items():
            if torch.is_tensor(v) and (k.startswith('rmsi/') or k.startswith('cvum/')):
                metric[k] = v.detach()

    epe = torch.abs(disp_pred[-1] - disp_gt)
    metric.update({
        'train/EPE': epe[valid_mask].mean().detach(),
        'train/1px': (epe[valid_mask] < 1).float().mean().detach(),
        'train/3px': (epe[valid_mask] < 3).float().mean().detach(),
        'train/5px': (epe[valid_mask] < 5).float().mean().detach(),
    })
    return disp_loss, metric
