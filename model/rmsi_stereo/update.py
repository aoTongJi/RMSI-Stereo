import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import *
from util.util import normalize_disparity


def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)


def interp(x, dest):
    return F.interpolate(x, dest.shape[-2:], mode='bilinear', align_corners=True)


class DispHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.pretrained_model.features
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=self.groups)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return self.skip_add.add(out, x)


class StructureEncoder(nn.Module):
    """Original PromptStereo monocular structure prompt.

    RMSI-Stereo disables this unconditional structure injection by default.
    The module is kept only to reproduce the PromptStereo baseline in ablations
    via ``use_legacy_structure_prompt=True``.
    """

    def __init__(self, cfg):
        super().__init__()
        dim = cfg.pretrained_model.features
        self.convc1 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.convc2 = nn.Conv2d(dim // 2, dim // 2, 3, padding=1)
        self.convd1 = nn.Conv2d(1, dim // 2, 7, padding=3)
        self.convd2 = nn.Conv2d(dim // 2, dim // 2, 3, padding=1)
        self.conv = nn.Conv2d(dim, dim - 1, 3, padding=1)

    def forward(self, ctx, norm_depth, norm_disp):
        diff = torch.abs(norm_depth - norm_disp)
        c = F.relu(self.convc1(ctx), True)
        c = F.relu(self.convc2(c), True)
        d = F.relu(self.convd1(diff), True)
        d = F.relu(self.convd2(d), True)
        out = F.relu(self.conv(torch.cat((c, d), dim=1)), True)
        return torch.cat((out, diff), dim=1)


class MotionEncoder(nn.Module):
    """Stereo motion cue E_M(V_t, d^(t)) retained from PromptStereo."""

    def __init__(self, cfg):
        super().__init__()
        dim = cfg.pretrained_model.features
        cor_plane = (cfg.gwc_group + 1) * (2 * cfg.corr_radius + 1) * cfg.corr_level
        self.convc1 = nn.Conv2d(cor_plane, dim // 2, 1)
        self.convc2 = nn.Conv2d(dim // 2, dim // 2, 3, padding=1)
        self.convd1 = nn.Conv2d(1, dim // 2, 7, padding=3)
        self.convd2 = nn.Conv2d(dim // 2, dim // 2, 3, padding=1)
        self.conv = nn.Conv2d(dim, dim - 1, 3, padding=1)

    def forward(self, corr, disp):
        cor = F.relu(self.convc1(corr), True)
        cor = F.relu(self.convc2(cor), True)
        dis = F.relu(self.convd1(disp), True)
        dis = F.relu(self.convd2(dis), True)
        out = F.relu(self.conv(torch.cat([cor, dis], dim=1)), True)
        return torch.cat([out, disp], dim=1)


class RMSIRecurrentUnit(nn.Module):
    def __init__(
        self,
        cfg,
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        motion=False,
        size=None,
    ):
        super().__init__()
        self.deconv = deconv
        self.expand = expand
        self.align_corners = align_corners
        self.size = size

        out_features = features // 2 if expand else features
        self.out_conv = nn.Conv2d(features, out_features, 1, 1, 0, bias=True)
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        if motion:
            # The structure branch is retained only for the PromptStereo baseline
            # ablation. Full RMSI-Stereo calls this unit with structure=None.
            self.resConfUnitStructure = nn.Sequential(
                BasicConv(features, features, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(features, features, 1, 1, 0),
            )
            self.resConfUnitMotion = nn.Sequential(
                BasicConv(features, features, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(features, features, 1, 1, 0),
            )

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs, structure=None, motion=None, size=None):
        output = xs[0]
        if len(xs) == 2:
            output = self.skip_add.add(output, self.resConfUnit1(xs[1]))
        output = self.resConfUnit2(output)

        if structure is not None:
            output = self.skip_add.add(output, self.resConfUnitStructure(structure))
        if motion is not None:
            output = self.skip_add.add(output, self.resConfUnitMotion(motion))

        return self.out_conv(output)


class CostVolumeUncertaintyModeling(nn.Module):
    """CVUM from the RMSI-Stereo paper.

    Given local level-0 all-pair correlation scores z_{t,j}, CVUM computes a
    masked local matching distribution, normalized entropy, top-2 peak-margin
    uncertainty, monocular-stereo compatibility, and the prior-support score

        S_t = U_t * C_t.

    ``mode`` supports the uncertainty ablations reported in the paper:
    ``entropy``, ``peak``, and ``entropy_peak``.  Setting ``enabled=False``
    yields S_t=1 and reproduces MSCA without CVUM.
    """

    def __init__(self, cfg):
        super().__init__()
        cvum_cfg = getattr(cfg, 'cvum', {})
        self.enabled = bool(getattr(cvum_cfg, 'enabled', True))
        self.temperature = float(getattr(cvum_cfg, 'temperature', 1.0))
        self.tau_d = float(getattr(cvum_cfg, 'tau_d', 0.5))
        self.mode = str(getattr(cvum_cfg, 'mode', 'entropy_peak')).lower()
        if self.mode not in {'entropy', 'peak', 'entropy_peak'}:
            raise ValueError(
                f'cvum.mode must be entropy, peak, or entropy_peak, got {self.mode}'
            )

    @staticmethod
    def _masked_probability(scores, valid_mask, temperature):
        scores = scores.float() / max(float(temperature), 1e-6)
        valid_mask = valid_mask.bool()

        # Use a finite large negative value for mixed-precision stability, then
        # explicitly zero invalid probabilities and renormalize.
        masked_scores = scores.masked_fill(~valid_mask, -1e4)
        prob = torch.softmax(masked_scores, dim=1)
        prob = prob * valid_mask.float()
        denom = prob.sum(dim=1, keepdim=True)
        prob = prob / denom.clamp_min(1e-6)
        return prob, denom

    def forward(self, evidence, norm_depth, norm_disp):
        target_size = norm_disp.shape[-2:]
        if norm_depth.shape[-2:] != target_size:
            norm_depth = F.interpolate(norm_depth, target_size, mode='bilinear', align_corners=True)

        diff = torch.abs(norm_disp.float() - norm_depth.float())
        compatibility = torch.exp(-diff / max(self.tau_d, 1e-6)).clamp(0.0, 1.0)

        if not self.enabled:
            ones = torch.ones_like(compatibility)
            return {
                'uncertainty': ones.detach(),
                'entropy': ones.detach(),
                'peak': ones.detach(),
                'compatibility': compatibility.detach(),
                'support': ones.detach(),
            }

        if evidence is None or evidence.get('scores') is None or evidence.get('valid_mask') is None:
            raise RuntimeError('CVUM requires local APC scores and a valid-candidate mask.')

        scores = evidence['scores']
        valid_mask = evidence['valid_mask']
        if scores.shape[-2:] != target_size:
            scores = F.interpolate(scores, target_size, mode='bilinear', align_corners=True)
            valid_mask = F.interpolate(valid_mask.float(), target_size, mode='nearest') >= 0.5

        prob, prob_sum = self._masked_probability(scores, valid_mask, self.temperature)
        L = scores.shape[1]
        log_l = math.log(max(L, 2))

        entropy = -(prob * torch.log(prob.clamp_min(1e-6))).sum(dim=1, keepdim=True)
        entropy = (entropy / log_l).clamp(0.0, 1.0)

        if L >= 2:
            top2 = torch.topk(prob, k=2, dim=1).values
            peak = 1.0 - (top2[:, 0:1] - top2[:, 1:2])
        else:
            peak = torch.ones_like(entropy)
        peak = peak.clamp(0.0, 1.0)

        # If fewer than two valid hypotheses remain, a meaningful peak
        # separation cannot be established. Treat the location as uncertain.
        valid_count = valid_mask.sum(dim=1, keepdim=True)
        insufficient = (valid_count < 2) | (prob_sum <= 1e-6)
        entropy = torch.where(insufficient, torch.ones_like(entropy), entropy)
        peak = torch.where(insufficient, torch.ones_like(peak), peak)

        if self.mode == 'entropy':
            uncertainty = entropy
        elif self.mode == 'peak':
            uncertainty = peak
        else:
            uncertainty = 0.5 * (entropy + peak)

        uncertainty = uncertainty.clamp(0.0, 1.0)
        support = (uncertainty * compatibility).clamp(0.0, 1.0)

        # Eq. (8) uses sg(S_t). Returning detached reliability descriptors makes
        # this explicit and prevents the network from gaming CVUM.
        return {
            'uncertainty': uncertainty.detach(),
            'entropy': entropy.detach(),
            'peak': peak.detach(),
            'compatibility': compatibility.detach(),
            'support': support.detach(),
        }


class MonocularStereoCrossAttention(nn.Module):
    """MSCA: local cross-attention with stereo queries and monocular K/V."""

    def __init__(self, cfg):
        super().__init__()
        msca_cfg = getattr(cfg, 'msca', {})
        dim = cfg.pretrained_model.features
        mono_dim = int(getattr(msca_cfg, 'mono_dim', dim // 2))
        attn_dim = int(getattr(msca_cfg, 'attn_dim', max(dim // 2, 64)))
        num_heads = int(getattr(msca_cfg, 'num_heads', 4))
        window_size = int(getattr(msca_cfg, 'window_size', 8))
        if attn_dim % num_heads != 0:
            raise ValueError(f'msca.attn_dim ({attn_dim}) must be divisible by num_heads ({num_heads}).')

        self.mono_dim = mono_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv2d(dim, attn_dim, 1)
        self.k_proj = nn.Conv2d(mono_dim, attn_dim, 1)
        self.v_proj = nn.Conv2d(mono_dim, attn_dim, 1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(attn_dim, dim, 1),
            BasicConv(dim, dim, kernel_size=3, stride=1, padding=1),
        )

    def _window_partition(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(B * (Hp // ws) * (Wp // ws), ws * ws, C)
        return x, (B, H, W, Hp, Wp)

    def _window_reverse(self, x, meta):
        B, H, W, Hp, Wp = meta
        ws = self.window_size
        C = x.shape[-1]
        x = x.view(B, Hp // ws, Wp // ws, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, Hp, Wp)
        return x[:, :, :H, :W]

    def forward(self, stereo, mono):
        if mono.shape[-2:] != stereo.shape[-2:]:
            mono = F.interpolate(mono, stereo.shape[-2:], mode='bilinear', align_corners=True)
        if mono.shape[1] != self.mono_dim:
            raise RuntimeError(
                f'MSCA expects monocular feature with {self.mono_dim} channels, got {mono.shape[1]}. '
                'Set model.instance.cfg.msca.mono_dim to the channel number of ctx_mono.'
            )

        q = self.q_proj(stereo)
        k = self.k_proj(mono)
        v = self.v_proj(mono)
        q, meta = self._window_partition(q)
        k, _ = self._window_partition(k)
        v, _ = self._window_partition(v)

        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(out.shape[0], -1, self.attn_dim)
        return self.out_proj(self._window_reverse(out, meta))


class ReliabilityAwareMonocularStereoInteraction(nn.Module):
    """Eq. (8): h_tilde = h_t + gamma * G_t * Abar_t.

    G_t = sg(S_t) * sigmoid(psi([h^(t), P_t^M])).
    """

    def __init__(self, cfg):
        super().__init__()
        dim = cfg.pretrained_model.features
        msca_cfg = getattr(cfg, 'msca', {})
        hidden_dim = int(getattr(msca_cfg, 'gate_hidden_dim', max(dim // 4, 32)))
        gate_bias = float(getattr(msca_cfg, 'gate_init_bias', 0.0))
        self.gamma = float(getattr(msca_cfg, 'gamma', 0.05))
        self.detach_mono = bool(getattr(msca_cfg, 'detach_mono', True))

        self.cross_attention = MonocularStereoCrossAttention(cfg)
        self.gate_predictor = nn.Sequential(
            nn.Conv2d(dim * 2, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
        )
        nn.init.constant_(self.gate_predictor[-1].bias, gate_bias)

    def forward(self, stereo_hidden, mono_feature, motion_prompt, support):
        if self.detach_mono:
            mono_feature = mono_feature.detach()

        cross_response = self.cross_attention(stereo_hidden, mono_feature)
        learned_gate = torch.sigmoid(
            self.gate_predictor(torch.cat([stereo_hidden, motion_prompt], dim=1))
        )
        support = support.detach().to(dtype=learned_gate.dtype)
        gate = support * learned_gate
        delta = self.gamma * gate * cross_response

        aux = {
            'rmsi/gate_mean': gate.detach().mean(),
            'rmsi/learned_gate_mean': learned_gate.detach().mean(),
            'rmsi/gamma': torch.as_tensor(self.gamma, device=stereo_hidden.device).detach(),
            'rmsi/delta_abs_mean': delta.detach().abs().mean(),
        }
        return delta, aux


class RMSIUpdateBlock(nn.Module):
    def __init__(self, cfg, pretrained_state):
        super().__init__()
        self.cfg = cfg
        dim = cfg.pretrained_model.features
        self.stereo_pru = nn.ModuleList([
            RMSIRecurrentUnit(
                cfg,
                features=dim,
                bn=cfg.pretrained_model.use_bn,
                motion=(i == 0),
            )
            for i in range(len(cfg.pretrained_model.out_channels))
        ])
        self.structure_encoder = StructureEncoder(cfg)
        self.motion_encoder = MotionEncoder(cfg)

        msca_cfg = getattr(cfg, 'msca', {})
        self.msca_enabled = bool(getattr(msca_cfg, 'enabled', True))
        self.use_legacy_structure_prompt = bool(getattr(cfg, 'use_legacy_structure_prompt', False))
        self.cvum = CostVolumeUncertaintyModeling(cfg)
        self.rmsi = ReliabilityAwareMonocularStereoInteraction(cfg) if self.msca_enabled else None

        self.disp_head = DispHead(cfg)
        self.mask = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(dim, (2 ** cfg.n_downsample ** 2) * 9, 1, padding=0),
        )

        # Keep the original 3*dim input of the finest update gate so a
        # PromptStereo checkpoint can initialize the recurrent updater exactly.
        self.update = nn.ModuleList([
            nn.Sequential(
                BasicConv(dim * (2 + (i == 0)), dim, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(dim, dim, 1, 1, 0),
                nn.Sigmoid(),
            )
            for i in range(len(cfg.pretrained_model.out_channels))
        ])

        if pretrained_state:
            block_state = self.state_dict()
            new_dict = {}
            for i in range(len(cfg.pretrained_model.out_channels)):
                ref_name = f'scratch.refinenet{i + 1}'
                tar_name = f'stereo_pru.{i}'
                for k, v in pretrained_state.items():
                    if k.startswith(ref_name):
                        new_k = k.replace(ref_name, tar_name)
                        if new_k in block_state and block_state[new_k].shape == v.shape:
                            new_dict[new_k] = v
            block_state.update(new_dict)
            self.load_state_dict(block_state, strict=True)

    def forward(self, net, corr, disp, ctx_mono, norm_depth, local_evidence=None):
        norm_disp, _, _ = normalize_disparity(disp)
        motion = self.motion_encoder(corr, disp)

        reliability = self.cvum(local_evidence, norm_depth, norm_disp)
        aux = {
            'cvum/uncertainty_mean': reliability['uncertainty'].mean(),
            'cvum/entropy_mean': reliability['entropy'].mean(),
            'cvum/peak_mean': reliability['peak'].mean(),
            'cvum/compatibility_mean': reliability['compatibility'].mean(),
            'cvum/support_mean': reliability['support'].mean(),
        }

        # Full RMSI-Stereo replaces PromptStereo's unconditional monocular
        # structure prompt. It can be restored only for the baseline ablation.
        if self.use_legacy_structure_prompt:
            structure = self.structure_encoder(ctx_mono, norm_depth, norm_disp)
            structure_for_update = structure
        else:
            structure = None
            structure_for_update = torch.zeros_like(motion)

        if self.rmsi is not None:
            delta, interaction_aux = self.rmsi(
                net[0], ctx_mono, motion, reliability['support']
            )
            net[0] = net[0] + delta
            aux.update(interaction_aux)

        for i in reversed(range(len(net))):
            if i == len(net) - 1:
                z = self.update[i](torch.cat([net[i], pool2x(net[i - 1])], dim=1))
                net[i] = (1 - z) * net[i] + z * self.stereo_pru[i](net[i])
            elif i == 0:
                z = self.update[i](torch.cat([net[i], structure_for_update, motion], dim=1))
                net[i] = (1 - z) * net[i] + z * self.stereo_pru[i](
                    net[i],
                    interp(net[i + 1], net[i]),
                    structure=structure,
                    motion=motion,
                )
            else:
                z = self.update[i](torch.cat([net[i], pool2x(net[i - 1])], dim=1))
                net[i] = (1 - z) * net[i] + z * self.stereo_pru[i](
                    net[i], interp(net[i + 1], net[i])
                )

        delta_disp = self.disp_head(net[0])
        mask = 0.25 * self.mask(net[0])
        return net, delta_disp, mask, reliability, aux
