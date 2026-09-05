import torch
import torch.nn.functional as F
from util.util import corr_sampler


class CombinedGeometryEncodingVolume:
    """PromptStereo geometry encoding plus local APC evidence for RMSI-Stereo.

    The recurrent motion encoder still consumes the original concatenated
    multi-level GWC + all-pairs correlation representation.  RMSI-Stereo's
    CVUM, however, must operate on a *single local disparity distribution*.
    Therefore this class additionally exposes the level-0 all-pairs
    correlation samples and their candidate-validity mask around the current
    disparity estimate.
    """

    def __init__(self, left, right, gwc_volume, level=2, radius=4):
        self.level = level
        self.radius = radius
        self.gev_pyramid = []
        self.apc_pyramid = []

        B, G, D, H, W = gwc_volume.shape
        gwc_volume = (
            gwc_volume.permute(0, 3, 4, 1, 2)
            .contiguous()
            .view(B * H * W, G, 1, D)
        )

        self.gev_pyramid.append(gwc_volume)
        for _ in range(level - 1):
            gwc_volume = F.avg_pool2d(gwc_volume, [1, 2], [1, 2])
            self.gev_pyramid.append(gwc_volume)

        apc_volume = CombinedGeometryEncodingVolume.corr(left, right)
        B, H, W, _, W2 = apc_volume.shape
        apc_volume = apc_volume.view(B * H * W, 1, 1, W2)

        self.apc_pyramid.append(apc_volume)
        for _ in range(level - 1):
            apc_volume = F.avg_pool2d(apc_volume, [1, 2], [1, 2])
            self.apc_pyramid.append(apc_volume)

    @staticmethod
    def corr(left, right):
        B, C, H, W = left.shape
        corr = torch.einsum('aijk,aijh->ajkh', left, right)
        return corr.contiguous().view(B, H, W, 1, W)

    def __call__(self, disp, return_local_evidence=False):
        """Query correlation features around ``disp``.

        Args:
            disp: current disparity, shape [B, 1, H, W] at 1/4 resolution.
            return_local_evidence: if True, additionally return a dict with
                ``scores`` [B, L, H, W] and ``valid_mask`` [B, L, H, W]
                from the level-0 all-pairs correlation volume.  These are the
                z_{t,j} candidates used by CVUM in the paper.
        """
        r = self.radius
        B, _, H, W = disp.shape
        disp_flat = disp.view(B * H * W, 1, 1, 1)
        x0 = (
            torch.arange(W, device=disp.device, dtype=disp.dtype)
            .view(1, 1, W, 1)
            .repeat(B, H, 1, 1)
            .contiguous()
            .view(B * H * W, 1, 1, 1)
        )

        pyramid = []
        local_scores = None
        local_valid = None

        # Symmetric local hypotheses. L = 2r + 1.
        dx = torch.linspace(-r, r, 2 * r + 1, device=disp.device, dtype=disp.dtype)
        dx = dx.view(1, 1, 2 * r + 1, 1)

        for i in range(self.level):
            gwc_volume = self.gev_pyramid[i]
            gwc_coord = dx + disp_flat / (2 ** i)
            gwc_sample = corr_sampler(gwc_volume, gwc_coord)
            gwc_sample = gwc_sample.view(B, H, W, -1).permute(0, 3, 1, 2)

            apc_volume = self.apc_pyramid[i]
            # Right-image x coordinate for local candidates around d^(t).
            apc_coord = dx + (x0 - disp_flat) / (2 ** i)
            apc_sample = corr_sampler(apc_volume, apc_coord)
            apc_sample = apc_sample.view(B, H, W, -1).permute(0, 3, 1, 2)

            if i == 0 and return_local_evidence:
                # Candidate coordinates outside the right image are invalid and
                # are masked *before* CVUM softmax, matching Eq. (5) in the paper.
                valid = (apc_coord >= 0.0) & (apc_coord <= float(W - 1))
                valid = valid.view(B, H, W, 2 * r + 1).permute(0, 3, 1, 2)
                local_scores = apc_sample
                local_valid = valid

            pyramid.append(gwc_sample)
            pyramid.append(apc_sample)

        corr = torch.cat(pyramid, dim=1)
        if not return_local_evidence:
            return corr

        evidence = {
            'scores': local_scores,
            'valid_mask': local_valid,
        }
        return corr, evidence
