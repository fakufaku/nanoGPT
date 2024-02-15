from typing import Tuple
import torch
import torch.nn as nn


def _compute_iss_update_ica(
    Y: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    masks: torch.Tensor,
    src: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Ys = Y[..., src, :]
    Ys_mag_sq = abs(Ys) ** 2
    ws = W[..., src, :].clone()

    # compute the demixing matrix update vector
    # note: for complex data, Ys should be conjugated
    numerator = torch.einsum("btsd,btsd,btd->bts", masks, Y, Ys)
    denominator = torch.einsum("btsd,btd->bts", masks, Ys_mag_sq)
    v = numerator / (eps + denominator)  # (batch, time, chan)
    ds = (eps + denominator[..., src]).sqrt()  # (batch, time)
    v[..., src] = 1.0 - (1.0 / ds)

    # compute the mixing matrix update vector
    # according to the Sherman-Morrison formula
    a = torch.einsum("bt,btcs,bts->btc", ds, A, v)

    # update the separated sources
    Y = Y - torch.einsum("btc,btd->btcd", v, Ys)

    # update the demixing matrix
    W = W - torch.einsum("btc,bts->btcs", v, ws)

    # update the mixing matrix
    up = A[..., :, src] + a
    A = torch.cat((A[..., :src], up[..., None], A[..., src + 1 :]), dim=-1)

    return Y, W, A


if torch.cuda.is_available():
    compute_iss_update_ica = torch.compile(_compute_iss_update_ica)
else:
    compute_iss_update_ica = _compute_iss_update_ica


def _ica(
    x: torch.Tensor, q: torch.Tensor, num_iter: int, mask_floor: float, eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_groups = x.shape[-2]

    # initialize param.
    eye = torch.eye(num_groups).type_as(x)
    W = torch.broadcast_to(  # (batch, time, grp, grp)
        eye[None, None], x.shape[:2] + eye.shape
    )
    A = torch.broadcast_to(  # (batch, time, grp, grp)
        eye[None, None], x.shape[:2] + eye.shape
    )

    for epoch in range(num_iter):
        # update auxiliary variables
        # mask shape: (batch, time, groups, 1)
        r = torch.mean(x**q, dim=-1, keepdim=True)
        masks = q / (2 * r ** (2.0 - q)).clamp(min=eps)
        masks = masks * (1.0 - mask_floor) + mask_floor
        masks = torch.broadcast_to(masks, x.shape)

        # run the iterative source steering updates
        for src in range(num_groups):
            x, W, A = compute_iss_update_ica(x, W, A, masks, src, eps=eps)

    # apply scaling to the separated sources
    x = x * torch.diagonal(A, dim1=-2, dim2=-1)[..., None]

    return x


if torch.cuda.is_available():
    ica_core = torch.compile(_ica)
else:
    ica_core = _ica


class FeatureICA(nn.Module):
    """
    Performs Independent Component Analysis along the feature dimension of the input
    tensor. It does so in parallel on all the time steps.

    The feature dimension is divided into a number of groups to be made independent
    """

    def __init__(
        self,
        num_groups=4,
        num_iter=10,
        mask_floor=1e-5,
        q=1.0,
        q_learnable=False,
        eps=1e-5,
    ):
        super().__init__()
        self.num_iter = num_iter
        self.eps = eps
        self.mask_floor = mask_floor

        self.num_groups = num_groups

        # learnable q parameter
        q = q * torch.ones(self.num_groups)
        inv_q = -torch.log((2.0 - eps) / (q - eps) - 1)
        if q_learnable:
            self.inv_q = nn.Parameter(inv_q)
            self.inv_q.requires_grad_(True)
        else:
            self.register_buffer("inv_q", inv_q)

    def _get_q(self):
        return (2.0 - self.eps) * torch.sigmoid(self.inv_q) + self.eps

    def forward(self, x):
        """x.shape == (batch, time, features)"""
        x_shape = x.shape
        x = x.reshape(x.shape[:2] + (self.num_groups, -1))

        # independent vector analysis loop
        q = self._get_q()[..., None]

        x, W, A = ica_core(x, q, self.num_iter, self.mask_floor, self.eps)

        x = x.reshape(x_shape)
        return x


if __name__ == "__main__":
    batch = 2
    time = 512
    features = 384

    x = torch.zeros((batch, time, features)).to("mps")
    ica = FeatureICA(
        num_groups=4,
        num_iter=10,
        mask_floor=1e-5,
        q=1.0,
        eps=1e-5,
    ).to("mps")
    u = torch.zeros((batch, time, features)).normal_().to("mps")
    z = ica(u)
    print(u.shape, z.shape)
    breakpoint()
