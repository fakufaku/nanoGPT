import torch
import torch.nn as nn


def _compute_iss_update(Y, W, A, masks, attention, src, eps=1e-5):
    Ys = Y[..., src, :, :]
    Ys_mag_sq = abs(Ys) ** 2
    ws = W[..., src, :].clone()

    # compute the demixing matrix update vector
    numerator = torch.einsum("bsft,bsft,bft->bsft", masks, Y, Ys.conj())
    numerator = torch.einsum("bsnt,bsft->bsfn", attention.type_as(numerator), numerator)
    denominator = torch.einsum("bsft,bft->bsft", masks, Ys_mag_sq)
    denominator = torch.einsum(
        "bsnt,bsft->bsfn", attention.type_as(denominator), denominator
    )
    v = numerator / (eps[..., None] + denominator)  # (batch, chan, freq, time)
    ds = (
        eps[0, :, :, None] + denominator[..., src, :, :]
    ).sqrt()  # (batch, freq, time)
    v[..., src, :, :] = 1.0 - (1.0 / ds)

    # compute the mixing matrix update vector
    # according to the Sherman-Morrison formula
    a = torch.einsum("bft,bftcs,bsft->bftc", ds, A, v)

    # update the separated sources
    Y = Y - torch.einsum("bcft,bft->bcft", v, Ys)

    # update the demixing matrix
    W = W - torch.einsum("bsft,bftc->bftsc", v, ws)

    # update the mixing matrix
    up = A[..., :, src] + a
    A = torch.cat((A[..., :src], up[..., None], A[..., src + 1 :]), dim=-1)

    return Y, W, A


def _mva_one_iter(Y, W, A, mask_model, attention, eps):
    """
    Y: (batch, groups, features, time)
        Input signal
    W: (batch, features, time, groups, groups)
        Separation matrics
    A: (batch, features, time, groups, groups)
        Mixing matrics
    mask_model: Module
        mask computation module
    attention: tensor
        fixed attention mask
    eps: float
        regularization constant
    """
    n_chan = Y.shape[-3]

    masks = mask_model(Y)  # (batch, src, freq, time)

    for src in range(n_chan):
        Y, W, A = _compute_iss_update(Y, W, A, masks, attention, src, eps=eps)

    return Y, W, A


class CausalConv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("Please use odd filter length")
        self.pad_len = kernel_size - 1
        self.pad_causal = nn.ConstantPad2d(
            (self.pad_len, 0, kernel_size // 2, kernel_size // 2), 0
        )
        self.conv = nn.Conv2d(in_features, out_features, kernel_size)

    def forward(self, x):
        x = self.pad_causal(x)
        x = self.conv(x)
        return x


class MaskNetwork(nn.Module):
    def __init__(self, hidden_features, kernel_size=3, mask_floor=1e-5):
        super().__init__()
        self.conv1 = CausalConv2d(1, hidden_features, kernel_size)
        self.conv2 = CausalConv2d(hidden_features, 1, kernel_size)
        self.act = nn.ReLU()
        self.mask_floor = mask_floor
        if not (0.0 <= mask_floor < 1.0):
            raise ValueError("Mask floor parameter must be in [0, 1]")

    def forward(self, x):
        """x.shape == (batch, groups, features, time)"""
        shape = x.shape
        x = x.reshape(shape[0] * shape[1], 1, shape[2], shape[3])
        x = self.conv2(self.act(self.conv1(x)))
        x = torch.softmax(x, dim=-1)
        x = x * (1.0 - self.mask_floor) + self.mask_floor
        x = x.reshape(shape)
        return x


class IVA(nn.Module):
    def __init__(
        self,
        in_features,
        aux_hidden,
        num_groups=4,
        num_iter=10,
        mask_floor=1e-5,
        eps=1e-5,
    ):
        super().__init__()
        self.num_iter = num_iter
        self.eps = eps

        if in_features % num_groups != 0:
            raise ValueError("Number of groups must divide number of features")

        self.num_groups = num_groups
        self.features = in_features // num_groups

        # learnable regularization parameter
        self.reg = nn.Parameter(1e-3 * torch.ones((1, 1, self.features)))

        self.mask_net = MaskNetwork(hidden_features=aux_hidden, mask_floor=mask_floor)

    def _make_attention_matrix(self, x):
        att = torch.tril(x.new_ones(x.shape[-1], x.shape[-1]))
        att = att / att.sum(dim=-1, keepdim=True)  # time averaging
        att = torch.broadcast_to(att[None, None], x.shape[:2] + att.shape)
        return att

    def forward(self, x):
        """x.shape == (batch, time, features)"""
        x = x.transpose(-2, -1).contiguous()

        # reshape to (batch, groups, features, time)
        shape = x.shape
        x = x.reshape(x.shape[0], self.num_groups, -1, x.shape[-1])

        # initialize param.
        eye = torch.eye(x.shape[1]).type_as(x)
        W = torch.broadcast_to(  # (batch, feat, time, grp, grp)
            eye[None, None, None], (x.shape[0], x.shape[2], x.shape[3]) + eye.shape
        )
        A = torch.broadcast_to(  # (batch, feat, time, grp, grp)
            eye[None, None, None], (x.shape[0], x.shape[2], x.shape[3]) + eye.shape
        )
        attention = self._make_attention_matrix(x)

        # independent vector analysis loop
        reg = self.reg**2  # positive number
        for epoch in range(self.num_iter):
            x, W, A = _mva_one_iter(x, W, A, self.mask_net, attention, eps=reg)

        x = x.reshape(shape)
        x = x.transpose(-2, -1).contiguous()

        return x


if __name__ == "__main__":
    batch = 2
    time = 512
    features = 384

    x = torch.zeros((batch, 1, features, time)).to("mps")
    causal_conv = CausalConv2d(1, 1, 3).to("mps")
    with torch.no_grad():
        causal_conv.conv.weight[:] = 1.0
    y = causal_conv(x)
    print(y.shape)

    iva = IVA(
        features,
        features,
        num_groups=4,
        num_iter=10,
        mask_floor=1e-5,
        eps=1e-5,
    ).to("mps")
    u = torch.zeros((batch, time, features)).normal_().to("mps")
    z = iva(u)
    print(u.shape, z.shape)
    breakpoint()
