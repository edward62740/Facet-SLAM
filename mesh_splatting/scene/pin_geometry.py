"""
Fixed triangle centroids,see paper for details.

Essentially this sets the constraint x = mu 1 + Qe where Q is the orthonormal basis complementary to 1
"""
import math
import torch

_S3, _S2, _S6 = 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0)


def q_matrix(device, dtype=torch.float32):
    return torch.tensor([[_S3, _S2, _S6], [_S3, -_S2, _S6], [_S3, 0.0, -2.0 * _S6]], device=device, dtype=dtype)


def to_q(X, Q):
    """per triangle and axis: X (T, 3, d) -> Q^T X"""
    return torch.einsum("kr,tkd->trd", Q, X)


def from_q(Xq, Q):
    return torch.einsum("kr,trd->tkd", Q, Xq)
