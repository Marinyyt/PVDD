import torch.nn.functional as nn_func
import torch

import sys


def reflection_pad_2d(in_seq, pad):
    mode = 'reflect' if pad[0] > 1 or pad[1] > 1 else 'replicate'
    pad = list(x for x in reversed(pad) for _ in range(2))  # fixing conv / pad inconsistency bug

    b, c, v, h = in_seq.shape
    out_seq = nn_func.pad(in_seq, pad, mode)

    return out_seq


def reflection_pad_vh_3d(in_seq, pad):
    mode = 'reflect' if pad[0] > 1 or pad[1] > 1 else 'replicate'
    pad = list(x for x in reversed(pad) for _ in range(2))  # fixing conv / pad inconsistency bug

    b, c, t, v, h = in_seq.shape
    in_seq = in_seq.transpose(1, 2).reshape(b * t, c, v, h)
    out_seq = nn_func.pad(in_seq, pad, mode).\
        view(b, t, c, v + pad[0] + pad[1], h + pad[2] + pad[3]).transpose(1, 2)

    return out_seq


def replication_pad_3d(in_seq, pad):
    pad = list(x for x in reversed(pad) for _ in range(2))  # fixing conv / pad inconsistency bug

    return nn_func.pad(in_seq, pad, 'replicate')


def reflection_pad_t_3d(in_seq, pad):
    mode = 'reflect' if pad > 1 else 'replicate'
    pad = [pad, pad]

    b, c, t, v, h = in_seq.shape
    in_seq = in_seq.permute(0, 3, 4, 1, 2).reshape(b * v * h, c, t)
    out_seq = nn_func.pad(in_seq, pad, mode).view(b, v, h, c, t + pad[0] + pad[1]).permute(0, 3, 4, 1, 2)

    return out_seq


def reflection_pad_3d(in_seq, pad):
    if pad[0] <= 1 and pad[1] <= 1 and pad[2] <= 1:
        return replication_pad_3d(in_seq, pad)

    else:
        if pad[1] != 0 or pad[2] != 0:
            out_seq = reflection_pad_vh_3d(in_seq, pad[1:3])
        else:
            out_seq = in_seq

        if pad[0] != 0:
            out_seq = reflection_pad_t_3d(out_seq, pad[0])

        return out_seq
