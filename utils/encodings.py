import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import numpy as np
import torchac
import math
import multiprocessing

anchor_round_digits = 16
Q_anchor = 1/(2 ** anchor_round_digits - 1)
use_clamp = True
use_multiprocessor = False  # Always False plz. Not yet implemented for True.

def get_binary_vxl_size(binary_vxl):
    # binary_vxl: {0, 1}
    # assert torch.unique(binary_vxl).mean() == 0.5
    ttl_num = binary_vxl.numel()

    pos_num = torch.sum(binary_vxl)
    neg_num = ttl_num - pos_num

    Pg = pos_num / ttl_num  #  + 1e-6
    Pg = torch.clamp(Pg, min=1e-6, max=1-1e-6)
    pos_prob = Pg
    neg_prob = (1 - Pg)
    pos_bit = pos_num * (-torch.log2(pos_prob))
    neg_bit = neg_num * (-torch.log2(neg_prob))
    ttl_bit = pos_bit + neg_bit
    ttl_bit += 32  # Pg
    # print('binary_vxl:', Pg.item(), ttl_bit.item(), ttl_num, pos_num.item(), neg_num.item())
    return Pg, ttl_bit, ttl_bit.item()/8.0/1024/1024, ttl_num


def multiprocess_encoder(lower, symbol, file_name, chunk_num=10):
    def enc_func(l, s, f, b_l, i):
        byte_stream = torchac.encode_float_cdf(l, s, check_input_bounds=True)
        with open(f, 'wb') as fout:
            fout.write(byte_stream)
        bit_len = len(byte_stream) * 8
        b_l[i] = bit_len
    encoding_len = lower.shape[0]
    chunk_len = int(math.ceil(encoding_len / chunk_num))
    processes = []
    manager = multiprocessing.Manager()
    b_list = manager.list([None] * chunk_num)
    for m_id in range(chunk_num):
        lower_m = lower[m_id * chunk_len:(m_id + 1) * chunk_len]
        symbol_m = symbol[m_id * chunk_len:(m_id + 1) * chunk_len]
        file_name_m = file_name.replace('.b', f'_{m_id}.b')
        process = multiprocessing.Process(target=enc_func, args=(lower_m, symbol_m, file_name_m, b_list, m_id))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    ttl_bit_len = sum(list(b_list))
    return ttl_bit_len


def multiprocess_deoder(lower, file_name, chunk_num=10):
    def dec_func(l, f, o_l, i):
        with open(f, 'rb') as fin:
            byte_stream_d = fin.read()
        o = torchac.decode_float_cdf(l, byte_stream_d).to(torch.float32)
        o_l[i] = o
    encoding_len = lower.shape[0]
    chunk_len = int(math.ceil(encoding_len / chunk_num))
    processes = []
    manager = multiprocessing.Manager()
    output_list = manager.list([None] * chunk_num)
    for m_id in range(chunk_num):
        lower_m = lower[m_id * chunk_len:(m_id + 1) * chunk_len]
        file_name_m = file_name.replace('.b', f'_{m_id}.b')
        process = multiprocessing.Process(target=dec_func, args=(lower_m, file_name_m, output_list, m_id))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    output_list = torch.cat(list(output_list), dim=0).cuda()
    return output_list


def encoder_gaussian(x, mean, scale, Q, file_name=None):
    if file_name is not None: assert file_name.endswith('.b')
    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor([Q], dtype=mean.dtype, device=mean.device).repeat(mean.shape[0])
    assert x.shape == mean.shape == scale.shape == Q.shape
    x_int_round = torch.round(x / Q)  # [100]
    max_value = x_int_round.max()
    min_value = x_int_round.min()
    samples = torch.tensor(range(int(min_value.item()), int(max_value.item()) + 1 + 1)).to(
        torch.float).to(x.device)  # from min_value to max_value+1. shape = [max_value+1+1 - min_value]
    samples = samples.unsqueeze(0).repeat(mean.shape[0], 1)  # [100, max_value+1+1 - min_value]
    mean = mean.unsqueeze(-1).repeat(1, samples.shape[-1])
    scale = scale.unsqueeze(-1).repeat(1, samples.shape[-1])
    GD = torch.distributions.normal.Normal(mean, scale)
    lower = GD.cdf((samples - 0.5) * Q.unsqueeze(-1))
    del samples
    del mean
    del scale
    del GD
    x_int_round_idx = (x_int_round - min_value).to(torch.int16)
    assert (x_int_round_idx.to(torch.int32) == x_int_round - min_value).all()
    # if x_int_round_idx.max() >= lower.shape[-1] - 1:  x_int_round_idx.max() exceed 65536 but to int6, that's why error
        # assert False

    if not use_multiprocessor:
        byte_stream = torchac.encode_float_cdf(lower.cpu(), x_int_round_idx.cpu(), check_input_bounds=True)
        if file_name is not None:
            with open(file_name, 'wb') as fout:
                fout.write(byte_stream)
        bit_len = len(byte_stream)*8
    else:
        bit_len = multiprocess_encoder(lower.cpu(), x_int_round_idx.cpu(), file_name)
    torch.cuda.empty_cache()
    return byte_stream, bit_len, min_value, max_value


def decoder_gaussian(mean, scale, Q, file_name=None, min_value=-100, max_value=100, bstream=None):
    if file_name is not None: assert file_name.endswith('.b')
    else: assert bstream is not None
    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor([Q], dtype=mean.dtype, device=mean.device).repeat(mean.shape[0])
    assert mean.shape == scale.shape == Q.shape
    samples = torch.tensor(range(min_value, max_value+ 1 + 1)).to(
        torch.float).to(mean.device)  # from min_value to max_value+1. shape = [max_value+1+1 - min_value]
    samples = samples.unsqueeze(0).repeat(mean.shape[0], 1)  # [100, max_value+1+1 - min_value]
    mean = mean.unsqueeze(-1).repeat(1, samples.shape[-1])
    scale = scale.unsqueeze(-1).repeat(1, samples.shape[-1])
    GD = torch.distributions.normal.Normal(mean, scale)
    lower = GD.cdf((samples - 0.5) * Q.unsqueeze(-1))
    if not use_multiprocessor:
        if file_name is not None:
            with open(file_name, 'rb') as fin:
                byte_stream_d = fin.read()
        else:
            byte_stream_d = bstream
        sym_out = torchac.decode_float_cdf(lower.cpu(), byte_stream_d).to(mean.device).to(torch.float32)
    else:
        sym_out = multiprocess_deoder(lower.cpu(), file_name, chunk_num=10).to(torch.float32)
    x = sym_out + min_value
    x = x * Q
    torch.cuda.empty_cache()
    return x


def encoder(x, p, file_name):
    x = x.detach().cpu()
    p = p.detach().cpu()
    assert file_name[-2:] == '.b'
    p_u = 1 - p.unsqueeze(-1)
    p_0 = torch.zeros_like(p_u)
    p_1 = torch.ones_like(p_u)
    # Encode to bytestream.
    output_cdf = torch.cat([p_0, p_u, p_1], dim=-1)
    sym = torch.floor(((x+1)/2)).to(torch.int16)
    byte_stream = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True)
    # Number of bits taken by the stream
    bit_len = len(byte_stream) * 8
    # Write to a file.
    with open(file_name, 'wb') as fout:
        fout.write(byte_stream)
    return bit_len

def decoder(p, file_name):
    dvc = p.device
    p = p.detach().cpu()
    assert file_name[-2:] == '.b'
    p_u = 1 - p.unsqueeze(-1)
    p_0 = torch.zeros_like(p_u)
    p_1 = torch.ones_like(p_u)
    # Encode to bytestream.
    output_cdf = torch.cat([p_0, p_u, p_1], dim=-1)
    # Read from a file.
    with open(file_name, 'rb') as fin:
        byte_stream = fin.read()
    # Decode from bytestream.
    sym_out = torchac.decode_float_cdf(output_cdf, byte_stream)
    sym_out = (sym_out * 2 - 1).to(torch.float32)
    return sym_out.to(dvc)


class STE_binary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = torch.clamp(input, min=-1, max=1)
        # out = torch.sign(input)
        p = (input >= 0) * (+1.0)
        n = (input < 0) * (-1.0)
        out = p + n
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # mask: to ensure x belongs to (-1, 1)
        input, = ctx.saved_tensors
        i2 = input.clone().detach()
        i3 = torch.clamp(i2, -1, 1)
        mask = (i3 == i2) + 0.0
        return grad_output * mask


class STE_multistep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, Q):
        if use_clamp:
            input_min = -15_000 * Q
            input_max = +15_000 * Q
            input = torch.clamp(input, min=input_min.detach(), max=input_max.detach())

        Q_round = torch.round(input / Q)
        Q_q = Q_round * Q
        return Q_q
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Quantize_anchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, anchors, min_v, max_v):
        interval = ((max_v - min_v) * Q_anchor + 1e-6)  # avoid 0, if max_v == min_v
        # quantized_v = (anchors - min_v) // interval
        quantized_v = torch.div(anchors - min_v, interval, rounding_mode='floor')
        quantized_v = torch.clamp(quantized_v, 0, 2 ** anchor_round_digits - 1)
        anchors_q = quantized_v * interval + min_v
        return anchors_q, quantized_v
    
    @staticmethod
    def backward(ctx, grad_output, tmp):  # tmp is for quantized_v:)
        return grad_output, None, None
