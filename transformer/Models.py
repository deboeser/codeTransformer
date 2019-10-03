''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"


def get_non_pad_mask(seq):
    # assert seq.dim() == 3
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask_from_mask(mask_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = mask_k
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(2)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(2).expand(-1, -1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_attn_key_pad_mask_enc(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(
        0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_terminal_vocab, n_path_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.src_terminal_emb = nn.Embedding(
            n_terminal_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.src_path_emb = nn.Embedding(
            n_path_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, starts, paths, ends, starts_pos, paths_pos, ends_pos, return_attns=False):

        enc_slf_attn_list = []

        # TODO: Consider learned section embeddings

        starts_emb = self.src_terminal_emb(starts) + self.position_enc(starts_pos)
        paths_emb = self.src_path_emb(paths) + self.position_enc(paths_pos)
        ends_emb = self.src_terminal_emb(ends) + self.position_enc(ends_pos)

        src_seq = torch.cat([starts, paths, ends], dim=2)

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = torch.cat((starts_emb, paths_emb, ends_emb), 2)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_target_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_target_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_mask, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask_enc(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq.bool()).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask_from_mask(mask_k=src_mask, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output.unsqueeze(1), enc_output.unsqueeze(1),
                non_pad_mask=non_pad_mask.unsqueeze(1),
                slf_attn_mask=slf_attn_mask.unsqueeze(1),
                dec_enc_attn_mask=dec_enc_attn_mask.unsqueeze(1))
            dec_output = dec_output.squeeze(1)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_terminal_vocab, n_path_vocab, n_target_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True):

        super().__init__()

        self.encoder = Encoder(
            n_terminal_vocab=n_terminal_vocab, n_path_vocab=n_path_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_target_vocab=n_target_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.global_token_attn = nn.Linear(d_word_vec, 1, bias=False)
        self.token_attn_softmax = nn.Softmax(dim=2)

        self.global_context_attn = nn.Linear(d_word_vec, 1, bias=False)
        self.context_attn_softmax = nn.Softmax(dim=1)

        self.tgt_word_prj = nn.Linear(d_model, n_target_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self,
                starts, paths, ends,
                starts_pos, paths_pos, ends_pos,
                targets, targets_pos):

        targets, targets_pos = targets[:, :-1], targets_pos[:, :-1]

        enc_output, *_ = self.encoder(starts, paths, ends, starts_pos, paths_pos, ends_pos)

        b, a, l, _ = enc_output.size()

        src_seq = torch.cat([starts, paths, ends], dim=2)
        src_mask = torch.sum(src_seq.ne(Constants.PAD), dim=1).eq(0)
        token_pad_mask = src_seq.eq(Constants.PAD)
        context_pad_mask_context = starts[:, :, 0].eq(0)
        context_pad_mask_token = context_pad_mask_context.unsqueeze(-1).repeat(1, 1, 32)

        token_attn = self.global_token_attn(enc_output).squeeze(-1)
        token_attn = token_attn.masked_fill(token_pad_mask, -np.inf)
        token_attn = self.token_attn_softmax(token_attn)
        token_attn = token_attn.masked_fill(context_pad_mask_token, 0.)

        context_attn = token_attn.unsqueeze(-1) * enc_output
        context_attn = torch.sum(context_attn, dim=2)

        context_attn = self.global_context_attn(context_attn).squeeze(-1)
        context_attn = context_attn.masked_fill(context_pad_mask_context, -np.inf)
        context_attn = self.context_attn_softmax(context_attn)

        dec_input = enc_output.view(b, a, -1, 1).squeeze(-1) * context_attn.unsqueeze(-1)
        dec_input = dec_input.view(b, a, l, -1)
        dec_input = torch.sum(dec_input, dim=1)

        # TODO: Attend over all path contexts in encoder output

        dec_output, *_ = self.decoder(targets, targets_pos, src_mask, dec_input)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))
