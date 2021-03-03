import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class BasicModel(nn.Module):
    def __init__(self, inp_voc, out_voc, emb_size=64, hid_size=128):
        """
        A simple encoder-decoder seq2seq model
        """
        
        super().__init__()

        self.inp_voc, self.out_voc = inp_voc, out_voc
        self.hid_size = hid_size
        
        self.emb_inp = nn.Embedding(len(inp_voc), emb_size)
        self.emb_out = nn.Embedding(len(out_voc), emb_size)
        self.enc0 = nn.GRU(emb_size, hid_size, batch_first=True)

        self.dec_start = nn.Linear(hid_size, hid_size)
        self.dec0 = nn.GRUCell(emb_size, hid_size)
        self.logits = nn.Linear(hid_size, len(out_voc))
        
        
    def forward(self, inp, out):
        """ Apply model in training mode """
        
        initial_state = self.encode(inp)
        return self.decode(initial_state, out)


    def encode(self, inp, **flags):
        """ 
        Takes symbolic input sequence, computes initial state
        :param inp: matrix of input tokens [batch, time]
        :returns: initial decoder state tensors, one or many
        """
        
        inp_emb = self.emb_inp(inp)
        batch_size = inp.shape[0]
        
        enc_seq, [last_state_but_not_really] = self.enc0(inp_emb)
        # enc_seq: [batch, time, hid_size], last_state: [batch, hid_size]
        
        # last_state is not actually last because of padding, let's find the real last_state
        lengths = (inp != self.inp_voc.eos_ix).to(torch.int64).sum(dim=1).clamp_max(inp.shape[1] - 1)
        last_state = enc_seq[torch.arange(len(enc_seq)), lengths] # shape: [batch_size, hid_size]
        return [self.dec_start(last_state)]

    
    def decode_step(self, prev_state, prev_tokens, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits for next tokens
        :param prev_state: a list of previous decoder state tensors, same as returned by encode(...)
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch, len(out_voc)]
        """
        
        x = self.emb_out(prev_tokens)
        x = self.dec0(x, prev_state[0])
        return [x], self.logits(x)

    
    def decode(self, initial_state, out_tokens, **flags):
        """ Iterate over reference tokens (out_tokens) with decode_step """
        
        batch_size = out_tokens.shape[0]
        state = initial_state
        
        # initial logits: always predict BOS
        onehot_bos = F.one_hot(torch.full([batch_size], self.out_voc.bos_ix, dtype=torch.int64),
                               num_classes=len(self.out_voc)).to(device=out_tokens.device)
        first_logits = torch.log(onehot_bos.to(torch.float32) + 1e-9)
        
        logits_sequence = [first_logits]
        for i in range(out_tokens.shape[1] - 1):
            state, logits = self.decode_step(state, out_tokens[:, i])
            logits_sequence.append(logits)
        return torch.stack(logits_sequence, dim=1)

    def decode_inference(self, initial_state, max_len=100, **flags):
        """ Generate translations from model (greedy version) """
        
        batch_size, device = len(initial_state[0]), initial_state[0].device
        state = initial_state
        outputs = [torch.full([batch_size], self.out_voc.bos_ix, dtype=torch.int64, 
                              device=device)]
        all_states = [initial_state]

        for i in range(max_len):
            state, logits = self.decode_step(state, outputs[-1])
            outputs.append(logits.argmax(dim=-1))
            all_states.append(state) 
        return torch.stack(outputs, dim=1), all_states

    # def decode_inference_beam_search_slow(self, initial_state, beam_size, max_len=100, **flags):
    #     batch_size, device = len(initial_state[0]), initial_state[0].device
    #     outputs = []
    #     for batch_idx in range(batch_size):
    #         beam_seqs = [(self.out_voc.bos_ix,)]
    #         state = initial_state
    #         beam_states = [initial_state[0].unsqueeze(0)]
    #         beam_probs = [0]

    #         for _ in range(max_len):
    #             new_seqs = []
    #             states_history = []
    #             new_probs = []

    #             for seq, state, beam_prob in zip(beam_seqs, beam_states, beam_probs):
    #                 prev_tokens = torch.tensor(seq, dtype=torch.int64, device=device).unsqueeze(0)
    #                 new_state, logits = self.decode_step(state, prev_tokens)
    #                 states_history.append(new_state)
    #                 probs = torch.log_softmax(logits, dim=-1)[0].detach().cpu().numpy()
    #                 for idx in np.argpartition(probs, -beam_size)[-beam_size:]:
    #                     new_prob = beam_prob + probs[idx]
    #                     new_seqs.append(seq + (idx,))
    #                     new_probs.append(beam_prob + probs[idx])

    #             beam_seqs, beam_states, beam_probs = [], [], []
    #             for idx in np.argsort(new_probs)[-beam_size:]:
    #                 beam_seqs.append(new_seqs[idx])
    #                 beam_states.append(states_history[idx % beam_size])
    #                 beam_probs.append(new_probs[idx])

    #         outputs.append(beam_seqs[-1])

    #     return outputs, None
    
    def decode_inference_beam_search(self, initial_state, beam_size, max_len=100, **flags):
        batch_size, device = len(initial_state[0]), initial_state[0].device
        state = initial_state

        outputs = [[(self.out_voc.bos_ix,)] * batch_size]
        probs = np.zeros(shape=(beam_size, batch_size))
        states = [deepcopy(initial_state) for _ in range(beam_size)]
        hypos = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            next_beams = [[] for _ in range(batch_size)] 
            states_history = []
            for i in range(len(outputs)):
                prev_tokens = torch.tensor([tokens[-1] for tokens in outputs[i]], device=device)
                # prev_tokens = (prev_tokens, dtype=torch.int64, )

                cur_states, logits = self.decode_step(states[i], prev_tokens)
                logits = torch.log_softmax(logits, dim=-1).detach().cpu().numpy()
                states_history.append(cur_states)

                for b, logit in enumerate(logits):
                    for idx in np.argpartition(logit, -beam_size)[-beam_size:]:
                        if idx == 1 and np.exp(logit[idx]) < 0.6:
                            idx = 228
                        next_beams[b].append([outputs[i][b] + (idx,), logit[idx] + probs[i, b], i])
         
            outputs = [[None] * batch_size for _ in range(beam_size)]
            for i in range(batch_size):
                next_beams[i].sort(key=lambda x: x[1], reverse=True)
                for j in range(beam_size):
                    outputs[j][i], probs[j, i], beam_idx = next_beams[i][j]
                    if outputs[j][i][-1] == 1:
                        hypos[i].append([probs[j, i] + 0.1 * _, outputs[j][i]])
                    states[j][0][i] = states_history[beam_idx][0][i]

        for i in range(len(hypos)):
            if not hypos[i]:
                hypos[i].append([0, outputs[0]])
            hypos[i].sort()
        return [hypo[-1][1] for hypo in hypos]

    def translate_lines(self, inp_lines, device, beam_size=None, **kwargs):
        inp = self.inp_voc.to_matrix(inp_lines).to(device)
        initial_state = self.encode(inp)
        if beam_size is None:
            out_ids, states = self.decode_inference(initial_state, **kwargs)
        else:
            out_ids, states = self.decode_inference_beam_search(initial_state, beam_size, **kwargs), None
        return self.out_voc.to_lines(out_ids), states
  

class AttentionLayer(nn.Module):
    def __init__(self, enc_size, dec_size, hid_size, activ=torch.tanh):
        """ A layer that computes additive attention response and weights """
        
        super().__init__()
        
        self.enc_size = enc_size # num units in encoder state
        self.dec_size = dec_size # num units in decoder state
        self.hid_size = hid_size # attention layer hidden units
        self.activ = activ       # attention layer hidden nonlinearity
        
        self.linear1 = nn.Linear(enc_size, hid_size)
        self.linear2 = nn.Linear(dec_size, hid_size)
        self.linear3 = nn.Linear(hid_size, 1)
        self.soft = nn.Softmax(dim=-1)

        
    def forward(self, enc, dec, inp_mask):
        """
        Computes attention response and weights
        :param enc: encoder activation sequence, float32[batch_size, ninp, enc_size]
        :param dec: single decoder state used as "query", float32[batch_size, dec_size]
        :param inp_mask: mask on enc activatons (0 after first eos), float32 [batch_size, ninp]
        :returns: attn[batch_size, enc_size], probs[batch_size, ninp]
            - attn - attention response vector (weighted sum of enc)
            - probs - attention weights after softmax
        """
        
        batch_size, n_inp, enc_size = enc.shape

        tmp = self.linear2(dec)
        tmp = tmp.reshape(-1, 1, self.hid_size)
        
        x = self.linear1(enc)
        x = self.activ(x + tmp)
        x = self.linear3(x)

        # Apply mask - if mask is 0, logits should be -inf or -1e9
        x[torch.where(inp_mask == False)] = -1e9

        # Compute attention probabilities (softmax)
        probs = self.soft(x.reshape(batch_size, n_inp))

        # Compute attention response using enc and probs
        attn = (probs.reshape(batch_size, n_inp, 1) * enc).sum(axis=1)
        
        assert tuple(attn.shape) == (batch_size, enc_size)
        assert tuple(probs.shape) == (batch_size, n_inp)

        return attn, probs
    
    
class AttentiveModel(BasicModel):
    def __init__(self, inp_voc, out_voc, emb_size=256, hid_size=256, attn_size=256):
        """ 
        Translation model that uses attention. See instructions above. 
        """
        
        super().__init__(inp_voc, out_voc, emb_size, hid_size)
    
        self.dec0 = nn.GRUCell(emb_size + hid_size, hid_size)
        self.attn = AttentionLayer(hid_size, hid_size, attn_size)


    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: matrix of input tokens [batch, time]
        :return: a list of initial decoder state tensors
        """       

        inp_emb = self.emb_inp(inp)
        
        enc_seq, [last_state_but_not_really] = self.enc0(inp_emb)
        
        [dec_start] = super().encode(inp, **flags)    
        
        enc_mask = self.out_voc.compute_mask(inp)
        
        # apply attention layer from initial decoder hidden state
        first_attn = self.attn(enc_seq, dec_start, enc_mask)[1]
        
        # [initial states for decoder recurrent layers,
        #  encoder sequence,
        #  encoder attn mask (for attention),
        #  attention probabilities tensor]
        
        return [dec_start, enc_seq, enc_mask, first_attn]
            
   
    def decode_step(self, prev_state, prev_tokens, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits for next tokens
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch, n_tokens]
        """
        
        prev_gru0_state, enc_seq, enc_mask, _ = prev_state
        attn, attn_prob = self.attn(enc_seq, prev_gru0_state, enc_mask)
        
        x = self.emb_out(prev_tokens)
        assert len(x.shape) == 2 and len(attn.shape) == 2
        
        x = torch.cat([attn, x], dim=-1)
        x = self.dec0(x, prev_gru0_state)
        
        new_dec_state = [x, enc_seq, enc_mask, attn_prob]
        output_logits = self.logits(x)
        
        return new_dec_state, output_logits
