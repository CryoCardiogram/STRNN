import numpy as np
import math
import torch
import torch.nn as nn

class STRNN(nn.Module):
    def __init__(self, hidden_size, loc_cnt, user_cnt):
        super(STRNN, self).__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(hidden_size, hidden_size)) # C
        self.weight_th_upper = nn.Parameter(torch.randn(hidden_size, hidden_size)) # T
        self.weight_th_lower = nn.Parameter(torch.randn(hidden_size, hidden_size)) # T
        self.weight_sh_upper = nn.Parameter(torch.randn(hidden_size, hidden_size)) # S
        self.weight_sh_lower = nn.Parameter(torch.randn(hidden_size, hidden_size)) # S

        self.location_weight = nn.Embedding(loc_cnt, hidden_size)
        self.permanet_weight = nn.Embedding(user_cnt, hidden_size)

        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, td_upper, td_lower, ld_upper, ld_lower, loc, hx):
        loc_len = len(loc)
        Ttd = [((self.weight_th_upper*td_upper[i] + self.weight_th_lower*td_lower[i])\
                / (td_upper[i]+td_lower[i])) for i in range(loc_len)]
        Sld = [((self.weight_sh_upper*ld_upper[i] + self.weight_sh_lower*ld_lower[i])\
                / (ld_upper[i]+ld_lower[i])) for i in range(loc_len)]

        loc = self.location_weight(loc).view(-1, self.hidden_size, 1)
        loc_vec = torch.sum(torch.cat(tuple(torch.mm(Sld[i], torch.mm(Ttd[i], loc[i])) for i in range(loc_len))))
        usr_vec = torch.mm(self.weight_ih, hx)
        hx = torch.add(loc_vec, usr_vec)  # hidden_size x 1
        return self.sigmoid(hx)

    def loss(self, user, td_upper, td_lower, ld_upper, ld_lower, loc, dst, hx):
        h_tq = self.forward(td_upper, td_lower, ld_upper, ld_lower, loc, hx)
        p_u = self.permanet_weight(user)
        q_v = self.location_weight(dst)
        output = torch.mm(q_v, (torch.add(h_tq, torch.t(p_u))))

        return torch.log(torch.add(torch.exp(torch.neg(output)), 1))

    def validation(self, user, td_upper, td_lower, ld_upper, ld_lower, loc, dst, hx):
        # error exist in distance (ld_upper, ld_lower)
        h_tq = self.forward(td_upper, td_lower, ld_upper, ld_lower, loc, hx)
        p_u = self.permanet_weight(user)
        user_vector = torch.add(h_tq, torch.t(p_u))
        ret = torch.mm(self.location_weight.weight, user_vector).data.cpu().numpy()
        return np.argsort(np.squeeze(-1*ret))
