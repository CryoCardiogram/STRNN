#! /usr/bin/env python

import os
import datetime
import math
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.autograd import Variable
import data_loader

# Parameters
# ==================================================
ftype = torch.FloatTensor
ltype = torch.LongTensor

# Data loading params
train_file = "./prepro_train_50.txt"
valid_file = "./prepro_valid_50.txt"
test_file = "./prepro_test_50.txt"

# Model Hyperparameters
dim = 13    # dimensionality
ww = 360  # winodw width (6h)
up_time = 560632.0  # min
lw_time = 0.
up_dist = 457.335   # km
lw_dist = 0.
reg_lambda = 0.1

# Training Parameters
batch_size = 2
num_epochs = 30
learning_rate = 0.001
momentum = 0.9
evaluate_every = 1
h_0 = Variable(torch.randn(dim, 1), requires_grad=False).type(ftype)

#user_cnt = 32899 #50 #107092#0
#loc_cnt = 1115406 #50 #1280969#0
#user_cnt = 42242 #30
#loc_cnt = 1164559 #30

try:
    xrange
except NameError:
    xrange = range

# Data Preparation
# ===========================================================
# Load data
print("Loading data...")
train_user, train_td, train_ld, train_loc, train_dst = data_loader.treat_prepro(train_file, step=1)
valid_user, valid_td, valid_ld, valid_loc, valid_dst = data_loader.treat_prepro(valid_file, step=2)
test_user, test_td, test_ld, test_loc, test_dst = data_loader.treat_prepro(test_file, step=3)

user_cnt = len(train_user) + len(valid_user) + len(test_user)
loc_cnt = sum(len(i) for i in train_loc) + sum(len(i) for i in valid_loc) + sum(len(i) for i in test_loc)

print("User/Location: {:d}/{:d}".format(user_cnt, loc_cnt))
print("==================================================================================")

class STRNNCell(nn.Module):
    def __init__(self, hidden_size):
        super(STRNNCell, self).__init__()
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
                / (td_upper[i]+td_lower[i])) for i in xrange(loc_len)]
        Sld = [((self.weight_sh_upper*ld_upper[i] + self.weight_sh_lower*ld_lower[i])\
                / (ld_upper[i]+ld_lower[i])) for i in xrange(loc_len)]

        loc = self.location_weight(loc).view(-1, self.hidden_size, 1)
        loc_vec = torch.sum(torch.cat(tuple(torch.mm(Sld[i], torch.mm(Ttd[i], loc[i])) for i in xrange(loc_len)) ))
        usr_vec = torch.mm(self.weight_ih, hx)
        hx = loc_vec + usr_vec # hidden_size x 1
        return self.sigmoid(hx)

    def loss(self, user, td_upper, td_lower, ld_upper, ld_lower, loc, dst, hx):
        h_tq = self.forward(td_upper, td_lower, ld_upper, ld_lower, loc, hx)
        p_u = self.permanet_weight(user)
        q_v = self.location_weight(dst)
        output = torch.mm(q_v, (h_tq + torch.t(p_u)))

        return torch.log(1+torch.exp(torch.neg(output)))

    def validation(self, user, td_upper, td_lower, ld_upper, ld_lower, loc, dst, hx):
        # error exist in distance (ld_upper, ld_lower)
        h_tq = self.forward(td_upper, td_lower, ld_upper, ld_lower, loc, hx)
        p_u = self.permanet_weight(user)
        user_vector = h_tq + torch.t(p_u)
        ret = torch.mm(self.location_weight.weight, user_vector).data.cpu().numpy()
        return np.argsort(np.squeeze(-1*ret))

###############################################################################################
def parameters(strnn_model):
    params = []
    for model in [strnn_model]:
        params += list(model.parameters())

    return params


def print_score(batches, step):
    recall1 = 0.
    recall5 = 0.
    recall10 = 0.
    recall100 = 0.
    recall1000 = 0.
    recall10000 = 0.
    iter_cnt = 0

    for batch in tqdm.tqdm(batches, desc="validation"):
        batch_user, batch_td, batch_ld, batch_loc, batch_dst = batch
        if len(batch_loc) < 3:
            continue
        iter_cnt += 1
        batch_o, target = run(batch_user, batch_td, batch_ld, batch_loc, batch_dst, step=step)

        recall1 += target in batch_o[:1]
        recall5 += target in batch_o[:5]
        recall10 += target in batch_o[:10]
        recall100 += target in batch_o[:100]
        recall1000 += target in batch_o[:1000]
        recall10000 += target in batch_o[:10000]

    print("recall@1: ", recall1/iter_cnt)
    print("recall@5: ", recall5/iter_cnt)
    print("recall@10: ", recall10/iter_cnt)
    print("recall@100: ", recall100/iter_cnt)
    print("recall@1000: ", recall1000/iter_cnt)
    print("recall@10000: ", recall10000/iter_cnt)

###############################################################################################


def run(user, td, ld, loc, dst, step):

    optimizer.zero_grad()

    seqlen = len(td)
    user = Variable(torch.from_numpy(np.asarray([user]))).type(ltype)

    #neg_loc = Variable(torch.FloatTensor(1).uniform_(0, len(poi2pos)-1).long()).type(ltype)
    #(neg_lati, neg_longi) = poi2pos.get(neg_loc.data.cpu().numpy()[0])
    rnn_output = h_0
    for idx in xrange(seqlen-1):
        td_upper = Variable(torch.from_numpy(np.asarray(up_time-td[idx]))).type(ftype)
        td_lower = Variable(torch.from_numpy(np.asarray(td[idx]-lw_time))).type(ftype)
        ld_upper = Variable(torch.from_numpy(np.asarray(up_dist-ld[idx]))).type(ftype)
        ld_lower = Variable(torch.from_numpy(np.asarray(ld[idx]-lw_dist))).type(ftype)
        location = Variable(torch.from_numpy(np.asarray(loc[idx]))).type(ltype)
        rnn_output = strnn_model(td_upper, td_lower, ld_upper, ld_lower, location, rnn_output)#, neg_lati, neg_longi, neg_loc, step)

    td_upper = Variable(torch.from_numpy(np.asarray(up_time-td[-1]))).type(ftype)
    td_lower = Variable(torch.from_numpy(np.asarray(td[-1]-lw_time))).type(ftype)
    ld_upper = Variable(torch.from_numpy(np.asarray(up_dist-ld[-1]))).type(ftype)
    ld_lower = Variable(torch.from_numpy(np.asarray(ld[-1]-lw_dist))).type(ftype)
    location = Variable(torch.from_numpy(np.asarray(loc[-1]))).type(ltype)

    if step > 1:
        return strnn_model.validation(user, td_upper, td_lower, ld_upper, ld_lower, location, dst[-1], rnn_output), dst[-1]

    destination = Variable(torch.from_numpy(np.asarray([dst[-1]]))).type(ltype)
    J = strnn_model.loss(user, td_upper, td_lower, ld_upper, ld_lower, location, destination, rnn_output)#, neg_lati, neg_longi, neg_loc, step)

    J.backward()
    optimizer.step()

    return J.data.cpu().numpy()

###############################################################################################


def train(model):

    loss_per_epoch = {}

    first_pass = False
    # load data from checkpoint
    try:
        checkpoint = torch.load('checkpoint.tar')
        loss_per_epoch = checkpoint['loss_per_epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['model_state_dict'])
        model.train()
        first_pass = True
    except FileNotFoundError:
        checkpoint = None

    for i in xrange(num_epochs if checkpoint is None else num_epochs - checkpoint['epoch']):
        # Training
        total_loss = 0.

        if first_pass and checkpoint is not None:
            # restore currrent total loss
            total_loss = checkpoint['current_loss']
            first_pass = False

        train_batches = list(zip(train_user, train_td, train_ld, train_loc, train_dst))
        try:
            for j, train_batch in enumerate(tqdm.tqdm(train_batches, desc="train")):
                # inner_batches = data_loader.inner_iter(train_batch, batch_size)
                # for k, inner_batch in inner_batches:
                batch_user, batch_td, batch_ld, batch_loc, batch_dst = train_batch  # inner_batch)
                if len(batch_loc) < 3:
                    continue
                loss = run(batch_user, batch_td, batch_ld, batch_loc, batch_dst, step=1)
                total_loss += loss
                # if (j+1) % 2000 == 0:
                #    print("batch #{:d}: ".format(j+1)), "batch_loss :", total_loss/j, datetime.datetime.now()
            # Evaluation
            if (i + 1) % evaluate_every == 0:
                print("==================================================================================")
                # print("Evaluation at epoch #{:d}: ".format(i+1)), total_loss/j, datetime.datetime.now()
                valid_batches = list(zip(valid_user, valid_td, valid_ld, valid_loc, valid_dst))
                print_score(valid_batches, step=2)

            loss_per_epoch[i] = total_loss
        except KeyboardInterrupt:
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'current_loss': total_loss,
                'loss_per_epoch': loss_per_epoch
            }, "checkpoint.tar")
            raise KeyboardInterrupt()


###############################################################################################


def multiproc_train(model, num_proc):
    """
    Train the model across `num_proc` processes
    :param model: torch model instance
    :param optimizer: torch optimizer instance
    :param: num_proc: number of processes
    :return:
    """
    model.share_memory()
    processes = []
    # launch the processes
    print("launching processes...")
    for rank in range(num_proc):
        p = mp.Process(target=train, args=(model,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()





if __name__ == '__main__':
    strnn_model = STRNNCell(dim)
    optimizer = optim.SGD(parameters(strnn_model), lr=learning_rate, momentum=momentum, weight_decay=reg_lambda)
    num_processes = torch.get_num_threads()
    train(strnn_model)


    # Testing
    print("Training End..")
    print("==================================================================================")
    print("Test: ")
    test_batches = list(zip(test_user, test_td, test_ld, test_loc, test_dst))
    print_score(test_batches, step=3)
