import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from Baseline.utils import rand_split_ids
from Baseline.new_model import model_train as new_train

def training_once():
    # log: 917, 86% map, only horizontal_flip
    #
    SN = 3
    PN = 25
    # for process
    aug = True
    epoch = 100
    lr = 0.0002
    sgd = False
    train_epoch = 5
    eval_interval = 2
    reduce_lr_round = 2
    lr_limit = 1e-7
    warmup = True
    load_save = False
    # for loss
    loss_ratio = 0.5
    margin = None
    lamd = 0.01
    msml = False
    fake_dim = 2048

    # data handle
    train_perc = 0.9
    # certain_id = 2000

    # process data
    id_train, id_eval = rand_split_ids(train_perc)
    # id_train, id_eval = specific_split_ids(certain_id)
    identity_num = id_train.size

    new_train(id_train=id_train, id_eval=id_eval, SN=SN, PN=PN,
              identity_num=identity_num, epoch=epoch, lr=lr,
              loss_ratio=loss_ratio, margin=margin, msml=msml, aug=aug,
              lamd=lamd, sgd=sgd, lr_limit=lr_limit, warmup=warmup,
              fake_dim=fake_dim,
              train_epoch=train_epoch, eval_interval=eval_interval,
              reduce_lr_round=reduce_lr_round, load_save=load_save)

def training_whole():
    # for process
    aug = True
    sgd = False
    lr_limit = 1e-7
    warmup = True
    # for loss
    margin = None
    lamd = 0.01
    msml = False
    fake_dim = 2048

    ############ ITERATION 1 ###########
    SN = 3
    PN = 25
    lr = 0.0002
    epoch = 100
    train_epoch = 5
    loss_ratio = 0.5
    eval_interval = 2
    reduce_lr_round = 2
    # data handle
    train_perc = 0.8
    # process data
    id_train, id_eval = rand_split_ids(train_perc)
    identity_num = id_train.size
    load_save = False

    new_train(id_train=id_train, id_eval=id_eval, SN=SN, PN=PN,
              identity_num=identity_num, epoch=epoch, lr=lr,
              loss_ratio=loss_ratio, margin=margin, msml=msml, aug=aug,
              lamd=lamd, sgd=sgd, lr_limit=lr_limit, warmup=warmup,
              fake_dim=fake_dim,
              train_epoch=train_epoch, eval_interval=eval_interval,
              reduce_lr_round=reduce_lr_round, load_save=load_save)

    ############ ITERATION 2 ###########
    SN = 3
    PN = 25
    lr = 0.0001
    epoch = 50
    train_epoch = 1
    loss_ratio = 0.6
    eval_interval = 5
    reduce_lr_round = 1
    # data handle
    train_perc = 0.5
    # process data
    id_train, id_eval = rand_split_ids(train_perc)
    id_train_rev, id_eval_rev = id_eval, id_train
    identity_num = id_train.size
    identity_num_rev = id_train_rev.size
    load_save = True

    new_train(id_train=id_train, id_eval=id_eval, SN=SN, PN=PN,
              identity_num=identity_num, epoch=epoch, lr=lr,
              loss_ratio=loss_ratio, margin=margin, msml=msml, aug=aug,
              lamd=lamd, sgd=sgd, lr_limit=lr_limit, warmup=warmup,
              fake_dim=fake_dim,
              train_epoch=train_epoch, eval_interval=eval_interval,
              reduce_lr_round=reduce_lr_round, load_save=load_save)

    new_train(id_train=id_train_rev, id_eval=id_eval_rev, SN=SN, PN=PN,
              identity_num=identity_num_rev, epoch=epoch, lr=lr,
              loss_ratio=loss_ratio, margin=margin, msml=msml, aug=aug,
              lamd=lamd, sgd=sgd, lr_limit=lr_limit, warmup=warmup,
              fake_dim=fake_dim,
              train_epoch=train_epoch, eval_interval=eval_interval,
              reduce_lr_round=reduce_lr_round, load_save=load_save)

if __name__ == '__main__':
    training_once()
