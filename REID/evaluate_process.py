import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from Baseline.new_model import *

def evaluate_reid_ap_cmc(query_pid, query_cam, gallery_pids, gallery_cams, seperate_cam=False):
    """
    Input:
        query_pid: scalar value
        query_cam: scalar value
        gallery_pids: a 1xN ndarray
        gallery_cams: a 1xN ndarray
        seperate_cam: bool value, represent whether to use cross camera test or not
            when it is set as False, the different identities in the same camera of
            the query id are also used as effective samples in the gallery set. This
            could enlarge the gallery set and  the ap will decrease.
    Output:
        return a ap scalar and a cmc ndarray with shape 1*N
    """
    assert type(gallery_pids) == np.ndarray
    assert type(gallery_cams) == np.ndarray
    assert gallery_pids.shape[0] == 1 and len(gallery_pids.shape) == 2
    assert gallery_cams.shape[0] == 1 and len(gallery_cams.shape) == 2

    cmc = np.zeros(gallery_pids.shape)
    ngood = np.sum((query_pid == gallery_pids) & (query_cam != gallery_cams))
    ap = 0.0
    good_now = 1.0
    current_rank = 1
    first_flag = 0
    for n, (p, c) in enumerate(zip(gallery_pids[0, :], gallery_cams[0, :])):
        if good_now == ngood + 1:
            break
        # handle junk images
        if seperate_cam:
            junk = c == query_cam
        else:
            junk = p == query_pid and c == query_cam
        if junk:
            continue
        # compute the ap and cmc
        if p == query_pid and c != query_cam:
            # compute the average precision
            ap = ap + good_now/current_rank
            # compute the cmc curve
            if first_flag == 0:
                cmc[0, current_rank-1:] = 1
                first_flag = 1
            good_now = good_now + 1
        current_rank = current_rank + 1
    ap = ap/ngood
    return ap, cmc


def naive_evaluate():
    # setting
    # SN = 3
    # PN = 24

    certain_id = 2000
    id_train, id_eval = specific_split_ids(certain_id)
    identity_num = id_train.size

    # identity_num = 2589
    # specify the img number of query and gallery
    query_ind = 11315
    test_ind = range(11278, 12000)
    # specify if_train, if it's true, all data are from training imgs, else from test imgs
    if_train = True

    ###########evaluate###########
    query_pic, query_name = filename_transfer(query_ind, train=if_train)
    test_pics_set = np.array([filename_transfer(i, train=if_train) for i in test_ind])
    test_pics_paths = test_pics_set[:, 0]
    test_pics_names = test_pics_set[:, 1]
    sorted_order, sorted_score = model_predict(query_pic, test_pics_paths,
                                               identity_num, load_save=True, if_sort=True)
    sorted_names = test_pics_names[sorted_order]

    query_pid, query_cam = retrieve_id_cam(query_name)

    gallery_pids, gallery_cams = retrieve_id_cam(sorted_names)

    print (query_pid, query_cam)
    print (gallery_pids, gallery_cams)

    ap, cmc = evaluate_reid_ap_cmc(query_pid, query_cam, gallery_pids, gallery_cams, seperate_cam=False)

    print('AP: {}'.format(ap))

    print('CMC: {}'.format(cmc))

def eval_21_8():
    SN = 6
    PN = 15
    # for process
    aug = False
    epoch = 10000
    lr = 0.0001
    sgd = False
    train_epoch = 1
    eval_interval = 10
    reduce_lr_round = 10
    lr_limit = 1e-7
    warmup = True
    load_save = True
    # for loss
    loss_ratio = 0.7
    margin = None
    lamd = 0.01
    msml = False
    fake_dim = 2048

    # data handle
    train_perc = 0.9
    certain_id = 2000

    # process data
    # id_train, id_eval = rand_split_ids(train_perc)
    id_train, id_eval = specific_split_ids(certain_id)
    identity_num = id_train.size

    model_evaluate(id_train=id_train, id_eval=id_eval, SN=SN, PN=PN,
              identity_num=identity_num, epoch=epoch, lr=lr,
              loss_ratio=loss_ratio, margin=margin, msml=msml, aug=aug,
              lamd=lamd, sgd=sgd, lr_limit=lr_limit, warmup=warmup,
              fake_dim=fake_dim,
              train_epoch=train_epoch, eval_interval=eval_interval,
              reduce_lr_round=reduce_lr_round, load_save=load_save)

if __name__ == '__main__':
    eval_21_8()