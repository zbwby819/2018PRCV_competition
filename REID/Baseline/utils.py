import os
import numpy as np
import pandas as pd
import numpy.linalg as la
import cv2
import h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_BASE_DIR = os.path.dirname(BASE_DIR)
OUTER_DIR = os.path.dirname(PARENT_BASE_DIR)
TRAIN_DIR = os.path.join(OUTER_DIR,'data/ReID/training_images/')
TEST_DIR = os.path.join(OUTER_DIR,'data/ReID/test_images/')

eg = '../data/ReID/training_images/RAP_REID_TRAIN_00001.png'


def rand_split_ids(percent_train):
    """
    divide ids into training set and eval set
    :param percent_train: percentage of train
    :return: training id array and eval id array
    """
    dataframe = pd.read_csv(os.path.join(BASE_DIR, 'training_images_details.csv'))
    unique_id = np.unique(dataframe['ID'].values.flatten())
    num_ids = unique_id.size
    randperm = np.random.permutation(range(num_ids))
    num_train = int(round(percent_train * num_ids))
    id_train = unique_id[randperm[0:num_train]]
    id_eval = unique_id[randperm[num_train:]]
    return (id_train, id_eval)


def specific_split_ids(certain_id):
    """
    divide ids into training set and eval set according to certain_id
    :return: training id array and eval id array
    """
    dataframe = pd.read_csv(os.path.join(BASE_DIR, 'training_images_details.csv'))
    unique_id = np.unique(dataframe['ID'].values.flatten())
    id_train = unique_id[unique_id<=certain_id]
    id_eval = unique_id[unique_id>certain_id]
    return (id_train, id_eval)


def sample_img_batch(id_train, SN, PN):
    """
    generate triplet img samples
    :param id_train: ndarray
    :param SN: int, imgs per id in sample
    :param PN: int, num of id in sample
    :return: img ndarrays and labels
    """
    dataframe = pd.read_csv(os.path.join(BASE_DIR, 'training_images_details.csv'))
    imgs = []
    labels = []
    randperm = np.random.permutation(range(id_train.size))
    # select PN ids
    selected_ids = id_train[randperm[:PN]]
    for id in selected_ids:
        img_names = dataframe[dataframe['ID']==id]['IMG'].values.flatten()
        if img_names.size < SN:
            # use mod here for situation that total pic less than SN
            randperm = np.mod(np.random.permutation(range(SN)), img_names.size)
        else:
            randperm = np.random.permutation(range(img_names.size))[:SN]
        # select SN imgs
        selected_img_names = img_names[randperm]
        selected_imgs = image_read(selected_img_names)
        imgs.append(selected_imgs)
        label = np.argwhere(id_train==id).flatten()[0]
        labels += [label] * SN

    # convert to array
    imgs = np.concatenate(imgs, axis=0)
    labels = np.array(labels)
    return imgs, np.array(labels)

def filename_transfer(ind, train=True):
    """
    transfer filename to paths
    :param ind: the index of pic
    :param train: boolean, if true will choose train
    :return: img path and file name
    """
    embed1 = 'training' if train else 'test'
    embed2 = 'TRAIN' if train else 'TEST'
    ind = str(ind)
    ind = '0' * (5 - len(ind)) + ind
    img_name = 'RAP_REID_{0}_{1}.png'.format(embed2, ind)
    img_path = os.path.join(OUTER_DIR,
                            'data/ReID/{}_images/'.format(
                                embed1) + img_name)
    return [img_path, img_name]


def retrieve_id_cam(img_name):
    """
    retrive id and cam from csv
    :param img_name: can be string or 1d array
    :return: id and cam, can be string or 1d arrays
    """
    df = pd.read_csv(os.path.join(BASE_DIR, 'training_images_details.csv'))
    if type(img_name) is not np.ndarray:
        retrieved_df = df[df['IMG'] == img_name]
        retrieved_id = retrieved_df['ID'].values[0]
        retrieved_cam = retrieved_df['CAM'].values[0]
    else:
        retrieved_id = []
        retrieved_cam = []
        for i in range(img_name.size):
            retrieved_df = df[df['IMG'] == img_name[i]]
            retrieved_id.append(retrieved_df['ID'].values[0])
            retrieved_cam.append(retrieved_df['CAM'].values[0])

        retrieved_id = np.array(retrieved_id).reshape(1, -1)
        retrieved_cam = np.array(retrieved_cam).reshape(1, -1)

    return retrieved_id, retrieved_cam

def distance_combined(query_feature, test_all):
    le = len(test_all)
    query_feature = np.tile(query_feature, (le, 1))

    sub = test_all - query_feature
    dis = la.norm(sub, axis=-1)

    return dis, np.arange(le)

def distance(query_feature, test_all):
    le = len(test_all)
    dis = np.zeros(le)
    for ind in range(le):
        sub = test_all[ind] - query_feature
        dis[ind] = la.norm(sub)

    return dis, np.arange(le)

def distance_order(query_feature, test_all):
    """
    return sort
    :param query_feature: 1-d array
    :param test_all: 2-d array
    :return: sorted distance array and a order index
    """
    le = len(test_all)
    dis = np.zeros(le)
    for ind in range(le):
        sub = test_all[ind] - query_feature
        dis[ind] = la.norm(sub)
    ii = sorted(range(len(dis)), key=lambda k: dis[k])
    return dis[ii], ii


def image_read(list_src, height=224, channel=3):
    """

    :param list_src: can be array or string
    :param height:
    :param channel:
    :return:
    """
    image_list = []
    if type(list_src) is not np.ndarray:
        image_list.append(square_image_read(list_src, height, channel))
    else:
        for src in list_src:
            img = square_image_read(src, height, channel)
            image_list.append(img)
    return np.stack(image_list)


def square_image_read(src, height=224, channel=3, train=True):
    """
    read image and resize it to a square shape
    :param src: the path of a picture
    :param height: specified height
    :return: 3d-array
    """

    if '/' not in src:
        src = TRAIN_DIR + src if train else TEST_DIR + src

    img = cv2.imread(src)

    (y, x) = img.shape[:-1]

    # for the case that the width greater than height
    if x > y:
        width =height

        new_height = int(y * height / x)

        if new_height % 2 != 0:
            new_height += 1

        res = cv2.resize(img, (width, new_height))

        top_append = np.zeros((int((height-new_height)/2), width, channel))

        bottom_append = np.zeros((int((height-new_height)/2), width, channel))

        res = np.concatenate((top_append, res, bottom_append), axis=0)

    else:
        width = int(x * height / y)

        if width % 2 != 0:
            width += 1

        res = cv2.resize(img, (width, height))

        left_append = np.zeros((height, int((height - width) / 2), channel))

        right_append = np.zeros((height, int((height - width) / 2), channel))

        res = np.concatenate((left_append, res, right_append), axis=1)

    return res


def print_keras_wegiths(weight_file_path):
    f = h5py.File(weight_file_path)

    g = f['fc8']

    for k0, v0 in g.attrs.items():
        print(k0, v0)
        for v1 in v0:
            print(v1, g[v1].shape)


if __name__ == '__main__':
    # print(square_image_read(eg).shape)
    # print_keras_wegiths(os.path.join(BASE_DIR, 'base_weights.h5'))
    # df = pd.read_csv(os.path.join(BASE_DIR, 'training_images_details.csv'))
    # image_read(df['IMG'].values.flatten())
    train_id = np.array([1,3,4,6,9,11,12])
    print(sample_img_batch(train_id,3,5)[1])
