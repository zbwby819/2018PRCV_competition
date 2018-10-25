# import os
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import datetime

from functools import partial, update_wrapper

from keras import optimizers
from keras.utils import np_utils, generic_utils
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input, GlobalAvgPool2D, \
    Activation, LeakyReLU, BatchNormalization
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import RandomNormal
from keras.callbacks import ModelCheckpoint as checkpoint, ReduceLROnPlateau, \
    CSVLogger

from Baseline.aug import aug_nhw3
from Baseline.utils import *
from Baseline.reid_ops import triplet_hard_loss, \
    softmargin_triplet_hard_loss, evaluation_loss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def reid_net(identity_num, lamd=0.01, load_save=False):
    """
    Refering to strong reid baseline
    """
    # load pre-trained resnet50,
    base_model = ResNet50(weights='imagenet', include_top=False,
                          input_tensor=Input(shape=(224, 224, 3)))

    print("Backbone Model Loaded!")
    # build base net
    x = base_model.output

    # feature = GlobalAvgPool2D(name='feature_out')(x)

    feature = GlobalAvgPool2D(name='feature')(x)
    #
    # feature = Activation('tanh', name='feature_out')(feature)

    fc1 = Dense(512, kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
                # kernel_regularizer=l2(lamd),
                name='feature_out')(feature)

    bn1 = BatchNormalization()(fc1)

    leaky_relu = LeakyReLU(0.1)(bn1)

    fc2 = Dropout(0.5)(leaky_relu)

    preds = Dense(identity_num, activation='softmax', name='classification',
                  # kernel_regularizer=l2(lamd),
                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(
        fc2)  # default glorot_uniform

    net = Model(inputs=base_model.input, outputs=preds)

    # print(feature_model.summary())

    class_triplet_model = Model(inputs=base_model.input,
                                outputs=[preds, feature])

    feature_model = Model(inputs=base_model.input, outputs=feature)

    if load_save:
        feature_model.load_weights(
            os.path.join(PARENT_BASE_DIR, 'WEIGHT/Weights_best.h5'), by_name=True)

        # feature_model.load_weights(
        #     os.path.join(PARENT_BASE_DIR, 'WEIGHT/Weights.h5'), by_name=True)
        print("Weight Loaded!")

    return class_triplet_model, net, feature_model


def model_train(id_train, id_eval, SN, PN, identity_num, epoch=100,
                lr=0.001, loss_ratio=1, margin=0.2, lamd=0.01,
                train_epoch=10, sgd=True, lr_limit=1e-7, warmup=False,
                eval_interval=10, reduce_lr_round=2, fake_dim=2048, msml=False,
                aug=False, load_save=False):
    print('Loading model...')
    # datagen = ImageDataGenerator(horizontal_flip=True)

    # get network:
    class_triplet_model, net, feature_model = reid_net(
        identity_num=identity_num, lamd=lamd, load_save=load_save)

    # training IDE model for all layers
    for layer in net.layers:
        layer.trainable = True

    # choose optimizer
    if sgd:
        optimizer = optimizers.SGD(lr=lr, momentum=0.9, decay=0.0005)
    else:
        optimizer = optimizers.adam(lr)

    # net.compile(optimizer=adam, loss='categorical_crossentropy',
    #             metric='accuracy')

    # wrap triplet loss
    if margin is not None:
        new_triplet_hard_loss = partial(triplet_hard_loss, SN=SN, PN=PN,
                                        a1=margin, msml=msml)
        update_wrapper(new_triplet_hard_loss, triplet_hard_loss)
    else:
        new_triplet_hard_loss = partial(softmargin_triplet_hard_loss, SN=SN,
                                        PN=PN,
                                        lamd=lamd, msml=msml)
        update_wrapper(new_triplet_hard_loss, softmargin_triplet_hard_loss)

    # evaluation loss
    new_eval_loss = partial(evaluation_loss, SN=SN, PN=PN)
    update_wrapper(new_eval_loss, evaluation_loss)

    feature_model.compile(optimizer=optimizer, loss=new_eval_loss)

    class_triplet_model.compile(optimizer=optimizer,
                                loss=['categorical_crossentropy',
                                      new_triplet_hard_loss],
                                loss_weights=[1.0 - float(loss_ratio),
                                              float(loss_ratio)])
    print('Loading complete.')

    ##### preprocessing #####
    train_size = int(id_train.size / PN) * PN
    eval_size = int(id_eval.size / PN) * PN

    # for loss function create fake label
    fake_label_train = np.ones([train_size * SN, fake_dim])
    fake_label_eval = np.ones([eval_size * SN, fake_dim])

    # Callbacks
    now_time = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    # checkpointer = checkpoint('Weights.h5', monitor='val_loss',
    #                           save_best_only=True)
    # reduce_lr = ReduceLROnPlateau(monitor='feature_out_loss', factor=0.5,
    #                               patience=reduce_lr_round, min_lr=0.0001)
    train_log = CSVLogger(os.path.join(PARENT_BASE_DIR,
                                       'LOG/trainLog{}.csv'.format(now_time)),
                          append=True)
    eval_log = os.path.join(PARENT_BASE_DIR,
                            'LOG/evalLog{}.csv'.format(now_time))
    print('Preprocessing complete.')

    ##### train #####
    ind = 0
    eval_best = 0
    eval_loss_list = []
    while (ind < epoch):
        print('**********EPOCH: {} START.**********'.format(ind))
        # train_img,train_label = get_triplet_data(PN)  #the data in a batch: A1 B1 C1 ...PN1 A2 B2 C2 ... PN2 G K S ... Negative(PN)
        train_img, train_label = sample_img_batch(id_train, SN,
                                                  train_size)  # the data in a batch : A1 A2 A3... ASN B1 B2 B3... BSN ... PN1 PN2 PN3... PNSN

        # Normalize
        train_img = preprocess_input(train_img)
        # train_img /= 255

        train_label_onehot = np_utils.to_categorical(train_label, identity_num)

        # augmentation
        if aug:
            train_img = aug_nhw3(train_img)

        class_triplet_model.fit(train_img,
                                y=[train_label_onehot, fake_label_train],
                                shuffle=False, epochs=train_epoch,
                                batch_size=PN * SN,
                                callbacks=[train_log])

        print('**********EPOCH: {} END.**********'.format(ind))

        ind = ind + 1

        # reduce lr, gradually warmup
        if np.mod(ind, reduce_lr_round) == 0:
            # -- Harley [8/15/2018]
            current_lr = float(K.get_value(class_triplet_model.optimizer.lr))
            if warmup and (ind <= reduce_lr_round * 10):
                new_lr = current_lr + lr
                K.set_value(class_triplet_model.optimizer.lr, new_lr)
            elif (not sgd) and (current_lr > lr_limit):
                new_lr = current_lr * 0.8
                K.set_value(class_triplet_model.optimizer.lr, new_lr)

            current_lr = float(K.get_value(class_triplet_model.optimizer.lr))
            print('Current Learning Rate: {}.'.format(current_lr))

        # eval interval, also expand margin
        if np.mod(ind, eval_interval) == 0:
            eval_img, _ = sample_img_batch(id_eval, SN, eval_size)

            # Normalize
            eval_img = preprocess_input(eval_img)
            # eval_img /= 255

            eval_loss = feature_model.evaluate(eval_img, fake_label_eval,
                                               batch_size=SN * PN)
            print('**********EPOCH: {}, EVAL LOSS: {}.**********'.format(ind,
                                                                         eval_loss))
            eval_loss_list.append(eval_loss)
            if eval_loss > eval_best:
                eval_best = eval_loss
                class_triplet_model.save_weights(
                    os.path.join(PARENT_BASE_DIR, 'WEIGHT/Weights_best.h5'))
            else:
                class_triplet_model.save_weights(
                    os.path.join(PARENT_BASE_DIR, 'WEIGHT/Weights.h5'))

            ##### post processing #####
            eval_data = pd.DataFrame(np.array([[ind, eval_loss]]),
                                     columns=['epoch', 'eval_triplet_loss'])
            eval_data.to_csv(eval_log, mode='a', header=False)


def model_evaluate(id_train, id_eval, SN, PN, identity_num, epoch=100,
                   lr=0.001, loss_ratio=1, margin=0.2, lamd=0.01,
                   train_epoch=10, sgd=True, lr_limit=1e-7, warmup=False,
                   eval_interval=10, reduce_lr_round=2, fake_dim=2048,
                   msml=False,
                   aug=False, load_save=False):
    print('Loading model...')
    # datagen = ImageDataGenerator(horizontal_flip=True)

    # get network:
    class_triplet_model, net, feature_model = reid_net(
        identity_num=identity_num, lamd=lamd, load_save=load_save)

    # training IDE model for all layers
    for layer in net.layers:
        layer.trainable = True

    # choose optimizer
    if sgd:
        optimizer = optimizers.SGD(lr=lr, momentum=0.9, decay=0.0005)
    else:
        optimizer = optimizers.adam(lr)

    # evaluation loss
    new_eval_loss = partial(evaluation_loss, SN=SN, PN=PN)
    update_wrapper(new_eval_loss, evaluation_loss)

    feature_model.compile(optimizer=optimizer, loss=new_eval_loss)

    print('Loading complete.')

    ##### preprocessing #####
    eval_size = int(id_eval.size / PN) * PN

    # for loss function create fake label
    fake_label_eval = np.ones([eval_size * SN, fake_dim])

    print('Preprocessing complete.')

    eval_img, _ = sample_img_batch(id_eval, SN, eval_size)

    # Normalize
    eval_img = preprocess_input(eval_img)
    # eval_img /= 255

    eval_loss = feature_model.evaluate(eval_img, fake_label_eval,
                                       batch_size=SN * PN)
    print('**********EVAL LOSS: {}.**********'.format(eval_loss))


def model_predict(query_pic, gallery=None, identity_num=10, load_save=True, if_sort=False):
    """
    given query pic name, sort test pic according to similarity
    :param query_pic: path of query pic
    :param gallery: list of paths of test pics
    :param identity_num: how many identity in total
    :param load_save: if true will load saved weight
    :return:
    """
    # get network:
    _, _, feature_model = reid_net(identity_num, load_save=load_save)

    if gallery is not None:
        # process query and test:
        query_img = preprocess_input(image_read(query_pic))
        query_feature = feature_model.predict(query_img)
        test_imgs = preprocess_input(image_read(gallery))
        test_features = feature_model.predict(test_imgs)
        if if_sort:
            sorted_score, sorted_order = distance_order(query_feature, test_features)
        else:
            sorted_score, sorted_order = distance(query_feature, test_features)

        output = (sorted_score, sorted_order)

    else:
        query_img = preprocess_input(image_read(query_pic))
        query_feature = feature_model.predict(query_img)
        output = query_feature

    return output


if __name__ == '__main__':
    base_model = ResNet50(weights='imagenet', include_top=False,
                          input_tensor=Input(shape=(224, 224, 3)))
    print(base_model.summary())
