import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from Baseline.new_model import *

def single_query(query_ind):
    query_list = []
    with open(os.path.join(OUTER_DIR,
                           'data/ReID/query_test_image_name.txt')) as f:
        line = f.readline()
        while line:
            query_list.append(os.path.join(TEST_DIR, line.strip()))
            line = f.readline()

    test_ind = range(13460)
    if_train = False
    test_pics_set = np.array(
        [filename_transfer(i + 1, train=if_train) for i in test_ind])
    test_pics_paths = test_pics_set[:, 0]

    # calculate features
    sorted_score, sorted_order = model_predict(query_list[query_ind], test_pics_paths, if_sort=True)
    sorted_path = test_pics_paths[sorted_order]
    print(sorted_path[:10])

def generate_test_output():

    query_list = []
    with open(os.path.join(OUTER_DIR, 'data/ReID/query_test_image_name.txt')) as f:
        line = f.readline()
        while line:
            query_list.append(os.path.join(TEST_DIR, line.strip()))
            line = f.readline()

    test_ind = range(13460)
    if_train = False
    test_pics_set = np.array(
        [filename_transfer(i+1, train=if_train) for i in test_ind])
    test_pics_paths = test_pics_set[:, 0]

    # calculate features
    query_num = len(query_list)
    all_features = model_predict(np.concatenate((np.array(query_list), test_pics_paths)))
    query_features = all_features[:query_num]
    gallery_feautures = all_features[query_num:]
    print('features generated!')

    # generate outputs
    data_all = []

    for i in range(len(query_list)):
        print('row {}'.format(i))

        start = np.array([i,])

        scores, indexes = distance_combined(query_features[i], gallery_feautures)

        combined = np.stack((indexes, scores), axis=-1).flatten()

        combined = np.concatenate((start, combined))

        data_all.append(combined)

    data_all = np.stack(data_all)
    df = pd.DataFrame(data_all)
    df.to_csv('result/Final_Output.csv', index=False, header=None)



if __name__ == '__main__':
    generate_test_output()
    # single_query(0)