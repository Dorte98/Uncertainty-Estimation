import tensorflow as tf
import numpy as np
from getTestTrainVal import getData, change_floats_for_regression_to_classes_for_classification_labels

def dataset_selection(dataset=None, num_train = None, num_feature = None):

    headings = None

    if dataset in ['Random_Dataset']:
        x = np.random.rand(num_train, num_feature)
        response_noise = np.zeros(num_train,)

        # 噪声
        # 响应噪声-高斯噪声
        response_noise = np.random.randn(num_train,)
        # response_noise = np.random.normal(loc=0, scale=np.sqrt(1), size=(num_train,))

        # # 响应噪声-均匀噪声[0,1]
        # response_noise = np.random.uniform(0, 1, num_train)
        #
        # # 输入噪声-高斯噪声
        # x[:, 0] += np.random.randn(num_train,)
        # x[:, 0] = 1 / (1 + np.exp(-x[:, 0]))

        # # 输入噪声-均匀噪声[0,1]
        # x[:, 0] += np.random.uniform(0, 1, num_train)
        # x[:, 0] = 1 / (1 + np.exp(-x[:, 0]))


        y = x[:, 0]**2 + 10*x[:, 1]*x[:, 2]*x[:, 3] + 5*x[:, 4]*x[:, 5] + response_noise

        # 冗余
        x[:, 6] = np.copy(x[:, 5])

        task = 'regression'

    elif dataset in ['NAKO_Diabetes_Dataset', 'NAKO_Normal_Dataset']:
        if dataset in ['NAKO_Diabetes_Dataset']:
            NAKO_getData_object = getData(diabetes_path=r"NAKO_dataset/13k_diabetes/NAKO_536_61223_diabetes_metadata.csv")  # Class in getTestTrainVal.py
        elif dataset in ['NAKO_Normal_Dataset']:
            NAKO_getData_object = getData(diabetes_path=r"NAKO_dataset/30k_metadata/NAKO-536_general_metadata.csv")  # Class in getTestTrainVal.py

        just_oGTT2h = False

        if just_oGTT2h is False:
            # Gets Test/Train/Val sets (after lots of filtering to change "missings indicators" to np.nan)
            x_tr, y_tr, x_te, y_te, x_val, y_val = NAKO_getData_object.get_NAKO_diabetes_data(drop_actual_diabetes_measurement_methods=True)

            y_tr = change_floats_for_regression_to_classes_for_classification_labels(y=y_tr)
            y_te = change_floats_for_regression_to_classes_for_classification_labels(y=y_te)
            y_val = change_floats_for_regression_to_classes_for_classification_labels(y=y_val)
        else:
            x_tr, y_tr, x_te, y_te, x_val, y_val = NAKO_getData_object.get_NAKO_diabetes_classes(drop_actual_diabetes_measurement_methods=True,
                                                                                                 split_train_into_train_and_val=False)
            # x_tr, y_tr, x_te, y_te, x_val, y_val = NAKO_getData_object.get_PyRad_feats_and_NAKO_diabetes_classes(average_feats_over_signals=True)

        x = x_tr
        y = y_tr
        headings = x.columns.to_list()
        x = np.array(x)
        x = np.nan_to_num(x, nan=0.0)
        task = 'regression'

    elif dataset in ['MNIST']:
        # 加载 MNIST 数据集
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # 归一化图像数据到0-1范围
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # 将训练和测试数据集展平
        x_train_flattened = x_train.reshape(x_train.shape[0], -1)
        x_test_flattened = x_test.reshape(x_test.shape[0], -1)


        # 将标签转换为 one-hot 编码
        y_train_one_hot = tf.keras.utils.to_categorical(y_train, 10)
        y_test_one_hot = tf.keras.utils.to_categorical(y_test, 10)

        x = x_train_flattened
        y = y_train_one_hot
        task = 'classification'

    else:
        raise ValueError('Please enter a valid dataset name...')

    return x, y, headings, task
