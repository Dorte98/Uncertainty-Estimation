import numpy as np
import matplotlib.pyplot as plt

from dataset_selector import dataset_selection
from getTestTrainVal import getData, change_floats_for_regression_to_classes_for_classification_labels
from training_monitor import SelectorMonitor, plot_fea_importance_hist
from variable_selector_wrapper import FeatureSelector, DeepEnsemble_FeatureSelector, Bagging_FeatureSelector, Boosting_FeatureSelector
import scipy.stats as stats

if __name__ == '__main__':

    EPOCHS = 8000
    BATCH_SIZE = 256  # This large batch size is good for efficient training
    STEP = 200
    DEVICE_ID = 0  # You can also change to other device IDs from 0 to 8
    METHOD = 'FM'  # You can also test other reference methods, e.g. ConAE, CancelOut, BSF, DFS
    LAT_DIM = 8  # This is default for FM
    DATASET = 'Random_Dataset'
    # DATASET = 'NAKO_Diabetes_Dataset'
    # DATASET = 'NAKO_Normal_Dataset'
    # DATASET = 'MNIST'
    DeepEnsemble = False
    Bagging = False
    Boosting = True


    NUM_TRAIN = 5000
    NUM_FEATURE = 15

    # 数据库加载
    x, y, HEADINGS, TASK = dataset_selection(DATASET, NUM_TRAIN, NUM_FEATURE)
    data = [x, y]
    NUM_FEATURE = len(data[0][0])  #data[0].shape[1]
    TICK_NAMES = ['x%d' % i for i in range(NUM_FEATURE)]


    if DATASET in ['MNIST']:
        OUTPUT_SHAPE = len(data[1][0])
    else:
        OUTPUT_SHAPE = 1

    print('x_shape', data[0].shape)
    print('y_shape', data[1].shape)
    print('NUM_FEATURE', NUM_FEATURE)
    print('OUTPUT_SHAPE', OUTPUT_SHAPE)
    print('task', TASK)


    if DeepEnsemble is True:
        feature_selector = DeepEnsemble_FeatureSelector(ensemble_size=2,
                                       method=METHOD, task=TASK, num_target_fea=5,
                                       input_shape=(NUM_FEATURE, ), output_shape=OUTPUT_SHAPE,
                                       learning_reg=None, selector_reg=None, ln_dim=32,
                                       lat_dim=LAT_DIM, input_dim=NUM_FEATURE, kl_weight=1e-4)

    elif Bagging is True:
        feature_selector = Bagging_FeatureSelector(ensemble_size=2, sample_fraction=0.8,
                                       method=METHOD, task=TASK, num_target_fea=5,
                                       input_shape=(NUM_FEATURE, ), output_shape=OUTPUT_SHAPE,
                                       learning_reg=None, selector_reg=None, ln_dim=32,
                                       lat_dim=LAT_DIM, input_dim=NUM_FEATURE, kl_weight=1e-4)

    elif Boosting is True:
        feature_selector = Boosting_FeatureSelector(ensemble_size=2,
                                       method=METHOD, task=TASK, num_target_fea=5,
                                       input_shape=(NUM_FEATURE, ), output_shape=OUTPUT_SHAPE,
                                       learning_reg=None, selector_reg=None, ln_dim=32,
                                       lat_dim=LAT_DIM, input_dim=NUM_FEATURE, kl_weight=1e-4)

    else:
        feature_selector = FeatureSelector(method=METHOD, task=TASK, num_target_fea=5,
                                       input_shape=(NUM_FEATURE, ), output_shape=OUTPUT_SHAPE,
                                       learning_reg=None, selector_reg=None, ln_dim=32,
                                       lat_dim=LAT_DIM, input_dim=NUM_FEATURE, kl_weight=1e-4)


    fs_monitor = SelectorMonitor(x=data[0], y=data[1], method=METHOD, save_name=METHOD,
                                 tick_names=TICK_NAMES, step=STEP, top_k=5, dataset=DATASET)

    feature_selector.fit(x=data[0], y=data[1], batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0, callbacks=[fs_monitor])

    m = feature_selector.get_mask(data[0][:])
    print(f"MASK: {m}")

    mean, uncertainty = feature_selector.predict_with_uncertainty(inputs=data[0], n_iter=100)


    def modify_data(x=None, mask=None, top_k=None, keep_zeros=False):
        idx = np.argsort(mask)[::-1]
        if keep_zeros:
            x = np.copy(x)
            x[idx[top_k:]] = 0
        else:
            x = x[idx[:top_k]]
        return x


    modified_mean = modify_data(mean, mean, top_k=6)
    print(f"modified_mean: {modified_mean}")
    modified_uncertainty = modify_data(uncertainty, mean, top_k=6)
    print(f"modified_uncertainty: {modified_uncertainty}")


    # for i in range(len(mean)):
    #     # print(f"Input: {data[0][0][i]:.2f}, Prediction: {mean[i]:.4f}, Uncertainty: {uncertainty[i]:.4f}")
    #   print(f"Input: {data[0][0][i]}, Prediction: {mean[i]}, Uncertainty: {uncertainty[i]}")


    # 变异系数CV
    cv = np.where(mean != 0, uncertainty / mean , np.inf)
    print("变异系数:", cv)
    overall_cv = np.mean(cv)
    print(f"整体变异系数: {overall_cv}")

    cv_modified_data = np.where(modified_mean != 0, modified_uncertainty / modified_mean , np.inf)
    print("变异系数_top_k=6:", cv_modified_data)
    overall_cv_modified_data = np.mean(cv_modified_data)
    print(f"整体变异系数_top_k=6: {overall_cv_modified_data}")

    #
    # # 计算 95% 置信区间
    # def visualize_confidence_interval(mean_prediction, std_prediction, num_samples):
    #     alpha = 0.15
    #     z_value = stats.norm.ppf(1 - alpha / 2)
    #     # lower_bound = mean_prediction - z_value * std_prediction
    #     # upper_bound = mean_prediction + z_value * std_prediction
    #     error = z_value * std_prediction
    #
    #     # 可视化
    #     plt.figure(figsize=(10, 6))
    #     # 横轴：输入编号
    #     x_axis = np.arange(num_samples)
    #     # 绘制预测的均值曲线
    #     # plt.bar(x_axis, mean_prediction, yerr=error, capsize=5, color='lightblue', label="Mean Prediction")
    #     plt.errorbar(x_axis, mean_prediction, yerr=error, fmt='o', capsize=5, label='Mean with 95% CI')
    #     # # 填充置信区间
    #     # plt.fill_between(x_axis, lower_bound, upper_bound, color='orange', alpha=0.3, label="95% Confidence Interval")
    #
    #     # 标题和图例
    #     plt.title("Monte Carlo Dropout with 95% Confidence Interval")
    #     plt.xlabel("Input Index")
    #     plt.ylabel("Prediction Mean")
    #     plt.legend()
    #
    #     plt.show()
    #
    # visualize_confidence_interval(mean, uncertainty, NUM_FEATURE)


    # plot_fea_importance_hist(fea_importance=m,
    #                          name='Exp_for_%s_with_%s' % (METHOD, DATASET),
    #                          tick_names=TICK_NAMES,
    #                          method_name=METHOD,
    #                          show_order=True,
    #                          dataset=DATASET,
    #                          headings=HEADINGS)
    #
    # res = []
    # for i in range(50):
    #     m = feature_selector.selection_net.get_layer('BFM').get_support(data[0])
    #     res.append(m)
    # res = np.asarray(res)
    # print(res.shape)
    #
    # res_mean = np.mean(res, axis=0, dtype='float64')
    # res_std = np.std(res, axis=0, dtype='float64')
    #
    #
    # TICK_NAMES = ['x%d' % (i+1) for i in range(NUM_FEATURE)]
    #
    # n_fea = NUM_FEATURE
    # plt.bar(x=np.arange(n_fea), height=res_mean, yerr=res_std,
    #         align='center', alpha=0.5, ecolor='dimgray', capsize=6)
    # plt.xticks(ticks=np.arange(n_fea), labels=TICK_NAMES)
    # plt.grid(True, linestyle='--', which='major', color='grey', alpha=.55)
    # plt.savefig('syn_x1_x6_x7_eq_x6.png', format='png', dpi=200, bbox_inches='tight')
    # plt.show()
