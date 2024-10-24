import timeit

from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Dropout, Lambda, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np

from selection_layer import build_selection_layer
from learning_network import build_learning_net


def time_decorator(func):
    def wrapper(*args, **kwargs):
        t1 = timeit.default_timer()
        func(*args,  **kwargs)
        t2 = timeit.default_timer()
        compute_time = t2-t1
        print('----- Running time: %.4fs -----\n' % compute_time)
    return wrapper
#测量函数的运行时间FeatureSelector

class FeatureSelector():

    def __init__(self, method='FM', task='regression', num_target_fea=5, input_shape=(11,),
                 output_shape=1, selector_reg=None, learning_reg=None, ln_dim=64, **kwargs):
        """
        Crete a variable selector.
        :param method: The name of feature selection method.
        :param task: The name of the learning task, e.g. regression or classification.
        :param num_target_fea: The number of the features which is desired to be selected.
        :param input_shape: The shape of the input data, not taking the batch size into consideration.
        :param output_shape: The number of output layer neurons.
        :param selector_reg: The regularization term for the variable selection method, default=None.
        :param learning_reg: The regularization term for the learning network, default=None.
        :param ln_dim: The number of the hidden neurons in the first hidden layer.
        :param kwargs: Only works for FM (the number of hidden neurons for FM => for insight study of FM).
        """

        self.task = task
        self.hist = None
        self.mask = None
        self.accu = 0
        self.method = method
        self.num_target_fea = num_target_fea
        self.input_shape = input_shape
        self.output_shape = output_shape

        # ---------- define variable selection layer and learning network ----------
        fs_layer = build_selection_layer(method=method, reg=selector_reg,
                                         num_target_fea=num_target_fea, **kwargs)

        learning_net = build_learning_net(method=method, input_shape=input_shape,
                                          output_shape=output_shape, task=task,
                                          num_target_fea=num_target_fea, reg=learning_reg, ln_dim=ln_dim)

        # ---------- define the entire model for end-to-end variable selection ----------
        input_layer = Input(shape=input_shape)
        x = fs_layer(input_layer)
        output_layer = learning_net(x)

        self.fs_wrapper = Model(inputs=input_layer, outputs=output_layer)
        # 整个选择器+学习器的模型
        self.selection_net = Model(inputs=input_layer, outputs=x)
        # 选择器的模型
        self.learning_net = learning_net
        # 学习器的模型
        self.optimizer = Adam(learning_rate=1e-3, decay=1e-9)
        # 定义优化器

        if task in ['classification']:
            loss = 'categorical_crossentropy'
        elif task in ['regression', 'multihead']:
            loss = 'mse'
        else:
            raise ValueError('Please enter valid losses and metrics...')

        self.fs_wrapper.compile(optimizer=self.optimizer, loss=loss)
        self.selection_net.compile('sgd', 'mse')  # placeholder
        self.learning_net.compile('sgd', 'mse')  # placeholder
        # 设备优化器和选择函数


    @time_decorator
    def fit(self, x=None, y=None, **kwargs):
        """
        Training model.
        :param x: The input dataset.
        :param y: The target variables (labels/ground truth)
        :param kwargs: Other parameters, such as batch_size, epochs, verbose, callbacks.
        :return: None
        """
        self.hist = self.fs_wrapper.fit(x=x, y=y, **kwargs)
        # 训练整个选择器+学习器的模型

    def get_mask(self, inputs):
        """
        Calcualte the learned feature mask.
        :param inputs: The entire training dataset.
        :return: The learned feature mask calculated on the entire training dataset.
        """
        # if self.mask is None:
        #     self.mask = self.selection_net.get_layer(self.method).get_support(inputs)
        self.mask = self.selection_net.get_layer(self.method).get_support(inputs)
        return self.mask
        # 获取importance scores的mask

    def show_selection(self, inputs, ticksname=None, top_k=10):
        if ticksname is None:
            ticksname = ['x%d' % i for i in range(top_k)]
        # 如果没有提供tick名称，则生成一个由'x0', 'x1', ...组成的列表

        if self.mask is None:
            self.mask = self.selection_net.get_layer(self.method).get_support(inputs)
        # 如果还没有mask，则通过调用selection_net的get_layer方法获取support
        idx_mask = np.argsort(self.mask)[::-1]
        # 对mask进行排序，获取降序排序后的索引

        results = ''
        for i in range(top_k):
            var_results = ''
            for j in range(i+1):
                var_results += '%s, ' % ticksname[idx_mask[j]]
            var_results = var_results[:-2]
            # 删除最后一个逗号和空格，使字符串格式正确
            results += 'top-%d: %s\n' % (i+1, var_results)
        results = results[:-1]
        # 删除最后一个换行符
        num_char = results[::-1].index('\n')
        results = 'Selection results:\n' + '-' * num_char + '\n' + results + '\n' + '-' * num_char
        print(results)
        return results

    def predict_with_uncertainty(self, inputs, n_iter=100):
        predictions = np.zeros((n_iter, inputs.shape[1]))
        for i in range(n_iter):
            # 进行 Monte Carlo 采样，获取每次的特征选择结果
            predictions[i] = self.selection_net.get_layer(self.method).get_support(inputs)
        mean = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        return mean, uncertainty


class DeepEnsemble_FeatureSelector():
    def __init__(self, ensemble_size=5, **kwargs):
        """
        :param ensemble_size: Number of models in the ensemble.
        :param kwargs: Parameters for the FeatureSelector class.
        """
        self.ensemble_size = ensemble_size
        # Generate multiple instances of FeatureSelector
        self.models = [FeatureSelector(**kwargs) for _ in range(ensemble_size)]

    def fit(self, x, y, **kwargs):
        # Train each model in the ensemble.
        for model in self.models:
            model.fit(x, y, **kwargs)
            # Train every model

    def get_mask(self, inputs):
        # Calculate the feature masks for all models and return the average mask.

        # 获取每个模型的特征mask
        masks = [model.get_mask(inputs) for model in self.models]
        # 对不同模型的mask取平均
        return np.mean(masks, axis=0)

    def predict_with_uncertainty(self, inputs, n_iter=100):
        predictions = np.zeros((self.ensemble_size, n_iter, inputs.shape[1]))
        # 遍历每个模型
        for model_idx, model in enumerate(self.models):
            # 对每个模型进行 n_iter 次采样
            for i in range(n_iter):
                predictions[model_idx, i] = model.get_mask(inputs)

        # 在 ensemble_size 和 n_iter 两个维度上计算平均值和标准差
        mean = predictions.mean(axis=(0, 1))  # 先对模型和采样求均值
        uncertainty = predictions.std(axis=(0, 1))  # 再计算模型和采样的不确定性（标准差）
        return mean, uncertainty


class Bagging_FeatureSelector():
    def __init__(self, ensemble_size=5, sample_fraction=0.8, **kwargs):
        """
        :param ensemble_size: Number of models in the ensemble.
        :param sample_fraction: Fraction of the dataset to be sampled with replacement for each model.
        :param kwargs: Parameters for the FeatureSelector class.
        """
        self.ensemble_size = ensemble_size
        self.sample_fraction = sample_fraction  # 数据集的采样比例
        self.models = [FeatureSelector(**kwargs) for _ in range(ensemble_size)]

    def fit(self, x, y, **kwargs):
        # Train each model in the ensemble with Bagging.
        n_samples = x.shape[0]
        for model in self.models:
            # 通过有放回采样生成训练数据子集
            indices = np.random.choice(n_samples, size=int(self.sample_fraction * n_samples), replace=True)
            x_sample = x[indices]
            y_sample = y[indices]

            # 使用采样子集训练模型
            model.fit(x_sample, y_sample, **kwargs)

    def get_mask(self, inputs):
        # Calculate the feature masks for all models and return the average mask.

        # 获取每个模型的特征mask
        masks = [model.get_mask(inputs) for model in self.models]
        # 对不同模型的mask取平均
        return np.mean(masks, axis=0)

    def predict_with_uncertainty(self, inputs, n_iter=100):
        predictions = np.zeros((self.ensemble_size, n_iter, inputs.shape[1]))
        # 遍历每个模型
        for model_idx, model in enumerate(self.models):
            # 对每个模型进行 n_iter 次采样
            for i in range(n_iter):
                predictions[model_idx, i] = model.get_mask(inputs)

        # 在 ensemble_size 和 n_iter 两个维度上计算平均值和标准差
        mean = predictions.mean(axis=(0, 1))  # 先对模型和采样求均值
        uncertainty = predictions.std(axis=(0, 1))  # 再计算模型和采样的不确定性（标准差）
        return mean, uncertainty


class Boosting_FeatureSelector():
    def __init__(self, ensemble_size=5, **kwargs):
        """
        :param ensemble_size: Number of models in the ensemble (stages of boosting).
        :param kwargs: Parameters for the FeatureSelector class.
        """
        self.ensemble_size = ensemble_size
        # 创建多个 FeatureSelector 实例，每个作为一个阶段的模型
        self.models = [FeatureSelector(**kwargs) for _ in range(ensemble_size)]

    def fit(self, x, y, **kwargs):
        # Sequentially train each model in the ensemble using boosting.
        residuals = y  # 初始化为目标值

        for model_idx, model in enumerate(self.models):
            print(f"Training model {model_idx + 1}/{self.ensemble_size}")

            # 使用当前的残差训练模型
            model.fit(x, residuals, **kwargs)

            # 预测当前模型的输出
            predictions = model.fs_wrapper.predict(x)

            # 计算新的残差（更新目标为剩余误差）
            residuals = residuals - predictions  # 更新残差以减去当前模型的预测

            # 模型的残差越小，训练越能接近目标

    def get_mask(self, inputs):
        # Calculate the feature masks for all models and return the average mask.

        # 获取每个模型的特征mask
        masks = [model.get_mask(inputs) for model in self.models]
        # 对不同模型的mask取平均
        return np.mean(masks, axis=0)

    def predict_with_uncertainty(self, inputs, n_iter=100):
        # Perform prediction with uncertainty estimation using Monte Carlo sampling and ensemble models.
        predictions = np.zeros((self.ensemble_size, n_iter, inputs.shape[1]))

        # 遍历每个模型进行 boosting 预测
        for model_idx, model in enumerate(self.models):
            for i in range(n_iter):
                # 每个模型采样多次，并累积结果
                predictions[model_idx, i] = model.get_mask(inputs)

        # 计算所有模型和所有采样的平均值和标准差
        mean = predictions.mean(axis=(0, 1))  # 先对模型和采样求均值
        uncertainty = predictions.std(axis=(0, 1))  # 再计算标准差作为不确定性

        return mean, uncertainty
