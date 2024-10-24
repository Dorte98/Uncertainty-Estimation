import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models
from tensorflow_probability import stats
import numpy as np

# 生成一些模拟数据
def generate_data(num_samples=1000):
    X = np.linspace(-1, 1, num_samples)
    y = X**3 + np.random.normal(0, 0.1, size=X.shape)
    return X, y

X_train, y_train = generate_data()
X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

# print (X_train.shape[0])
# print (y_train.shape[0])


# 构建模型
def build_model():
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(1,)),
        layers.Dropout(0.2),  # Dropout层
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),  # Dropout层
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

model = build_model()

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

def predict_with_uncertainty(model, X, n_iter=100):
    predictions = np.zeros((n_iter, X.shape[0]))

    for i in range(n_iter):
        # 预测时开启Dropout
        predictions[i] = model(X, training=True).numpy().flatten()

    mean = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)

    return mean, uncertainty

# 生成一些新数据进行预测
X_test = np.linspace(-1, 1, 100).reshape(-1, 1)


print (X_test.shape[0])

# 使用模型进行预测并计算不确定性
mean, uncertainty = predict_with_uncertainty(model, X_test)

# print(mean.shape)
# print(uncertainty.shape)
# 打印前5个样本的预测结果和不确定性
for i in range(5):
    # print(f"Input: {X_test[i]}")
    # print(X_test[i], X_test[i][0])
    print(f"Input: {X_test[i][0]:.2f}, Prediction: {mean[i]:.4f}, Uncertainty: {uncertainty[i]:.4f}")

# # 计算置信区间
# confidence_level = 0.95
# mean = 5
# std_dev = 1
# n = 100
# confidence_interval = stats.norm.interval(confidence_level, loc=mean, scale=std_dev / np.sqrt(n))
# print(f"95% 置信区间: {confidence_interval}")

def visualize_confidence_interval(mean_prediction, std_prediction, num_samples):
    # 计算 95% 置信区间
    z_value = 1.96
    lower_bound = mean_prediction - z_value * std_prediction
    upper_bound = mean_prediction + z_value * std_prediction

    # 可视化
    plt.figure(figsize=(10, 6))

    # 横轴：输入编号
    x_axis = np.arange(num_samples)

    # 绘制预测的均值曲线
    plt.plot(x_axis, mean_prediction, label="Mean Prediction", color="blue")

    # 填充置信区间
    plt.fill_between(x_axis, lower_bound, upper_bound, color='orange', alpha=0.3, label="95% Confidence Interval")

    # 标题和图例
    plt.title("Monte Carlo Dropout with 95% Confidence Interval")
    plt.xlabel("Input Index")
    plt.ylabel("Prediction Mean")
    plt.legend()

    plt.show()

visualize_confidence_interval(mean, uncertainty, len(mean))
