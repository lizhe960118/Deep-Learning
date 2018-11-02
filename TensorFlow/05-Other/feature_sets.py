import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv(
    'https://storage.googleapis.com/mledu-datasets/california_housing_train.csv',
    sep=',')
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))


def preprocess_features(california_housing_dataframe):
    selected_features = california_housing_dataframe[['latitude',
                                                      'longitude',
                                                      'housing_median_age',
                                                      'total_rooms',
                                                      'total_bedrooms',
                                                      'population',
                                                      'households',
                                                      'median_income']]
    processed_features = selected_features.copy()
    processed_features['rooms_per_person'] = (
        california_housing_dataframe['total_rooms'] /
        california_housing_dataframe['population'])
    return processed_features


def preprocess_targets(california_housing_dataframe):
    output_targets = pd.DataFrame()
    output_targets['median_house_value'] = (
        california_housing_dataframe['median_house_value'] / 1000.0)
    return output_targets


training_examples = preprocess_features(
    california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

validation_examples = preprocess_features(
    california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(
    california_housing_dataframe.tail(5000))

print 'Training examples summary:'
display.display(training_examples.describe())
print 'Validation examples summary'
display.display(validation_examples.describe())

print 'Training targets summary'
display.display(training_targets.describe())
print 'Validation targets summary'
display.display(validation_targets.describe())

# 查看数据的相关性
correlation_dataframe = training_examples.copy()
correlation_dataframe['targets'] = training_targets['median_house_value']
correlation_dataframe.corr()


def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature) fpr my_feature in input_features])


def my_input_fn(
        features,
        targets,
        batch_size=1,
        shuffle=True,
        num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.train.GrandientDescentOptimizer(
        learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(
        my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_column=construct_feature_columns(training_examples),
        optimizer=my_optimizer)

    def training_input_fn(): return my_input_fn(
        training_examples,
        training_targets['median_house_value'],
        batch_size=batch_size)

    def predict_training_input_fn(): return my_input_fn(
        training_examples,
        training_targets['median_house_value'],
        num_epochs=1,
        shuffle=False)

    def predict_validation_input_fn(): return my_input_fn(
        validation_examples,
        validation_targets['median_house_value'],
        num_epochs=1,
        shuffle=False)

    print "Traininf model"
    print "RMSE (on training data):"
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period)
        training_predictions = linear_regressor.predict(
            input_fn=predict_training_input_fn)
        training_predictions = np.array(
            [item['predictions'][0] for item in training_predictions])
        validation_predictions = linear_regressor.predict(
            input_fn=validation_predictions)
        validation_predictions = np.array(
            [item['predictions'][0] for item in validation_predictions])

        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(
                training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(
                validation_predictions, validation_targets))

        print "period %02d : %0.2f" % (period, training_root_mean_squared_error)
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print 'Model training finished'
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title('Root Mean Squared Error vs. Periods')
    plt.tight_layout()
    plt.plot(training_rmse, label='training')
    plt.plot(validation_rmse, label='validation')
    plt.legend()
    return linear_regressor


_ = train_model(
    learning_rate=1.0,
    steps=500,
    batch_size=100,
    feature_columns=construct_feature_columns(training_examples),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

'''
要为分桶特征定义特征列，我们可以使用 bucketized_column（而不是使用 numeric_column），
该列将数字列作为输入，并使用 boundardies 参数中指定的分桶边界将其转换为分桶特征。
以下代码为 households 和 longitude 定义了分桶特征列；
get_quantile_based_boundaries 函数会根据分位数计算边界，以便每个分桶包含相同数量的元素。
'''


def get_quantile_based_boundaries(feature_values, num_buckets):
    boundaries = np.arange(1.0, num_buckets) / num_buckets
    quantiles = feature_values.quantile(boundaries)
    return [quantiles[q] for q in quantiles.keys()]


def construct_feature_columns():
    households = tf.feature_column.numeric_column('households')
    longitude = tf.feature_column.numeric_column('longitude')
    latitude = tf.feature_column.numeric_column('latitude')
    housing_median_age = tf.feature_column.numeric_column('housing_median_age')
    median_income = tf.feature_column.numeric_column('median_income')
    rooms_per_person = tf.feature_column.numeric_column('rooms_per_person')

    bucketized_households = tf.feature_column.bucketized_column(
        households, boundaries=get_quantile_based_boundaries(
            training_examples['households'], 7))
    bucketized_longitude = tf.feature_column.bucketized_column(
        longitude, boundaries=get_quantile_based_boundaries(
            training_examples['longitude'], 10))
    bucketized_latitude = tf.feature_column.bucketized_column(
        latitude, boundaries=get_quantile_based_boundaries(
            training_examples['latitude'], 10))
    bucketized_housing_median_age = tf.feature_column.bucketized_column(
        housing_median_age, boundaries=get_quantile_based_boundaries(
            training_examples['housing_median_age'], 10))
    bucketized_median_income = tf.feature_column.bucketized_column(
        median_income, boundaries=get_quantile_based_boundaries(
            training_examples['median_income'], 10))
    bucketized_rooms_per_person = tf.feature_column.bucketized_column(
        rooms_per_person, boundaries=get_quantile_based_boundaries(
            training_examples['rooms_per_person'], 10))

    feature_columns = set([
        bucketized_longitude,
        bucketized_latitude,
        bucketized_housing_median_age,
        bucketized_median_income,
        bucketized_rooms_per_person])

    return feature_columns


_ = train_model(
    learning_rate=1.0,
    steps=500,
    batch_size=100,
    feature_columns=construct_feature_columns(),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
