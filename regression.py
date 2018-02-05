import pandas as pd
import tensorflow as tf
CSV_COLUMN_NAMES = ['date_', 'store',
                    'department', 'item', 'unit_price', 'quantity', 'promotion_type', 'on_promotion']


def load_data(label_name='quantity'):
    """Parses the csv file in TRAIN_URL and TEST_URL."""

    # Create a local copy of the training set.
    train_path = '/Users/ASharaf/Desktop/hackathon_data/jan09.csv'
    # train_path now holds the pathname: ~/.keras/datasets/iris_training.csv

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,  # list of column names
                        header=0  # ignore the first row of the CSV file.
                       )
    # train now holds a pandas DataFrame, which is data structure
    # analogous to a table.
    dums = pd.get_dummies(train.department, prefix='Dept', drop_first=True)
    train = pd.concat([train, dums], axis=1)

    # 1. Assign the DataFrame's labels (the right-most column) to train_label.
    # 2. Delete (pop) the labels from the DataFrame.
    # 3. Assign the remainder of the DataFrame to train_features
    train_features, train_label = train, train.pop(label_name)

    # Apply the preceding logic to the test set.
    test_path = '/Users/ASharaf/Desktop/hackathon_data/jan09.csv'
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_features, test_label = test, test.pop(label_name)

    # Return four DataFrames.
    return (train_features, train_label), (test_features, test_label)

def train_input_fn():
    features, labels = input_fn()
    batch_size = 1000
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(10).repeat().batch(batch_size)

    # Build the Iterator, and return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

(train_feature, train_label), (test_feature, test_label) = load_data()

def input_fn():
    # Create a local copy of the training set.
    train_path = '/Users/ASharaf/Desktop/hackathon_data/jan09.csv'
    # train_path now holds the pathname: ~/.keras/datasets/iris_training.csv

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,  # list of column names
                        header=0  # ignore the first row of the CSV file.
                        )
    # train now holds a pandas DataFrame, which is data structure
    # analogous to a table.

    # 1. Assign the DataFrame's labels (the right-most column) to train_label.
    # 2. Delete (pop) the labels from the DataFrame.
    # 3. Assign the remainder of the DataFrame to train_features
    train_features, train_label = train, train.pop('quantity')
    return train_feature, train_label

def input_fn2():
    # Create a local copy of the training set.
    train_path = '/Users/ASharaf/Desktop/hackathon_data/jan10.csv'
    # train_path now holds the pathname: ~/.keras/datasets/iris_training.csv

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,  # list of column names
                        header=0  # ignore the first row of the CSV file.
                        )
    # train now holds a pandas DataFrame, which is data structure
    # analogous to a table.

    # 1. Assign the DataFrame's labels (the right-most column) to train_label.
    # 2. Delete (pop) the labels from the DataFrame.
    # 3. Assign the remainder of the DataFrame to train_features
    train_features, train_label = train, train.pop('quantity')
    return train_feature, train_label

def train_input_fn2():
    features, labels = input_fn2()
    batch_size = 5
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Build the Iterator, and return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()
estimator = tf.estimator.DNNRegressor(
    feature_columns=[tf.feature_column.numeric_column(key='unit_price'), tf.feature_column.numeric_column(key='department')],
    hidden_units=[5000, 8000, 2000])

trained = estimator.train(train_input_fn, steps=1000)
print(trained)

evaluated = estimator.evaluate(train_input_fn2, steps=10)
print(evaluated)

#predictions = estimator.predict(train_input_fn2)
#print(predictions)
