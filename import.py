import tensorflow as tf
import csv
import pandas

# Metadata describing the text columns
COLUMNS = ['date_', 'store',
           'department', 'item',
           'unit_price','on_promotion', 'promotion_type', 'quantity']
FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0], [0], [0], [0]]


def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, FIELD_DEFAULTS)

    # Pack the result into a dictionary
    features = dict(zip(COLUMNS,fields))

    # Separate the label from the features
    label = features.pop('quantity')
    print(line)
    #print(features)
    return features, label

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Build the Iterator, and return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()



if __name__ == "__main__":
    # All the inputs are numeric
    dataframe = pandas.read_csv('/Users/ASharaf/Desktop/hackathon_data/trial.csv', header=0)
    features = pandas.read_csv('/Users/ASharaf/Desktop/hackathon_data/trial.csv', header=0)
    label = features.pop('quantity')
    


