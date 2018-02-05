import tensorflow as tf

COLUMNS = ['date_', 'store',
           'department', 'item',
           'unit_price','on_promotion', 'promotion_type', 'quantity']
FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0], [0], [0], [0]]

def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have either run data_download.py or '
      'set both arguments --train_data and --test_data.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=FIELD_DEFAULTS)
    features = dict(zip(FIELD_DEFAULTS, columns))
    labels = features.pop('income_bracket')
    return features, tf.equal(labels, '>50K')

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=10)

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels

input_fn("/Users/ASharaf/Desktop/hackathon_data/trial.csv", 0, False, 10)