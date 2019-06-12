import tensorflow as tf
from pathlib import Path
import json
import sys
import numpy as np

def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders
    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    words = tf.placeholder(dtype=tf.string, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='nwords')
    receiver_tensors = {'words': words, 'nwords': nwords}
    features = {'words': words, 'nwords': nwords}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def ner_model_fn(features, labels, mode, params):
    # For serving, features are a bit different
    words, nwords = features['words'], features['nwords']
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    # 加载字表, 如果字表里边没有就是UNK
    vocab_words = tf.contrib.lookup.index_table_from_file(params['words'], num_oov_buckets=1)
    word_ids = vocab_words.lookup(words)

    # Word Embeddings
    embeddings = np.load(params['embedding'])['embeddings']  # np.array
    variable = np.vstack([embeddings, [[0.] * params['dim']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    embedding = tf.nn.embedding_lookup(variable, word_ids)
    embedding = tf.layers.dropout(embedding, rate=params["dropout"], training=training)

    # LSTM
    t = tf.transpose(embedding, perm=[1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=params["dropout"], training=training)

    # CRF
    logits = tf.layers.dense(output, params["num_tags"])
    crf_params = tf.get_variable("crf", [params["num_tags"], params["num_tags"]], dtype=tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(params['tags'], default_value='0')
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'], num_oov_buckets=1)
        tags = vocab_tags.lookup(labels)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, tags, nwords, crf_params)
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(nwords)
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'precision': tf.metrics.precision(tags, pred_ids, weights),
            'recall': tf.metrics.recall(tags, pred_ids, weights),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(learning_rate=params["learning_rate"]).minimize(loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

if __name__ == '__main__':
    with Path("{}/params.json".format(sys.argv[1])).open() as f:
        params = json.load(f)
    estimator = tf.estimator.Estimator(ner_model_fn, "{}/model".format(params["root_path"]), params=params)
    estimator.export_saved_model('{}/saved_model'.format(params["root_path"]), serving_input_receiver_fn)
