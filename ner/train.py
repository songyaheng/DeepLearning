# -*- coding: utf-8 -*-
import tensorflow as tf
import functools
from pathlib import Path
import numpy as np
import logging
import sys
import json
# Logging
Path('log').mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('log/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.split()]
    tags = [t.encode() for t in line_tags.split()]
    assert len(words) == len(tags), "Words and tags lengths don't match"
    return (words, len(words)), tags


def generator_fn(words, tags):
    with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)


def input_fn(words_path, tags_path, params, shuffle_and_repeat=False):
    shapes = (([None], ()), [None])
    types = ((tf.string, tf.int32), tf.string)
    defaults = (('UNK', 0), 'O')
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words_path, tags_path),
        output_shapes=shapes, output_types=types)
    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])
    dataset = dataset.padded_batch(params['batch_size'], shapes, defaults)
    return dataset


def ner_model_fn(features, labels, mode, params):
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    # 加载字表, 如果字表里边没有就是UNK
    vocab_words = tf.contrib.lookup.index_table_from_file(params['words'], num_oov_buckets=1)
    words, nwords = features[0], features[1]
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
    # output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.add(output_fw, output_bw)
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
    params = {
        "root_path": "/data/songyaheng/data",
        "words": "/data/songyaheng/data/vocab.words.txt",
        "tags": "/data/songyaheng/data/vocab.tags.txt",
        "embedding": "/data/songyaheng/data/embedding.npz",
        "dim": 100,
        "batch_size": 64,
        "learning_rate": 0.001,
        "save_checkpoints_secs": 120,
        "dropout": 0.5,
        "lstm_size": 100,
        "train_step": 1000,
        "buffer": 15000,
        "epochs": 25
    }


    def fwords(name):
        return str(Path(params["root_path"], '{}.words.txt'.format(name)))


    def ftags(name):
        return str(Path(params["root_path"], '{}.tags.txt'.format(name)))

    # 2.读取record 数据，组成batch
    train_input_fn = functools.partial(input_fn, fwords('train'), ftags('train'),
                                       params, shuffle_and_repeat=True)
    eval_input_fn = functools.partial(input_fn, fwords('test'), ftags('test'), params)

    cfg = tf.estimator.RunConfig(
        save_checkpoints_secs=params["save_checkpoints_secs"]
    )

    with Path(params['tags']).open() as f:
        params["num_tags"] = len(f.readlines()) + 1

    with Path('{}/params.json'.format(params["root_path"])).open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    estimator = tf.estimator.Estimator(ner_model_fn, '{}/model'.format(params["root_path"]), cfg, params)
    hook = tf.contrib.estimator.stop_if_no_increase_hook(
        estimator, 'acc', 0.95, min_steps=params["train_step"], run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

