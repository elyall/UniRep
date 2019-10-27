"""
Runs embedding/inference for an input fasta file:

    ls -S -r ../github/ml-datasets/*/*.faa | parallel -j1 "python run_inference.py {} {}.unirep1900.pkl"
    ls -S -r ../github/ml-datasets/*/*.faa | parallel -j1 "python run_inference.py --reverse {} {}.unirep1900_rev.pkl"

Conda environment configured as follows:
    conda create -n py35tfv1opt python=3.5
    conda activate py35tfv1opt
    pip install https://github.com/lakshayg/tensorflow-build/releases/download/tf1.13.1-ubuntu16.04-py3/tensorflow-1.13.1-cp35-cp35m-linux_x86_64.whl
    pip install pandas click biopython

"""


import numpy as np
import pandas as pd

import tensorflow as tf

import click

import unirep
import data_utils

@click.command()
@click.argument('seq_fn')
@click.argument('out_fn')
@click.option('--batch_size', default=32)
@click.option('--reverse', is_flag=True)
def run_inference(seq_fn, out_fn, batch_size, reverse):
    # set up babbler object
    b = unirep.babbler1900(batch_size=batch_size, model_path="./1900_weights")

    # read sequences
    seqs = series_from_seqio(seq_fn, 'fasta')
    seqs = seqs.str.rstrip('*')
    df_seqs = seqs.to_frame()

    # sort by length
    df_seqs['len'] = df_seqs['seq'].str.len()
    df_seqs.sort_values('len', inplace=True)
    df_seqs.reset_index(drop=True, inplace=True)

    df_seqs['grp'] = df_seqs.groupby('len')['len'].transform(lambda x: np.arange(np.size(x))) // batch_size

    if reverse:
        seqs = seqs.str[::-1]

    # set up tf session, then run inference
    with tf.Session() as sess:
        unirep.initialize_uninitialized(sess)
        df_calc = df_seqs.groupby(['grp', 'len'], as_index=False, sort=False).apply(lambda d: b.get_rep(d['seq'], sess=sess))

    df_calc.to_pickle(out_fn)
    return

def np_to_list(arr):
    return [arr[i] for i in np.ndindex(arr.shape[:-1])]

def aa_seq_to_int(s):
    """Monkey patch to return unknown if not in alphabet
    """
    s = s.strip()
    s_int = [24] + [data_utils.aa_to_int.get(a, data_utils.aa_to_int['X']) for a in s] + [25]
    return s_int[:-1]

def get_rep(self, seqs, sess):
    """
    Monkey-patch get_rep to accept a tensorflow session (instead of initializing one each time)
    """
    if isinstance(seqs, str):
        seqs = pd.Series([seqs])

    coded_seqs = [aa_seq_to_int(s) for s in seqs]
    n_seqs = len(coded_seqs)

    if n_seqs == self._batch_size:
        zero_batch = self._zero_state
    else:
        zero = self._zero_state[0]
        zero_batch = [zero[:n_seqs,:], zero[:n_seqs, :]]

    final_state_, hs = sess.run(
            [self._final_state, self._output], feed_dict={
                self._batch_size_placeholder: n_seqs,
                self._minibatch_x_placeholder: coded_seqs,
                self._initial_state_placeholder: zero_batch
            })

    final_cell_all, final_hidden_all = final_state_
    avg_hidden = np.mean(hs, axis=1)

    df = seqs.to_frame()
    df['seq'] = seqs
    df['final_hs'] = np_to_list(final_hidden_all)[:n_seqs]
    df['final_cell'] = np_to_list(final_cell_all)[:n_seqs]
    df['avg_hs'] = np_to_list(avg_hidden)[:n_seqs]

    return df
unirep.babbler1900.get_rep = get_rep

from Bio import SeqIO
from Bio import AlignIO

def series_from_seqio(fn, format, **kwargs):
    if format in SeqIO._FormatToIterator.keys():
        reader = SeqIO.parse
    elif format in AlignIO._FormatToIterator.keys():
        reader = AlignIO.read
    else:
        raise ValueError("format {} not recognized by either SeqIO or AlignIO".format(format))

    if isinstance(fn, str) and 'gz' in fn:
        with gzip.open(fn, "rt") as fh:
            seqs = reader(fh, format, *kwargs)
    else:
        seqs = reader(fn, format, *kwargs)

    seqs = [(r.id, str(r.seq)) for r in seqs]
    seqs = list(zip(*seqs))
    seqs = pd.Series(seqs[1], index=seqs[0], name="seq")

    return seqs


if __name__ == '__main__':
    run_inference()

