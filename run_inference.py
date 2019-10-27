"""
Runs embedding/inference for an input fasta file:

    ls -S -r ../github/ml-datasets/*/*.faa | parallel -j1 "python run_inference.py {} {}.unirep1200.pkl" 
    ls -S -r ../github/ml-datasets/*/*.faa | parallel -j1 "python run_inference.py --reverse {} {}.unirep1200_rev.pkl" 

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
@click.option('--reverse', is_flag=True)
def run_inference(seq_fn, out_fn, reverse):
    # set up babbler object
    b = unirep.babbler1900(batch_size=1, model_path="./1900_weights")
    
    # read sequences
    seqs = series_from_seqio(seq_fn, 'fasta')
    seqs = seqs.str.rstrip('*')

    if reverse:
        seqs = seqs.str[::-1]

    # set up tf session, then run inference
    with tf.Session() as sess:
        unirep.initialize_uninitialized(sess)
        seqs_pred = seqs.apply(b.get_rep, sess=sess)

    seqs_pred.to_pickle(out_fn)
    return 


def aa_seq_to_int(s):
    """Monkey patch to return unknown if not in alphabet
    """
    return [24] + [data_utils.aa_to_int.get(a, data_utils.aa_to_int['X']) for a in s] + [25]

def get_rep(self, seq, sess):
    """
    Monkey-patch get_rep to accept a tensorflow session (instead of initializing one each time)
    """
    int_seq = aa_seq_to_int(seq.strip())[:-1]

    final_state_, hs = sess.run(
            [self._final_state, self._output], feed_dict={
                self._batch_size_placeholder: 1,
                self._minibatch_x_placeholder: [int_seq],
                self._initial_state_placeholder: self._zero_state}
        )

    final_cell, final_hidden = final_state_

    # Drop the batch dimension so it is just seq len by
    # representation size
    final_cell = final_cell[0]
    final_hidden = final_hidden[0]
    hs = hs[0]

    avg_hidden = np.mean(hs, axis=0)
    return pd.Series({
        'seq': seq,
        'avg_hs': avg_hidden,
        'final_hs': final_hidden,
        'final_cell': final_cell
        })

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

