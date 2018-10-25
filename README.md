# 2018PRCV_competition

This repo provides the solutions for 2018 PRCV_comptition Large Scale Person Re-identification & Attributes Retrival by team Astelli.   

Both solutions depends on <a href='https://github.com/keras-team/keras'>Keras</a> and <a href='https://github.com/tensorflow/tensorflow'>Tensorflow</a>. The solution for attributes retrival **Ranked No.2** in the contest.

Contributed by Binyu Wang(zbzdwby819@gmail.com), Du Tang(dutang@outlook.com) and Harley Li(harley.li@connect.polyu.hk).


# Installation

``pip install -r requirements.txt``


# Highlights in Attr Model

*Model*: Multi-heads output network

# Highlights in ReID Model

*Model*: Inspired a lot by this <a href='https://zhuanlan.zhihu.com/p/40514536'>article</a>. Deep metric learing + Softmax classification.

*Loss function*: Provided Batch_Hard_mining(BH) Triplet loss<a href='https://arxiv.org/abs/1703.07737'>[1]</a>, BH Triplet with soft margin<a href='https://arxiv.org/abs/1703.07737'>[1]</a> and MSML<a href='https://arxiv.org/abs/1710.00478'>[2]</a>.

*Evaluation*: Implemented <a href='https://www.wikiwand.com/en/Evaluation_measures_(information_retrieval)'>MAP</a> metric using tensorflow.


# Reference

[1] <a href='https://arxiv.org/abs/1703.07737'>In Defense of the Triplet Loss for Person Re-Identification</a>

[2] <a href='https://arxiv.org/abs/1710.00478'>Margin Sample Mining Loss: A Deep Learning Based Method for Person Re-identification</a>


# Thank you!
