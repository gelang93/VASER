'''
Model:VASER
2020-1-14
'''

from __future__ import print_function
import pickle

from collections import OrderedDict
import sys
import time

import numpy as np
import theano
from theano import config
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from helpers import *
import data_process
from NormalizingRadialFlow import NormalizingRadialFlow
from NormalizingPlanarFlow import NormalizingPlanarFlow
from distributions import *
import shutil, gzip, os, cPickle, time, math, operator, argparse
datasets = {'rsc2015': (data_process.load_data, data_process.prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 42
np.random.seed(SEED)
profile = False
invert_condition = True
radial_flow_type = "Given in the paper on NF"
def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:

        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng, drop_p=0.5):
    retain = 1. - drop_p
    proj = T.switch(use_noise, (state_before * trng.binomial(state_before.shape,
                                                             p=retain, n=1,
                                                             dtype=state_before.dtype)), state_before * retain)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not GRU) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    params['Wemb'] =init_emb()
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])

    ctxdim = options['hidden_units']
    # init_state, init_cell
    params = param_init_fflayer(options, params, prefix='ff_state',
                                nin=200, nout=options['hidden_units'])
    # decoder
    params = param_init_gru_cond_simple(options, params,
                                              prefix='decoder',
                                              nin=options['dim_proj'],
                                              dim=options['hidden_units'],
                                              dimctx=ctxdim)
    # readout
    params = param_init_fflayer(options, params, prefix='ff_logit_lstm',
                                nin=options['hidden_units'], nout=options['dim_proj'],
                                ortho=False)

    # attention
    params['W_encoder'] = init_weights((options['hidden_units'], options['hidden_units']))
    params['W_decoder'] = init_weights((options['hidden_units'], options['hidden_units']))
    params['bl_vector'] = init_weights((1, options['hidden_units']))
    # classifier
    # params['U'] = init_weights((2*options['hidden_units'], options['n_items']))
    # params['b'] = np.zeros((options['n_items'],)).astype(config.floatX)
    params['bili'] = init_weights((options['dim_proj'], options['hidden_units']))

    previous_layer_size = 100
    n_z = 100

    params['q_Z_X_mu_W'] = initialW(previous_layer_size, n_z)
    params['q_Z_X_mu_b'] = numpy.zeros(n_z, dtype=theano.config.floatX)
    params['q_Z_X_sigma_W'] = initialW(previous_layer_size, n_z)
    params['q_Z_X_sigma_b'] = numpy.zeros(n_z, dtype=theano.config.floatX)

     # Weights for outputting normalizing flow parameters
    params['w_us'] = init_weights((100, 500))
    params['b_us'] = init_weights((500,))
    params['w_ws'] = init_weights((100, 500))
    params['b_ws'] = init_weights((500,))
    params['w_bs'] = init_weights((100, 500))
    params['b_bs'] = init_weights((500,))

    return params


def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params

def init_emb():

    emb_list = []
    f_o = open('./data/init_emb', 'r')
    for line in f_o.readlines():
        lines = line.replace('\n', '').split(' ')
        emb = []
        if line == '':
            break
        for i in lines:
            if i != '':
                emb.append(numpy_floatX(i))
        emb_list.append(emb)
    f_o.close()
    np_init_emb = np.array(emb_list)
    return np_init_emb

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def init_weights(shape):
    sigma = np.sqrt(2. / shape[0])
    return numpy_floatX(np.random.randn(*shape) * sigma)

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)

def param_init_gru(options, params, prefix='gru'):
    """
    Init the GRU parameter:

    :see: init_params
    """
    Wxrz = np.concatenate([init_weights((options['dim_proj'], options['hidden_units'])),
                           init_weights((options['dim_proj'], options['hidden_units'])),
                           init_weights((options['dim_proj'], options['hidden_units']))], axis=1)
    params[_p(prefix, 'Wxrz')] = Wxrz

    Urz = np.concatenate([ortho_weight(options['hidden_units']),
                          ortho_weight(options['hidden_units'])], axis=1)
    params[_p(prefix, 'Urz')] = Urz

    Uh = ortho_weight(options['hidden_units'])
    params[_p(prefix, 'Uh')] = Uh

    b = np.zeros((3 * options['hidden_units'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params

def param_init_gru_decoder(options, params, prefix='gru', nin=None, dim=None):

    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = np.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = np.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = np.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = np.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params

def gru_layer(tparams, state_below, options, prefix='gru', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_):
        preact = T.dot(h_, tparams[_p(prefix, 'Urz')])
        preact += x_[:, 0:2 * options['hidden_units']]

        z = T.nnet.hard_sigmoid(_slice(preact, 0, options['hidden_units']))
        r = T.nnet.hard_sigmoid(_slice(preact, 1, options['hidden_units']))
        h = T.tanh(T.dot((h_ * r), tparams[_p(prefix, 'Uh')]) + _slice(x_, 2, options['hidden_units']))

        h = (1.0 - z) * h_ + z * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    state_below = (T.dot(state_below, tparams[_p(prefix, 'Wxrz')]) +
                   tparams[_p(prefix, 'b')])

    hidden_units = options['hidden_units']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=T.alloc(numpy_floatX(0.), n_samples, hidden_units),
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval

def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype(config.floatX)



def tanh(x):
    return T.tanh(x)

def relu(x):
    return T.nnet.relu(x)

def linear(x):
    return x

def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = np.zeros((nout,)).astype(config.floatX)

    return params


def  fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return T.nnet.relu(
        T.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])

# Conditional GRU layer without Attention
def param_init_gru_cond_simple(options, params, prefix='gru_cond', nin=None,
                               dim=None, dimctx=None):
    if nin is None:
        nin = options['hidden_units']
    if dim is None:
        dim = options['hidden_units']
    if dimctx is None:
        dimctx = options['hidden_units']

    params = param_init_gru_decoder(options, params, prefix, nin=nin, dim=dim)

    # context to GRU gates
    Wc = norm_weight(dimctx, dim*2)
    params[_p(prefix, 'Wc')] = Wc

    # context to hidden proposal
    Wcx = norm_weight(dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    return params


def gru_cond_simple_layer(tparams, state_below, options, prefix='gru',
                          mask=None, context=None, one_step=False,
                          init_state=None,
                          **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = T.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = T.alloc(0., n_samples, dim)

    assert context.ndim == 2, 'Context must be 2-d: #sample x dim'
    # projected context to GRU gates
    pctx_ = T.dot(context, tparams[_p(prefix, 'Wc')])
    # projected context to hidden state proposal
    pctxx_ = T.dot(context, tparams[_p(prefix, 'Wcx')])

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x to gates
    state_belowx = T.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]
    # projected x to hidden state proposal
    state_below_ = T.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info|       non-seqs
    def _step_slice(m_, x_, xx_,    h_,       pctx_, pctxx_, U, Ux):
        preact = T.dot(h_, U)
        preact += x_
        preact += pctx_
        preact = T.nnet.sigmoid(preact)

        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = T.dot(h_, Ux)
        preactx *= r
        preactx += xx_
        preactx += pctxx_

        h = T.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    if one_step:
        rval = _step(*(seqs+[init_state, pctx_, pctxx_]+shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state],
                                    non_sequences=[pctx_,
                                                   pctxx_]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval

layers = {'gru': (param_init_gru, gru_layer)}


def adam(loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8, gamma=1-1e-8):

    updates = OrderedDict()
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=config.floatX))

        m = b1_t*m_previous + (1 - b1_t)*g  # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2   # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)               # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)               # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)

        updates[m_previous] = m
        updates[v_previous] = v
        updates[theta_previous] = theta
    updates[t] = t + 1.

    return updates

def cross_entropy_loss(prediction, actual, offset= 1e-4):
    """
    param prediction: tensor of observed values
    param actual: tensor of actual values
    """

    _prediction = T.clip(prediction, offset, 1 - offset)
    ce_loss= - T.sum(actual * T.log(_prediction)
                             + (1 - actual) * T.log(1 - _prediction), 1)

    return ce_loss

def log_diag_mvn(mu, var):
    def f(x):
        # expects batches
        k = mu.shape[1]
        var_new=T.maximum(T.sqrt(var), 1e-4)
        logp = (-k / 2.0) * np.log(2 * np.pi) - 0.5 * T.sum(var_new, axis=1) - T.sum(0.5 * (1.0 / var_new) * (x - mu) * (x - mu), axis=1)
        return logp
    return f


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = T.matrix('x', dtype='int64')
    mask = T.matrix('mask', dtype=config.floatX)
    y = T.vector('y', dtype='int64')
    alpha = T.scalar('alpha', dtype='float32')
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    if options['use_dropout']:
        emb = dropout_layer(emb, use_noise, trng, drop_p=0.25)

    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)

    def compute_alpha(state1, state2):
        tmp = T.nnet.hard_sigmoid(T.dot(tparams['W_encoder'], state1.T) + T.dot(tparams['W_decoder'], state2.T))
        alpha = T.dot(tparams['bl_vector'], tmp)
        res = T.sum(alpha, axis=0)
        return res

    last_h = proj[-1]

    sim_matrix, _ = theano.scan(
        fn=compute_alpha,
        sequences=proj,
        non_sequences=proj[-1]
    )
    att = T.nnet.softmax(sim_matrix.T * mask.T) * mask.T
    p = att.sum(axis=1)[:, None]
    weight = att / p
    atttention_proj = (proj * weight.T[:, :, None]).sum(axis=0)
    proj = T.concatenate([atttention_proj, last_h], axis=1)


    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng, drop_p=0.5)

    ctx_mean = proj
    init_state = fflayer(tparams, ctx_mean, options,
                         prefix='ff_state', activ='tanh')

    us = T.dot(init_state, tparams['w_us']) + tparams['b_us']
    ws = T.dot(init_state, tparams['w_ws']) + tparams['b_ws']
    bs = T.dot(init_state, tparams['w_bs']) + tparams['b_bs']
    flow_params = (us, ws, bs)
    seed = 42
    theano_rng = RandomStreams(numpy.random.RandomState(seed).randint(2 ** 30))
    e_noise = theano_rng.normal(size=(n_samples, 100), avg=0.0, std=1.0)
    q_Z_X_mu = T.dot(init_state, tparams['q_Z_X_mu_W']) + tparams['q_Z_X_mu_b']
    q_Z_X_log_sigma = T.dot(init_state, tparams['q_Z_X_sigma_W']) + tparams['q_Z_X_sigma_b']

    init_state_proj = q_Z_X_mu + T.exp(q_Z_X_log_sigma) * e_noise


    currentClass = NormalizingPlanarFlow(init_state_proj, 100)
    z_k, sum_log_detj, sum_detj_real = currentClass.planar_flow(init_state_proj, flow_params, 5,
                                                                100, invert_condition)

    z0 = init_state_proj                   #512*100
    zk = z_k                               #512*100

    log_q0_z0=log_normal2(z0, q_Z_X_mu, q_Z_X_log_sigma, eps=0.0).sum(axis=1) #1    512*100
    log_p_zk = log_stdnormal(zk).sum(axis=1)  # 3            512*100

    A=log_q0_z0 - log_p_zk
    loss_2=T.mean(A-sum_detj_real)

    proj_decoder = gru_cond_simple_layer(tparams, emb, options,
                                 prefix='decoder',
                                 mask=mask, context=last_h,
                                 one_step=False,
                                 init_state=z_k)

    logit_lstm = fflayer(tparams, proj_decoder, options,
                                    prefix='ff_logit_lstm', activ='linear')

    # log_px_given_zk = T.mean((logit_lstm - emb) ** 2)
    log_px_given_zk = 0.1 * T.mean((logit_lstm - emb) ** 2)                 #4

    ytem = T.dot(tparams['Wemb'], tparams['bili'])
    pred = T.nnet.softmax(T.dot(init_state_proj, ytem.T))
    # pred = T.nnet.softmax(T.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    # f_weight = theano.function([x, mask], weight, name='f_weight')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -T.log(pred[T.arange(n_samples), y] + off).mean() + alpha * loss_2 + log_px_given_zk

    # return use_noise, x, mask, y, f_pred_prob, cost ,tparams['Wemb'],ytem,proj_decoder,loss_2
    return use_noise, x, mask, y, f_pred_prob, cost, alpha

def pred_evaluation(f_pred_prob, prepare_data, data, iterator):
    """
    Compute recall@20 and mrr@20
    f_pred_prob: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    recall = 0.0
    mrr = 0.0
    evalutation_point_count = 0
    # pred_res = []
    # att = []

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  np.array(data[1])[valid_index])
        preds = f_pred_prob(x, mask)
        # weights = f_weight(x, mask)
        targets = y
        ranks = (preds.T > np.diag(preds.T[targets])).sum(axis=0) + 1
        #rank----> 512*1
        rank_ok = (ranks <= 20)
        # pred_res += list(rank_ok)
        recall += rank_ok.sum()
        mrr += (1.0 / ranks[rank_ok]).sum()
        evalutation_point_count += len(ranks)
        # att.append(weights)

    recall = numpy_floatX(recall) / evalutation_point_count
    mrr = numpy_floatX(mrr) / evalutation_point_count
    eval_score = (recall, mrr)

    return eval_score

def wen_pred_evaluation(f_pred_prob, prepare_data, data, iterator):
    """
    Compute recall@20 and mrr@20
    f_pred_prob: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    recall = 0.0
    mrr = 0.0
    evalutation_point_count = 0
    # pred_res = []
    # att = []

    ff = open('./show_length_result', 'w')
    f_2 = open('./show_all_result', 'w')
    # f_3 =  open('./show_false_result', 'w')
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  np.array(data[1])[valid_index])
        preds = f_pred_prob(x, mask)
        # weights = f_weight(x, mask)
        targets = y
        ranks = (preds.T > np.diag(preds.T[targets])).sum(axis=0) + 1
        rank_ok = (ranks <= 20)

        for i in range(len(rank_ok)):
            ff.write(str(rank_ok[i])+' ')
            f_2.write(str(rank_ok[i]) + ' ')
            a = []
            for j in x.T[i].tolist():

                if str(j) != '0':
                    a.append(str(j))
                    f_2.write(str(j) + ' ')
            ff.write(str(len(a)))

            ff.write('\n')
            f_2.write('\n')


        # pred_res += list(rank_ok)
        recall += rank_ok.sum()
        mrr += (1.0 / ranks[rank_ok]).sum()
        evalutation_point_count += len(ranks)
        # att.append(weights)

    recall = numpy_floatX(recall) / evalutation_point_count
    mrr = numpy_floatX(mrr) / evalutation_point_count
    eval_score = (recall, mrr)

    return eval_score


def train_gru(
    dim_proj=50,  # word embeding dimension
    hidden_units=100,  # GRU number of hidden units.
    patience=100,  # Number of epoch to wait before early stop if no progress
    max_epochs=30,  # The maximum number of epoch to run
    dispFreq=100,  # Display to stdout the training progress every N updates
    lrate=0.001,  # Learning rate
    n_items=43098,  # Vocabulary size
    # n_items=9013,
    # n_items=23384,
    encoder='gru',  # TODO: can be removed must be gru.
    saveto='gru_model.npz',  # The best model will be saved there
    is_valid=True,  # Compute the validation error after this number of update.
    is_save=False,  # Save the parameters after every saveFreq updates
    batch_size=512,  # The batch size during training.
    valid_batch_size=512,  # The batch size used for validation/test set.
    dataset='rsc2015',

    # Parameter for extra option
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
):

    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    load_data, prepare_data = get_dataset(dataset)

    print('Loading data')
    train, valid, test = load_data()

    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('gru_model.npz', params)

    tparams = init_tparams(params)


    (use_noise, x, mask,
     y, f_pred_prob, cost, alpha) = build_model(tparams, model_options)
    all_params = list(tparams.values())

    updates = adam(cost, all_params, lrate)                                                                                      #####----------
    train_function = theano.function(inputs=[x, mask, y,alpha], outputs=cost, updates=updates)

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    history_vali = []
    best_p = None
    bad_count = 0

    uidx = 0  # the number of update done
    estop = False  # early stop
    min_kl_epoch = 0.0
    min_kl = 0.0
    lens = 0.001
    alpha = 0.0
    try:
        for eidx in range(max_epochs):
            start_time = time.time()
            n_samples = 0
            epoch_loss = []

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]
                if eidx >=3:
                    if min_kl_epoch<1.0:
                        min_kl_epoch = min_kl + (eidx-3)* 0.0001
                    alpha=min_kl_epoch


                loss = train_function(x, mask, y,alpha)

                epoch_loss.append(loss)

                # if np.isnan(loss) or np.isinf(loss):
                #     print('bad loss detected: ', loss)
                #     return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Loss ', np.mean(epoch_loss),'   alpha',alpha)

            if saveto and is_save:
                print('Saving...')

                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                np.savez(saveto, history_errs=history_errs, **params)
                print('Saving done')

            if is_valid:
                use_noise.set_value(0.)

                valid_evaluation = pred_evaluation(f_pred_prob, prepare_data, valid, kf_valid)
                test_evaluation = pred_evaluation(f_pred_prob, prepare_data, test, kf_test)


                history_errs.append([valid_evaluation, test_evaluation])

                if best_p is None or valid_evaluation[0] >= np.array(history_vali).max():

                    best_p = unzip(tparams)
                    print('Best perfomance updated!')
                    bad_count = 0

                print('Valid Recall@20:', valid_evaluation[0], '   Valid Mrr@20:', valid_evaluation[1],
                      '\nTest Recall@20', test_evaluation[0], '   Test Mrr@20:', test_evaluation[1])

                if len(history_vali) > 10 and valid_evaluation[0] <= np.array(history_vali).max():
                    bad_count += 1
                    print('===========================>Bad counter: ' + str(bad_count))
                    print('current validation recall: ' + str(valid_evaluation[0]) +
                          '      history max recall:' + str(np.array(history_vali).max()))
                    if bad_count > patience:
                        print('Early Stop!')
                        estop = True

                history_vali.append(valid_evaluation[0])

            end_time = time.time()
            print('Seen %d samples' % n_samples)
            print(('This epoch took %.1fs' % (end_time - start_time)), file=sys.stderr)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")
        test_evaluation = wen_pred_evaluation(f_pred_prob, prepare_data, test, kf_test)


    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    valid_evaluation = pred_evaluation(f_pred_prob, prepare_data, valid, kf_valid)
    test_evaluation = pred_evaluation(f_pred_prob,  prepare_data, test, kf_test)

    print('=================Best performance=================')
    print('Valid Recall@20:', valid_evaluation[0], '   Valid Mrr@20:', valid_evaluation[1],
          '\nTest Recall@20', test_evaluation[0], '   Test Mrr@20:', test_evaluation[1])
    print('==================================================')
    if saveto and is_save:
        np.savez('Best_performance', valid_evaluation=valid_evaluation, test_evaluation=test_evaluation, history_errs=history_errs,
                 **best_p)

    return valid_evaluation, test_evaluation


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    eval_valid, eval_test = train_gru(max_epochs=500, test_size=-1)
