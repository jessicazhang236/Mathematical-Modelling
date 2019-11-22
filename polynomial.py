import warnings
import numpy as np
import numpy.linalg as la
import numpy.polynomial.polynomial as poly
import argparse

warnings.filterwarnings('ignore', message='RankWarning')
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

log_list = np.array(['00f0ad9c-27c0-8bdb-1db1-2610fd01a788'])
"""'0224916e-bb15-41bb-ef72-38a39923554a',
'02990aeb-3649-3188-18b1-ffc47c03185c',
'03548b25-b4bd-d3cb-7b21-a56887607ae8',
'03d2ab0f-4dbc-8863-43cd-9071ed265189',
'05cd1dc7-d73d-ed63-00e7-dbd51c900184',
'067e62dc-fe8e-4109-eee9-8a70e2f037bc',
'0696e5c3-9b1b-2754-1693-4d3c35e2c5c5',
'0711a84c-edb3-eded-8fda-1fed9aeda4be',
'082788fe-461e-4d30-fa94-efed1d508192',
'08abc439-72fa-4c86-f266-954226e45877',
'08c94aea-2b8d-40f3-d90b-8f6a732a296b',
'09d9227e-08bd-dfc1-1a75-33a1953c2cc3',
'0d367004-cd84-4946-fc72-a40d7c0aedb3',
'0df8eef4-4352-86f5-07c3-317ba54348b4',
'0f88ddc8-c2c0-4f44-1fc1-4e59310912be',
'10f8e3b9-5ab7-5e56-97e0-4701a56965e3',
'15ecd6e5-495e-747e-9d79-ebe6806f4fd3',
'18257908-e088-4a08-c6cb-c7bbd2a47922',
'239b269e-e15b-c7f6-3d26-e4df89497014',
'27581d62-01d2-5cc6-462f-4440435f9fc5',
'286310ad-bb7e-cefd-ebdf-c6d9e4715417',
'2b998b1d-50b7-9d81-a898-7a44f0230572',
'2bd6e8d5-112f-7ea7-c248-ae8507429e38',
'41e6c9f9-f1bb-5cfb-e923-eddd05b08464',
432aa902-c28d-9e73-a6f5-31934a2f95d3',
'43677200-e6f0-2051-5e6a-5e6fc3851fde',
'4763a435-5887-4459-ee65-fb6075217a35',
'4853043c-065a-462f-c9f0-67f8263a0eb6',
'5da97c51-65cd-a987-dffb-8b7599aa0c0b',
'632c320b-1b9e-de97-663e-1f78d0082617'])
"""
def evaluate(coeff, x):
    # print(coeff)
    dim = np.size(coeff, 1)
    ret = np.zeros((x.size, dim))

    for i in range(dim):
        ret[..., i] = np.polyval(coeff[..., i], x)
    return ret

def vandermonde(x, deg):
    x = np.array(x, copy=False, ndmin=1) + 0.0
    dims = (deg + 1,) + x.shape
    van = np.ones(dims)

    if deg > 0:
        van[1] = x
        for i in range(2, deg + 1):
            van[i] = van[i-1] * x

    return np.moveaxis(van, 0, -1)

def ofit(x, y, deg, rcond=None, full=False, w=None):
    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0
    deg = np.asarray(deg)

    if deg.ndim > 1 or deg.dtype.kind not in 'iu' or deg.size == 0:
        raise TypeError('deg must be an int or non-empty 1-D array of int')
    if deg.min() < 0:
        raise ValueError('expected deg >= 0')
    if x.ndim != 1:
        raise TypeError('expected 1D vector for x')
    if x.size == 0:
        raise TypeError('expected non-empty vector for x')
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError('expected 1D or 2D array for y')
    if len(x) != len(y):
        raise TypeError('expected x and y to have same length')

    if deg.ndim == 0:
        lmax = deg
        order = lmax + 1
        van = poly.polyvander(x, lmax)
    else:
        deg = np.sort(deg)
        lmax = deg(-1)
        order = len(deg)
        van = poly.polyvander(x, lmax)[:, deg]

    lhs = van.T
    rhs = y.T
    if w is not None:
        w = np.asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError('expected 1D vector for w')
        if len(x) != len(w):
            raise TypeError('expected x and w to have same length')
        lhs = lhs * w
        rhs = rhs * w

    if rcond is None:
        rcond = len(x) * np.finfo(x.dtype).eps
    if issubclass(lhs.dtype.type, np.complexfloating):
        scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
    else:
        scl = np.sqrt(np.square(lhs).sum(1))
    scl[scl == 0] = 1

    c, resids, rank, s = la.lstsq(lhs.T/scl, rhs.T, rcond)
    c = (c.T/scl).T

    if deg.ndim > 0:
        if c.ndim == 2:
            cc = np.zeros((lmax+1, c.shape[1]), dtype=c.dtype)
        else:
            cc = np.zeros(lmax+1, dtype=c.dtype)
        cc[deg] = c
        c = cc

    if rank != order and not full:
        msg = 'The fit may be poorly conditioned'
        warnings.warn(msg, RankWarning, stacklevel=2)
    
    if full:
        return c, [resids, rank, s, rcond]
    else:
        return c

def fitting(x, y, deg):
    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0
    deg = np.asarray(deg)

    if deg.ndim == 0:
        lm = deg
        order = lm + 1
        van = poly.polyvander(x, lm)
 
    else:
        deg = np.sort(deg)
        lm = deg[-1]
        order = len(deg)
        van = poly.polyvander(x, lm)[..., deg]

    ls = van.T
    rs = y.T
    scl = np.sqrt(np.square(ls).sum(1))
    scl[scl == 0] = 1
    
    coeff, res, rank, s = la.lstsq(ls.T / scl, rs.T, rcond=len(x)*np.finfo(x.dtype).eps)
    coeff = (coeff.T/scl).T
    
    if deg.ndim > 0:
        if coeff.ndim == 2:
            exp = np.zeros((lm + 1, coeff.shape[1]), dtype=coeff.dtype)

        else:
            exp = np.zeros(lm + 1, dtype=coeff.dtype)
        
        exp[deg] = coeff
        coeff = exp

    return coeff

def fitpoly(x, y, order, tr_err):
    # max_err = tr_err
    max_err = np.inf
    fit = None

    for i in y:
        for z in i:
            if z == np.nan:
                print(i, ', ', z)
                exit()

    for deg in range(order):
        model = ofit(x, y, deg)
        estimate = evaluate(model, x)
        loss = np.linalg.norm(np.subtract(estimate, y))

        if loss < max_err:
            max_err = loss
            fit = model

    return fit

def test(x, y, model, log, average, ind, out):
    pred = evaluate(model, x)
    error = np.linalg.norm(np.subtract(pred, y))
    # print(pred, y)
    output = '/mnt/yyz_data_2/users/jessicaz/pose_data/' + out + '/' + log + '/' + log + '_%.6d_polynomial_predictions.npy' % ind

    if error >= 10.0:
        print('Finished testing %.5d with average loss %.6f.' % (ind, error))
    print('Dumping predictions to ' + output)
    np.save(output, pred + average)

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_id', type=str, default='None', help='the GUID for the log that was preprocessed using the read_poses.py script')
    parser.add_argument('--train_src', type=str, default='None', help='the file that contains the training input')
    parser.add_argument('--train_tgt', type=str, default='None', help='the file that contains the training output')
    parser.add_argument('--test_src', type=str, default='None', help='the file that contains the testing input')
    parser.add_argument('--test_tgt', type=str, default='None', help='the file that contains the testing output')
    parser.add_argument('--degree_threshold', type=int, default=5, help='the maximum degree that the polynomial should attempt')
    parser.add_argument('--error_threshold', type=float, default=0.1, help='the maximum training error for the approximated model')
    args = parser.parse_args(argv)
    return args
    
# loc_list = np.asarray(['10_1_10_30', '1_0.1_10_30'])
spec_lst = np.asarray([[1, 0.1, 50, 100]])
loc_list = np.asarray(['1_0.1_50_100'])
log_size = np.zeros((log_list.size, loc_list.size))

def preproc(trs, trt, tss, tst):
    tmp1 = trs[0]
    tmp2 = trt[0]
    return trs - tmp1, trt - tmp2, tss - tmp1, tst - tmp2

def main(argv=None):
    args = parse_args(argv)
    log = args.log_id
    tr_src = args.train_src
    tr_tgt = args.train_tgt
    tst_src = args.test_src
    tst_tgt = args.test_tgt
    order = args.degree_threshold
    tr_err = args.error_threshold

    """if log == 'None':
        if tr_src == 'None':
            print('Missing training input data')
            exit()
        if tr_tgt == 'None':
            print('Missing training output data')
            exit()
        if tst_src == 'None':
            print('Missing testing input data')
            exit()
        if tst_tgt == 'None':
            print('Missing testing output data')
            exit()"""

    if log == 'None':
        for k in range(log_list.size):
            lg = log_list[k]
            for j in range(loc_list.size):
                pre = '/mnt/yyz_data_2/users/jessicaz/'
                tr_src = pre + 'pose_data/' + loc_list[j] + '/' + lg + '/' + lg + '_training_time.npy'   
                tr_tgt = pre + 'pose_data/' + loc_list[j] + '/' + lg + '/' + lg + '_training_pose.npy'
                tst_src = pre + 'pose_data/' + loc_list[j] + '/' + lg + '/' + lg + '_testing_time.npy'
                tst_tgt = pre + 'pose_data/' + loc_list[j] + '/' + lg + '/' + lg + '_testing_pose.npy'

                """else:
                tr_src = log + '_training_time.npy'
                tr_tgt = log + '_training_pose.npy'
                tst_src = log + '_testing_time.npy'
                tst_tgt = log + '_testing_pose.npy'"""

                trs_ = np.load(tr_src)
                trt_ = np.load(tr_tgt)
                tss_ = np.load(tst_src)
                tst_ = np.load(tst_tgt)
                # av = trt_[0, :3]
                # trs_, trt_, tss_, tst_ = preproc(trs_, trt_, tss_, tst_)

                tmp1 = (int)(spec_lst[j][0] * spec_lst[j][2])
                tmp2 = (int)(spec_lst[j][0] * spec_lst[j][3])
                tmp3 = (int)(spec_lst[j][1] * spec_lst[j][3])
                duration = trs_.size

                for i in range(min((int)(duration/tmp1), (int)(tss_.size/tmp2))):
                    tst_st = max(0, i*tmp2)
                    tst_lst = min(tss_.size-1, (i+1)*tmp2)
                    tr_ind = min((i+1)*tmp1, duration-1)
                    trs = trs_[i*tmp1:tr_ind]
                    trt = trt_[i*tmp1:tr_ind]

                    av = trt[0]
                    tss = tss_[tst_st:tst_lst]
                    tst = tst_[tst_st:tst_lst]
                    trs, trt, tss, tst = preproc(trs, trt, tss, tst)

                    model = fitpoly(trs, trt, order, tr_err)
                    log_size[k, j] += 1
                    test(tss, tst, model, lg, av, i, loc_list[j])
                    # np.save('training_time.npy', trs)
                    # np.save('training_pose.npy', trt)
                    # np.save('testing_time.npy', tss)
                    # np.save('testing_pose.npy', tst)
                    # exit()
                    # test(trs[i*tmp1:tr_ind], trt[i*tmp1:tr_ind], model, lg, av, i, loc_list[j])

    else:
        tr_src = log + '_training_time.npy'
        tr_tgt = log + '_training_pose.npy'
        tst_src = log + '_testing_time.npy'
        tst_tgt = log + '_testing_pose.npy'

        trs = np.load(tr_src)
        trt = np.load(tr_tgt)
        tss = np.load(tst_src)
        tst = np.load(tst_tgt)
        model = fitpoly(trs, trt, order, tr_err)
        test(tss, tst, model, log)

    np.save('log_size.npy', log_size)

if __name__ == '__main__':
    main()
