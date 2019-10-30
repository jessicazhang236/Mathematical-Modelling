import numpy as np
import argparse

def eval(coeff, x):
    dim = np.size(coeff, 1)
    ret = np.zeros((x.size, dim))
    for i in range(dim):
        ret[..., i] = np.polyval(coeff[..., i], x)
    return ret

def fitpoly(fsrc, ftgt, order, tr_err):
    x = np.load(fsrc)
    y = np.load(ftgt)[..., :3]
    max_err = tr_err
    fit = None
    for i in y:
        for z in i:
            if z == np.nan:
                print(i, ', ', z)
                exit()
    for deg in range(order):
        model = np.polyfit(x, y, deg)
        print(deg)
        estimate = eval(model, x)
        loss = np.linalg.norm(np.subtract(estimate, y))
        if loss < max_err:
            max_err = loss
            fit = model
    return fit

def test(fsrc, ftgt, model, log):
    x = np.load(fsrc)
    y = np.load(ftgt)[..., :3]
    pred = eval(model, x)
    error = np.linalg.norm(np.subtract(pred, y))
    output = log + '_polynomial_predictions.npy'
    print('Finished testing with average loss %.6f.' % error)
    print('Dumping predictions to ' + output)
    np.save(output, pred)

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_id', type=str, default='None', help='the GUID for the log that was preprocessed using the read_poses.py script')
    parser.add_argument('--train_src', type=str, default='None', help='the file that contains the training input')
    parser.add_argument('--train_tgt', type=str, default='None', help='the file that contains the training output')
    parser.add_argument('--test_src', type=str, default='None', help='the file that contains the testing input')
    parser.add_argument('--test_tgt', type=str, default='None', help='the file that contains the testing output')
    parser.add_argument('--degree_threshold', type=int, default=20, help='the maximum degree that the polynomial should attempt')
    parser.add_argument('--error_threshold', type=float, default=1.0, help='the maximum training error for the approximated model')
    args = parser.parse_args(argv)
    return args
    
def main(argv=None):
    args = parse_args(argv)
    log = args.log_id
    tr_src = args.train_src
    tr_tgt = args.train_tgt
    tst_src = args.test_src
    tst_tgt = args.test_tgt
    order = args.degree_threshold
    tr_err = args.error_threshold
    if log == 'None':
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
            exit()
    else:
        tr_src = log + '_training_time.npy'
        tr_tgt = log + '_training_pose.npy'
        tst_src = log + '_testing_time.npy'
        tst_tgt = log + '_testing_pose.npy'
    model = fitpoly(tr_src, tr_tgt, order, tr_err)
    test(tst_src, tst_tgt, model, log)

if __name__ == '__main__':
    main()
