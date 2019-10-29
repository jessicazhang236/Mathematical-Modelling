import numpy as np
import scipy.linalg as la
import argparse

dt = 0.01
start = 0.0
end = 10.0
n = 0
p = 3
dim = 3

def u(t): # evaluates parametric value from timestamp
    return (int)((t - start) * 1.0 / dt)

def n_coeff(t, _T):
    T = np.concatenate((_T, np.zeros((5))))
    N = np.zeros(n + 2)
    if t == start:
        N[0] = 1.0
        return N
    if t == end:
        N[n] = 1.0
        return N
    k = u(t)
    N[k] = 1.0
    for _ in range(p):
        d = _ + 1
        N[k-d] = N[k-d+1] * (T[k+1] - t) / (T[k+1] - T[k-d+1])
        i = k - d + 1
        while (i <= k - 1):
            N[i] = N[i] * (t - T[i]) / (T[i+d] - T[i]) + N[i+1] * (T[i+d+1] - t) / (T[i+d+1] - T[i+1])
            i += 1
        N[k] *= (t - T[k]) / (T[k+d] - T[k])
    return N

def setup(src): # setting the global values
    global dt, start, end, n
    time_src = np.load(src)
    start = time_src[0]
    n = time_src.size - 1
    end = time_src[n]
    dt = (end - start) * 1.0 / n
    T = np.concatenate((np.arange(start, end, dt), np.asarray([end])))
    N = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        N[i] = n_coeff(T[i], T)[:n+1]
    return T, N

def deBoor(t, T, P):
    k = u(t)
    h = p
    s = 0
    if t == T[k]:
        h = p - 1
        s = 1
    ctr_pts = np.zeros((k+2, h+2, dim))
    
    j = k - s
    while (j >= max(0, k - p)):
        ctr_pts[j][0] = P[j]
        j -= 1

    for _ in range(h):
        r = _ + 1
        i = max(0, k - p + r)
        while (i <= k - s):
            temp = (t - T[i]) * 1.0 / (T[i+p-r+1] - T[i])
            ctr_pts[i][r] = (1.0 - temp) * ctr_pts[i-1][r-1] + temp * ctr_pts[i][r-1]
            i += 1

    return ctr_pts[k-s][p-s]

def testing(fsrc, fgt, log, T, P):
    src = np.load(fsrc)[(p+2)*4:]
    gt = np.load(fgt)[(p+2)*4:, :3]
    loss = 0.0
    pred = gt
    for i in range(src.size):
        estimate = deBoor(src[i], T, P)
        loss += np.linalg.norm(np.subtract(estimate, gt[i]))
        pred[i] = estimate
    print('Finished testing with average error %.6f.' % (loss/src.size))
    output = log + '_cubic_b-spline_predictions.npy'
    print('Dumping data to ' + output)
    np.save(output, pred)

def parse_args(argv):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--interval', type=int, default=1000, help='the number of intervals the total time should be divided into')
    parser.add_argument('--log_id', type=str, default='None', help='the GUID for the log that was preprocessed using the read_poses.py script')
    parser.add_argument('--train_src', type=str, default='None', help='the file that contains the training input')
    parser.add_argument('--train_tgt', type=str, default='None', help='the file that contains the training output')
    parser.add_argument('--test_src', type=str, default='None', help='the file that contains the testing input')
    parser.add_argument('--test_tgt', type=str, default='None', help='the file that contains the testing output')
    args = parser.parse_args(argv)
    return args

def main(argv=None):
    args = parse_args(argv)
    log = args.log_id
    tr_src = args.train_src
    tr_tgt = args.train_tgt
    tst_src = args.test_src
    tst_tgt = args.test_tgt
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
    
    T, N = setup(tr_src) # timestamps of all the control points and the coefficient matrix
    D = np.load(tr_tgt)[..., :3]
    dim = D[0].size
    # P = np.matmul(np.linalg.inv(N), D) # control point array
    P = np.linalg.lstsq(N, D)[0]
    print(P)
    testing(tst_src, tst_tgt, log, T, P)

if __name__ == '__main__':
    main()
