import numpy as np
import numpy.linalg as la
import argparse

dt = 0.01
start = 0.0
end = 10.0
n = 0
p = 3
dim = 6

log_list = []
def u(t): # evaluates parametric value from timestamp
    if t < start:
        return 2
    if t >= end:
        return n + 3
    return (int)((t - start) * 1.0 / dt) + 3

def n_coeff(t, _T):
    T = np.concatenate((_T, _T[_T.size - 1] * np.ones((8))))
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
        if T[k+1] != T[k-d+1]:
            N[k-d] = N[k-d+1] * (T[k+1] - t) / (T[k+1] - T[k-d+1])
        i = k - d + 1
        while (i <= k - 1):
            if T[i+d] != T[i] and T[i+d+1] != T[i+1]:
                N[i] = N[i] * (t - T[i]) / (T[i+d] - T[i]) + N[i+1] * (T[i+d+1] - t) / (T[i+d+1] - T[i+1])
            i += 1
        if T[k+d] != T[k]:
            N[k] *= (t - T[k]) / (T[k+d] - T[k])
    return N

def setup(time_src): # setting the global values
    # time_src = np.load(ftime_src)
    global dt, start, end, n
    start = time_src[0]
    n = time_src.size - 1
    end = time_src[n]
    dt = (end - start) * 1.0 / n
    front_knot = np.ones(3) * time_src[0]
    back_knot = np.ones(3) * time_src[n]
    T = time_src 
    # T = np.concatenate((front_knot, time_src, back_knot))
    # T = np.concatenate((np.arange(start, end, dt), np.asarray([end])))
    # N = np.zeros((n + 1, n + 1))
    # for i in range(n + 1):
        # N[i] = n_coeff(T[i], T[3:])[:n+1]
    return T

"""def deBoor(t, T, P):
    T = np.concatenate((T, T[T.size - 1] * np.ones(6)))
    # P = np.concatenate((P, np.zeros((3, 6))))
    k = u(t)
    h = p
    s = 0
    if t == T[k]:
        h = p - 1
        s = 1
    ctr_pts = np.zeros((k+5, h+5, dim))
    
    j = k - s

    while (j >= max(0, k - p)):
        ctr_pts[j][0] = P[j]
        j -= 1

    for _ in range(h):
        r = _ + 1
        i = max(0, k - p + r)
        
00f0ad9c-27c0-8bdb-1db1-2610fd01a788_testing_pose.npy        while (i <= k - s and i+p-r+1 < T.size):
            if T[i+p-r+1] == T[i]:
                i += 1
            else:
                temp = (t - T[i]) * 1.0 / (T[i+p-r+1] - T[i])
                ctr_pts[i][r] = (1.0 - temp) * ctr_pts[i-1][r-1] + temp * ctr_pts[i][r-1]
                i += 1

    return ctr_pts[k-s][p-s]"""

"""def Nmat(t, T, k):
    i = u(t)
    N = np.zeros((n + 2, k + 2))
    DP = np.zeros(k)
    DM = DP
    N[1, 1] = 1
    s = 1
    while s <= k - 1:
        DP[s] = T[i+s] - t
        DM[s] = t - T[i-s+1]
        N[1, s+1] = 0
        r = 1
        while r <= s:
            tmp = 0
            if DP[r] + DM[s + 1 -r] != 0:
                tmp = N[r, s] / (DP[r] + DM[s + 1 - r])
            N[r, s+1] = N[r, s+1] + DP[r] * tmp
            N[r+1, s+1] = DM[s+1-r] * tmp
            r += 1
        s += 1
    return N

def deBoor(t, T, P):
    k = p + 1 # p + 1 or p; try both
    N = Nmat(t, T, k)
    ans = np.zeros(dim)
    i = max(0, u(t) - k + 1)
    while i <= u(t):
        ans += P[i] * N[i, k]
        i += 1
    return ans
"""

def deBoor(t, T, P):
    k = u(t)
    d = [P[j + k - p] for j in range(0, p+1)]
    for r in range(1, p+1):
        for j in range(p, r-1, -1):
            tmp = 0
            if T[j+1+k-r] != T[j+k-p]:
                tmp = (t - T[j + k - p]) * 1.0 / (T[j + 1 + k -r] - T[j + k - p])
            d[j] = (1.0 - tmp) * d[j - 1] + tmp * 1.0 * d[j]
    return d[p]

def calc(t, T, P):
    k = u(t)
    coeff = np.asarray([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 0, 3, 0], [1, 4, 1, 0]]) / 6.0
    tmat = np.asarray([t ** 3, t ** 2, t, 1])
    pmat = np.asarray([P[k], P[k+1], P[k+2], P[k+3]])
    return np.matmul(tmat, np.matmul(coeff, pmat))

def testing(fsrc, fgt, log, T, P, av, ind, out):
    src = fsrc
    gt = fgt[..., :dim]
    # src = fsrc
    # gt = fgt
    loss = 0.0
    pred = gt
    if src.size != 0:
        for i in range(src.size):
            estimate = deBoor(src[i], T, P)
            # estimate = calc(src[i], T, P)
            # print(estimate)
            # if np.isnan(np.linalg.norm(np.subtract(estimate, gt[i]))):
                # print(estimate)
            loss += np.linalg.norm(np.subtract(estimate, gt[i]))
            # print(pred[i].shape)
            # print(estimate.shape)
            pred[i] = estimate
        # print(loss)
        # print(src.size)
        print('Finished testing with average error %.6f.' % (loss/src.size))
        output = 'pose_data/' + out + '/' + log + '/' + log + '_%.6d_cubic_b-spline_predictions.npy' % ind
        # output = log + '_cubic_b-spline_predictions.npy'
        print('Dumping data to ' + output)
        np.save(output, pred + av)

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

def relative(trs, trt, tss, tst):
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
            exit()
    else:
        tr_src = log + '_training_time.npy'
        tr_tgt = log + '_training_pose.npy'
        tst_src = log + '_testing_time.npy'
        tst_tgt = log + '_testing_pose.npy'
    
    T, N = setup(tr_src) # timestamps of all the control points and the coefficient matrix
    D = np.load(tr_tgt)[..., :dim]
    # dim = D[0].size
    # P = np.matmul(np.linalg.inv(N), D) # control point array
    P = np.linalg.lstsq(N, D)[0]
    print(P)
    testing(tst_src, tst_tgt, log, T, P)"""

    dir_lst = np.asarray(['1_0.1_50_100'])
    spec_lst = np.asarray([[1, 0.1, 50, 100]])

    for lg in log_list:
        for j in range(dir_lst.size):
            tr_src = 'pose_data/' + dir_lst[j] + '/' + lg + '/' + lg + '_training_time.npy'   
            tr_tgt = 'pose_data/' + dir_lst[j] + '/' + lg + '/' + lg + '_training_pose.npy'
            tst_src = 'pose_data/' + dir_lst[j] + '/' + lg + '/' + lg + '_testing_time.npy'
            tst_tgt = 'pose_data/' + dir_lst[j] + '/' + lg + '/' + lg + '_testing_pose.npy'
            trs_ = np.load(tr_src)
            trt_ = np.load(tr_tgt)
            tss_ = np.load(tst_src)
            tst_ = np.load(tst_tgt)
            duration = trs_.size
            tmp = (int)(spec_lst[j][0] * spec_lst[j][2])
            tmp2 = (int)(spec_lst[j][0] * spec_lst[j][3])
            tmp3 = (int)(spec_lst[j][1] * spec_lst[j][3])
            for i in range(min((int)(duration/tmp), (int)(tss_.size/tmp2))):
                tr_ind = min((i+1)*tmp, duration-1)
                trs = trs_[i*tmp:tr_ind]
                trt = trt_[i*tmp:tr_ind]

                tst_st = max(0, i*tmp2)
                tst_lst = min(tss_.size-1, (i+1)*tmp2)
                tss = tss_[tst_st:tst_lst]
                tst = tst_[tst_st:tst_lst]
                
                av = trt[0, :dim]
                trs, trt, tss, tst = relative(trs, trt, tss, tst)

                # T_, N = setup(trs)
                T_ = setup(trs)
                D = trt[..., :dim]
                # P = D
                T = np.concatenate((T_[0] * np.ones(3), T_, T_[n] * np.ones(3)))
                P = np.vstack((D[0], D[0], D[0], D, D[n], D[n], D[n]))
                # D[0] *= 2
                # D[D.size/dim - 1] *= 2
                # print(N.shape)
                # print(trt[..., :dim].shape)
                # P_ = la.lstsq(N, D)[0]
                
                # P = np.vstack((D[0], D, D[n], D[n], D[n]))
                # print(P)
                testing(tss, tst[..., :dim], lg, T, P, av, i, dir_lst[j])
                # testing(trs[i*tmp:tr_ind], trt[i*tmp:tr_ind, :dim], lg, T, P, i, dir_lst[j])

if __name__ == '__main__':
    main()
