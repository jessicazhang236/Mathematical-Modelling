from math import *
import numpy as np
import numpy.linalg as la
import argparse

dt = 0.01
start = 0.0
end = 10.0
n = 0
p = 3
dim = 6
C = np.asarray([[6.0, 0.0, 0.0, 0.0], [5.0, 3.0, -3.0, 1.0], [1.0, 3.0, 3.0, -2.0], [0.0, 0.0, 0.0, 1.0]]) / 6.0

log_list = []
def R6toSE3(p):
    ans = np.identity(4)
    # print(p[:3])
    ans[:3, 3] = p[:3]
    a = p[3]
    b = p[4]
    c = p[5]
    ans[:3, :3] = np.asarray([[cos(a) * cos(b), cos(a) * sin(b) * sin(c) - sin(a) * cos(c), cos(a) * sin(b) * cos(c) + sin(a) * sin(c)],
                                [sin(a) * cos(b), sin(a) * sin(b) * sin(c) + cos(a) * cos(c), sin(a) * sin(b) * cos(c) - cos(a) * sin(c)],
                                [-sin(b), cos(b) * sin(c), cos(b) * cos(c)]])
    return ans

def SE3toR6(T):
    ans = np.zeros(6)
    ans[:3] = T[:3, 3]
    ans[4] = atan2(-T[2, 0], sqrt(T[0, 0] ** 2 + T[1, 0] ** 2))
    if ans[4] == -90.0:
        ans[3] = atan2(-T[1, 2], -T[0, 2])
    elif ans[4] == 90.0:
        ans[3] = atan2(T[1, 2], T[0, 2])
    else:
        ans[3] = atan2(T[1, 0], T[0, 0])
        ans[5] = atan2(T[2, 1], T[2, 2])
    return ans

def u(t): # evaluates parametric value from timestamp
    if t <= start:
        return 2
    if t >= end:
        return n + 3
    return (int)((t - start) * 1.0 / dt) + 3

def B(u):
    return np.matmul(C, np.asarray([1, u, u ** 2, u ** 3]))

def cross(w):
    x = w[0]
    y = w[1]    
    z = w[2]
    return np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]])

def invcross(R):
    return np.asarray([R[2, 1], R[0, 2], R[1, 0]])

def rodrigues(w):
    theta = la.norm(w)
    if theta == 0.0:
        return np.identity(3)
    return np.identity(3) + sin(theta) * 1.0 * cross(w) / theta + (1.0 - cos(theta)) * (cross(w) ** 2) / (theta ** 2)    

def V(w):
    theta = la.norm(w)
    wx = cross(w)
    if theta == 0.0:
        return np.identity(3)
    return np.identity(3) + (1.0 - cos(theta)) * wx / (theta ** 2) + (theta - sin(theta)) * (wx ** 2) / (theta ** 3) 

def matln(R):
    # print(R)
    # print((np.trace(R) - 1.0) / 2.0)
    if abs((np.trace(R) - 1.0) / 2.0) >= 1.0:
        return np.zeros((4, 4))
    theta = acos((np.trace(R) - 1.0) / 2.0)
    # if theta == 0.0:
        # return np.zeros((4, 4))
    return theta * (np.subtract(R, np.transpose(R))) / (2.0 * sin(theta))

# def matlog(R):
    # d = (np.trace(R) - 1.0) / 2.0
    # if -1 < d and d < 1:
        # return arccos(d) * 1.0 * (R - np.transpose(R)) / (2.0 * sqrt(1.0 - d ** 2))
    # else:
        # return (R - np.transpose(R)) / 2.0

def matlog(T):
    t = T[:3, 3]
    R = T[:3, :3]
    w = invcross(matln(R))
    tnew = np.matmul(la.inv(V(w)), t)
    return np.concatenate((tnew, w))

def matexp(v):
    t = v[:3]
    w = v[3:]
    ans = np.identity(4)
    ans[:3, :3] = rodrigues(w)
    tmp = V(w)
    ans[:3, 3] = np.matmul(tmp, t)
    return ans

def omega(pts, ind):
    if ind - 1 >= pts.size/6:
        return matlog(np.identity(4))
    elif ind >= pts.size/6:
        return matlog(la.inv(R6toSE3(pts[ind - 1])))
    return matlog(np.matmul(la.inv(R6toSE3(pts[ind - 1])), R6toSE3(pts[ind])))

def pose_eval(t, T, ctr_pts):
    k = u(t)
    u_ = t - T[k]
    ans = R6toSE3(ctr_pts[k - 1])
    for _j in range(3):
        j = _j + 1
        ans = np.matmul(ans, matexp(B(u_)[j] * omega(ctr_pts, k + j)))
    return SE3toR6(ans)

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

def setup(time_src): # setting the global values
    global dt, start, end, n
    # time_src = np.load(src)
    start = time_src[0]
    n = time_src.size - 1
    end = time_src[n]
    dt = (end - start) * 1.0 / n 
    # T = np.concatenate((np.arange(start, end, dt), np.asarray([end])))
    # N = np.zeros((n + 1, n + 1))
    # for i in range(n + 1):
        # N[i] = n_coeff(T[i], T)[:n+1]
    return time_src

def testing(src, gt, log, T, P, av, ind, out):
    # src = np.load(fsrc)
    # gt = np.load(fgt)
    loss = 0.0
    pred = gt
    # print(src[0])
    # print(start)
    for i in range(src.size):
        # print(i)
        estimate = pose_eval(src[i], T, P)
        # estimate = deBoor(src[i], T, P)
        loss += la.norm(np.subtract(estimate, gt[i]))
        pred[i] = estimate
    print('Finished testing with average error %.6f.' % (loss/src.size))
    output = 'pose_data/' + out + '/' + log + '/' + log + '_%.6d_spline_fusion_predictions.npy' % ind
    # output = log + '_spline_SE3_predictions.npy'
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

loc_list = np.asarray(['1_0.1_50_100'])
spec_lst = np.asarray([[1, 0.1, 50, 100]])

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
            exit()"""

    if log == 'None':
        for lg in log_list:
            for j in range(loc_list.size):
                tr_src = 'pose_data/' + loc_list[j] + '/' + lg + '/' + lg + '_training_time.npy'   
                tr_tgt = 'pose_data/' + loc_list[j] + '/' + lg + '/' + lg + '_training_pose.npy'
                tst_src = 'pose_data/' + loc_list[j] + '/' + lg + '/' + lg + '_testing_time.npy'
                tst_tgt = 'pose_data/' + loc_list[j] + '/' + lg + '/' + lg + '_testing_pose.npy'
                """else:
                tr_src = log + '_training_time.npy'
                tr_tgt = log + '_training_pose.npy'
                tst_src = log + '_testing_time.npy'
                tst_tgt = log + '_testing_pose.npy'"""
                trs_ = np.load(tr_src)
                trt_ = np.load(tr_tgt)
                tss_ = np.load(tst_src)
                tst_ = np.load(tst_tgt)
                tmp1 = (int)(spec_lst[j][0] * spec_lst[j][2])
                tmp2 = (int)(spec_lst[j][0] * spec_lst[j][3])
                tmp3 = (int)(spec_lst[j][1] * spec_lst[j][3])
                duration = trs_.size
                for i in range(min((int)(duration/tmp1), (int)(tss_.size/tmp2))):
                    tr_ind = min((i+1)*tmp1, duration-1)
                    # tst_st = max(0, i*tmp2-tmp3)
                    # tst_lst = min(tss_.size-1, (i+1)*tmp2+tmp3)
                    tst_st = i * tmp2
                    tst_lst = min(tss_.size - 1, (i+1) * tmp2)
                    trs = trs_[i*tmp1:tr_ind]
                    trt = trt_[i*tmp1:tr_ind]
                    tss = tss_[tst_st:tst_lst]
                    tst = tst_[tst_st:tst_lst]

                    av = trt[0]
                    trs, trt, tss, tst = relative(trs, trt, tss, tst)

                    T_ = setup(trs)
                    D = trt
                    T = np.concatenate((T_[0] * np.ones(3), T_, T_[n] * np.ones(3)))
                    # P = la.lstsq(N, D, rcond=-1)[0]
                    P = np.vstack((D[0], D[0], D[0], D, D[n], D[n], D[n]))
                    # P = D
                    """M = np.identity(n) * 4
                    for k in range(n - 1):
                        M[k, k+1] = 1
                        M[k+1, k] = 1
                    C = D[1:n+1] * 6.0
                    C[0] -= D[0]
                    C[n-2] -= D[n]     
                    P_ = np.matmul(la.inv(M), C)
                    P = np.vstack((D[0], P_, D[n]))"""
                    testing(tss, tst, lg, T, P, av, i, loc_list[j])
    else:
        tr_src = log + '_training_time.npy'
        tr_tgt = log + '_training_pose.npy'
        tst_src = log + '_testing_time.npy'
        tst_tgt = log + '_testing_pose.npy'
    
        T, N = setup(tr_src) # timestamps of all the control points and the coefficient matrix
        D = np.load(tr_tgt)
        P = la.lstsq(N, D, rcond=-1)[0]
        # P = D
        testing(tst_src, tst_tgt, log, T, P)

if __name__ == '__main__':
    main()
