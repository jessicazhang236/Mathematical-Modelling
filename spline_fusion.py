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
    return (int)((t - start) * 1.0 / dt)

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
    return np.identity(3) + sin(theta) * 1.0 * cross(w) / theta + (1.0 - cos(theta)) * (cross(w) ** 2) / (theta ** 2)    

def V(w):
    theta = la.norm(w)
    wx = cross(w)
    return np.identity(3) + (1.0 - cos(theta)) * wx / (theta ** 2) + (theta - sin(theta)) * (wx ** 2) / (theta ** 3) 

def matln(R):
    theta = acos((np.trace(R) - 1.0) / 2.0)
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

def testing(fsrc, fgt, log, T, P):
    src = np.load(fsrc)[(p+2)*4:]
    gt = np.load(fgt)[(p+2)*4:]
    loss = 0.0
    pred = gt
    for i in range(src.size):
        estimate = pose_eval(src[i], T, P)
        # estimate = deBoor(src[i], T, P)
        loss += la.norm(np.subtract(estimate, gt[i]))
        pred[i] = estimate
    print('Finished testing with average error %.6f.' % (loss/src.size))
    output = log + '_spline_SE3_predictions.npy'
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
    D = np.load(tr_tgt)
    # dim = D[0].size
    # P = np.matmul(np.linalg.inv(N), D) # control point array
    P = la.lstsq(N, D, rcond=-1)[0]
    # P = D
    testing(tst_src, tst_tgt, log, T, P)

if __name__ == '__main__':
    main()
