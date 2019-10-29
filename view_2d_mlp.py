import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_data(ftrain, fgt, fprediction):
    plt.figure(figsize=(12, 12)) 
    step = 1
    """gt = np.load(fgt)
    points_x = gt[..., 0]
    points_y = gt[..., 1]
    dirs_x = gt[..., 3]
    dirs_y = gt[..., 4]
    plt.quiver(points_x[::step], points_y[::step], dirs_x[::step], dirs_y[::step], scale=50, minshaft=0.5, headwidth=1.5, pivot='mid', color='g')
    plt.show() 
    pred = np.load(fprediction)
    points_x = pred[..., 0]
    points_y = pred[..., 1]
    dirs_x = pred[..., 3]
    dirs_y = pred[..., 4]
    plt.quiver(points_x[::step], points_y[::step], dirs_x[::step], dirs_y[::step], scale=50, minshaft=0.5, headwidth=1.5, pivot='mid', color='b')
    plt.show()"""
    tr = np.load(ftrain)
    points_x = tr[..., 0][:1000]
    points_y = tr[..., 1][:1000]
    # dirs_x = tr[..., 3]
    # dirs_y = tr[..., 4]
    # plt.quiver(points_x[::step], points_y[::step], dirs_x[::step], dirs_y[::step], scale=50, minshaft=0.5, headwidth=1.5, pivot='mid', color='r')
    # plt.scatter(np.arange(0, points_x.size * 0.01, 0.01)[:points_x.size], points_x)
    plt.scatter(points_y, points_x)

    plt.title("Estimated and Actual Continuous Pose (smooth, arbitrary origin, not aligned to any map)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_id', type=str, default='None', help='the GUID for the log that was preprocessed using the read_poses.py script')
    parser.add_argument('--train_tgt', type=str, default='None', help='the file that contains the training output')
    parser.add_argument('--test_tgt', type=str, default='None', help='the file that contains the testing output')
    parser.add_argument('--predictions', type=str, default='None', help='the file that contains the predictions')
    args = parser.parse_args(argv)
    return args

def main(argv=None):
    args = parse_args(argv)
    if args.log_id == 'None':
        if args.train_tgt == 'None':
            print('Missing training output data')
            exit()
        if args.test_tgt == 'None':
            print('Missing testing output data')
            exit()
        if args.predictions == 'None':
            print('Missing MLP prediction data')
            exit()
    else:
        args.train_tgt = args.log_id + '_training_pose.npy'
        args.test_tgt = args.log_id + '_testing_pose.npy'
        args.predictions = args.log_id + '_predictions.npy'
    plot_data(args.train_tgt, args.test_tgt, args.predictions)

if __name__ == '__main__':
    main()
