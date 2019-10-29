import argparse
import numpy as np
import open3d as o3d

def display(ftr, fgt, fpred):
    tr = np.load(ftr)[..., :3]
    gt = np.load(fgt)[..., :3]
    pred = np.load(fpred)

    pred_pcd = o3d.PointCloud()
    pred_pcd.points = o3d.Vector3dVector(pred)
    pred_pcd = o3d.voxel_down_sample(pred_pcd, voxel_size=0.05)
    pred_pcd.paint_uniform_color([0, 0, 1])

    gt_pcd = o3d.PointCloud()
    gt_pcd.points = o3d.Vector3dVector(gt)
    gt_pcd = o3d.voxel_down_sample(gt_pcd, voxel_size=0.05)
    gt_pcd.paint_uniform_color([0, 1, 0])

    tr_pcd = o3d.PointCloud()
    tr_pcd.points = o3d.Vector3dVector(tr)
    tr_pcd = o3d.voxel_down_sample(tr_pcd, voxel_size=0.05)
    tr_pcd.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([pred_pcd])
    # o3d.visualization.draw_geometries([pred_pcd, gt_pcd, tr_pcd])

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_id', type=str, default='None', help='the GUID for the log that was preprocessed using the read_poses.py script')
    parser.add_argument('--train_tgt', type=str, default='None', help='the file that contains the training output')
    parser.add_argument('--test_tgt', type=str, default='None', help='the file that contains the testing output')
    parser.add_argument('--predictions', type=str, default='None', help='the file that contains the predictions')
    parser.add_argument('--type', type=str, default='MLP', help='the type of model used to generate the predictions')
    args = parser.parse_args(argv)
    return args

def main(argv=None):
    args = parse_args(argv)
    tr_tgt = args.train_tgt
    tst_tgt = args.test_tgt
    pred = args.predictions
    log = args.log_id
    if log == 'None':
        if tr_tgt == 'None':
            print('Missing training output data')
            exit()
        if tst_tgt == 'None':
            print('Missing testing output data')
            exit()
        if pred == 'None':
            print('Missing prediction data')
            exit()
    else:
        tr_tgt = log + '_training_pose.npy'
        tst_tgt = log + '_testing_pose.npy'
        pred = log
        if args.type == 'b-spline':
            pred += '_cubic_b-spline'
        elif args.type == 'euler_spiral':
            pred += '_euler_spiral'
        pred += '_predictions.npy'
    display(tr_tgt, tst_tgt, pred)    

if __name__ == '__main__':
    main()
