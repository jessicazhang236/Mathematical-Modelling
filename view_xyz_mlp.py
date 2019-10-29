import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse

# missed = np.array([2, 21, 23, 25, 27, 28, 29, 33, 39, 44, 46, 48, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 70, 72, 82, 83, 86, 88, 89, 92, 93, 95, 96, 98, 101, 104, 106, 112, 115, 116, 122, 130, 132, 134, 139, 140, 142, 143, 148, 153, 158, 162, 163, 165, 166, 170, 185, 186, 187, 194, 196, 201, 204, 206, 214, 215, 221, 228, 231, 233, 235, 236, 239, 241, 242, 243, 247, 252, 254, 255, 256, 267, 268, 271, 273, 275, 277, 280, 282, 283, 284, 288, 289, 290, 294, 296, 297, 301, 308, 314, 317, 318, 322, 324, 325, 334, 335, 338, 339, 353, 355, 357, 358, 359, 365, 367, 370, 376, 409, 411, 444, 448, 458, 492])
# missed = np.array([63, 196, 322, 450, 581, 536, 51])
log_lst = np.array(['2b998b1d-50b7-9d81-a898-7a44f0230572'])

# def plot_data(ftrain, fgt, fprediction):
def plot_data(fgt, log_id):
    gt = np.load('img_data/' + log_id + '/' + fgt)[..., :3]
    gt_pcd = o3d.PointCloud()
    gt_pcd.points = o3d.Vector3dVector(gt)
    gt_pcd = o3d.voxel_down_sample(gt_pcd, voxel_size=0.05)
    gt_pcd.paint_uniform_color([1, 0, 0])
    # colors = np.zeros((gt.size/3, 3))
    for i in range(gt.size/300):
        # if i in missed:
        if True:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            tmp = o3d.PointCloud()
            tmp.points = o3d.Vector3dVector(gt[i*100:(i+1)*100])
            vis.add_geometry(tmp)
            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()
            # image = vis.capture_screen_float_buffer()
            # o3d.write_image('%.5d.png' % (i), image)
            vis.capture_screen_image('%.7d.png' % (i), do_render=False)
            vis.destroy_window()
            # np.asarray(gt_pcd.colors)[i*100:(i+1)*100] = i*300.0/gt.size*np.ones(3)
            np.asarray(gt_pcd.colors)[i*100:(i+1)*100] = np.asarray([0, 0, 1])
    # gt_pcd.colors = o3d.Vector3dVector(colors)
    print(np.asarray(gt_pcd.colors))
    # o3d.visualization.draw_geometries([gt_pcd])

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
    # if args.log_id == 'None':
        # if args.train_tgt == 'None':
            # print('Missing training output data')
            # exit()
        # if args.test_tgt == 'None':
            # print('Missing testing output data')
            # exit()
        # if args.predictions == 'None':
            # print('Missing MLP prediction data')
            # exit()
    # else:
    for log in log_lst:
        plot_data(log + '_training_pose.npy', log)
        # args.train_tgt = args.log_id + '_training_pose.npy'
        # args.test_tgt = args.log_id + '_testing_pose.npy'
        # args.predictions = args.log_id + '_predictions.npy'
    # plot_data(args.train_tgt, args.test_tgt, args.predictions)
    # plot_data(args.train_tgt, args.log_id)

if __name__ == '__main__':
    main()
