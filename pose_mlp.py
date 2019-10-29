import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
import open3d as o3d

input_dim = 1
hidden_dim1 = 180
hidden_dim2 = 300
hidden_dim3 = 120
output_dim = 6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normalize1(lst, il): # il is the input list
    return (lst - np.mean(il)) / np.std(il)

def normalize2(lst, il):
    return (lst - np.mean(il, axis=0)) / np.std(il, axis=0)

def normalize(f1, f2, f3, f4):
    l1 = np.load(f1)
    l2 = np.load(f2)
    l3 = np.load(f3)
    l4 = np.load(f4)
    # l1 is the training input, l2 training target, l3 testing input, l4 testing target
    nl1 = normalize1(l1, l1)
    nl2 = normalize2(l2, l2)
    nl3 = normalize1(l3, l1)
    nl4 = normalize2(l4, l2)
    return nl1, nl2, nl3, nl4
    
    
class PoseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(PoseMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)
    
    def forward(self, x_in, apply_softmax=False):
        intermediate = F.relu(self.fc1(x_in))
        intermediate = F.relu(self.fc2(intermediate))
        intermediate = F.relu(self.fc3(intermediate))
        output = self.fc4(intermediate)
        if apply_softmax:
            output = F.softmax(output, dim=1)
        return output    

posemlp = PoseMLP(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim)
if torch.cuda.device_count() > 1:
    posemlp = nn.DataParallel(posemlp)
posemlp.to(device)

criterion = nn.L1Loss()
def train(train_in, train_tgt):
    train_size = train_in.size
    tensor_x = torch.Tensor(train_in)
    tensor_y = torch.stack([torch.Tensor(i) for i in train_tgt])
    trainset = utils.TensorDataset(tensor_x, tensor_y)
    trainloader = utils.DataLoader(trainset)

    optimizer = optim.SGD(posemlp.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(300):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = posemlp(inputs)
            loss = criterion(outputs, targets[0])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % train_size == train_size - 1:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / train_size))
                running_loss = 0.0

    print('Finished training')

def test(tr_src, train_tgt, tst_src, gt, output, mean, std):
    tensor_x = torch.Tensor(tst_src)
    tensor_y = torch.stack([torch.Tensor(i) for i in gt])
    # print(tensor_y.shape)
    testset = utils.TensorDataset(tensor_x, tensor_y)
    testloader = utils.DataLoader(testset)
    loss = 0.0
    test_size = 0
    saved_data = np.zeros(gt.shape)

    with torch.no_grad():
        for data in testloader:
            # print(saved_data)
            inputs, targets = data[0].to(device), data[1].to(device)
            outputs = posemlp(inputs)
            loss += criterion(outputs, targets[0]).item()
            saved_data[test_size] = outputs.cpu().numpy()
            test_size += 1
    
    saved_data = saved_data * std + mean
    print(saved_data)
    print('Finished testing with average loss %.6f.' % (loss/test_size))
    
    gt = gt * std + mean
    train_tgt = train_tgt * std + mean

    print('Dumping data to ' + output)
    np.save(output, saved_data)

    pred_pcd = o3d.PointCloud()
    pred_pcd.points = o3d.Vector3dVector(saved_data[..., :3])
    pred_pcd = o3d.voxel_down_sample(pred_pcd, voxel_size=0.05)
    pt_cnt = np.asarray(pred_pcd.points).size/3
    pred_pcd.paint_uniform_color([0, 0, 1])
    colors = np.zeros((pt_cnt, 3))
    colors[..., 2] = np.ones(pt_cnt)
    tr_src = np.around(tr_src, decimals=4)
    tst_src = np.around(tst_src, decimals=4)
    for i in range(pt_cnt):
        if tst_src[i] in tr_src:
            np.asarray(pred_pcd.colors)[i, 2] = 0
            np.asarray(pred_pcd.colors)[i, 0] = 1
    print(np.asarray(pred_pcd.colors)) 
    gt_pcd = o3d.PointCloud()
    gt_pcd.points = o3d.Vector3dVector(gt)
    gt_pcd = o3d.voxel_down_sample(gt_pcd, voxel_size=0.05)
    gt_pcd.paint_uniform_color([0, 1, 0])

    tr_pcd = o3d.PointCloud()
    tr_pcd.points = o3d.Vector3dVector(train_tgt)
    tr_pcd = o3d.voxel_down_sample(tr_pcd, voxel_size=0.05)
    tr_pcd.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([pred_pcd, gt_pcd, tr_pcd])

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_id', type=str, default='None', help='the GUID for the log that was preprocessed using the read_poses.py script')
    parser.add_argument('--train_src', type=str, default='None', help='the file that contains the training input')
    parser.add_argument('--train_tgt', type=str, default='None', help='the file that contains the training output')
    parser.add_argument('--test_src', type=str, default='None', help='the file that contains the testing input')
    parser.add_argument('--test_tgt', type=str, default='None', help='the file that contains the testing output')
    parser.add_argument('--output_dir', type=str, default='.', help='the folder to which the estimated output will be dumped')
    args = parser.parse_args(argv)
    return args

def main(argv=None):
    args = parse_args(argv)
    if args.log_id == 'None':
        if args.train_src == 'None':
            print('Missing training input data.')
            exit()
        if args.train_tgt == 'None':
            print('Missing training output data.')
            exit()
        if args.test_src == 'None':
            print('Missing testing input data.')
            exit()
        if args.test_tgt == 'None':
            print('Missing testing output data.')
            exit()
    else:
        args.train_src = args.log_id + '_training_time.npy'
        args.train_tgt = args.log_id + '_training_pose.npy'
        args.test_src = args.log_id + '_testing_time.npy'
        args.test_tgt = args.log_id + '_testing_pose.npy'
    train_tgt = np.load(args.train_tgt)
    # train_src = np.load(args.train_src)
    # test_tgt = np.load(args.test_tgt)[..., :3]
    # test_src = np.load(args.test_src)
    mean = np.mean(train_tgt, axis=0)
    std = np.std(train_tgt, axis=0)
    # tr_src = normalize1(train_src, train_src)
    # tr_tgt = normalize1(train_tgt, train_tgt)
    # tst_src = normalize2(test_src, train_src)
    # tst_tgt = normalize2(test_tgt, train_tgt)
    tr_src, tr_tgt, tst_src, tst_tgt = normalize(args.train_src, args.train_tgt, args.test_src, args.test_tgt)
    train(tr_src, tr_tgt)
    test(tr_src, tr_tgt, tst_src, tst_tgt, os.path.join(args.output_dir, args.log_id+'_predictions.npy'), mean, std)
    
if __name__ == "__main__":
    main()
