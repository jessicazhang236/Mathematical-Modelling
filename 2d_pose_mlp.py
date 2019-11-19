import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
import matplotlib.pyplot as plt

input_dim = 1
hidden_dim1 = 480
hidden_dim2 = 480
hidden_dim3 = 480
output_dim = 6
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
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
'432aa902-c28d-9e73-a6f5-31934a2f95d3',
'43677200-e6f0-2051-5e6a-5e6fc3851fde',
'4763a435-5887-4459-ee65-fb6075217a35',
'4853043c-065a-462f-c9f0-67f8263a0eb6',
'5da97c51-65cd-a987-dffb-8b7599aa0c0b',
'632c320b-1b9e-de97-663e-1f78d0082617'])
"""
loc_lst = np.asarray(['1_0.1_50_100'])
spec_lst = np.asarray([[1, 0.1, 50, 100]])

def normalize1(lst, il): # il is the input list
    # return lst - il[0]
    return (lst - np.mean(il)) / np.std(il)

def normalize2(lst, il):
    # return lst - il[0]
    return (lst - np.mean(il, axis=0)) / np.std(il, axis=0)

def normalize(l1, l2, l3, l4):
    # l1 is the training input, l2 training target, l3 testing input, l4 testing target
    # np.save(f1, normalize1(l1, l1))
    # np.save(f2, normalize2(l2, l2))
    # np.save(f3, normalize1(l3, l1))
    # np.save(f4, normalize2(l4, l2))
    return normalize1(l1, l1), normalize2(l2, l2), normalize1(l3, l1), normalize2(l4, l2)
    
    
class PoseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(PoseMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        # self.fc2 = nn.DataParallel(self.fc2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        # self.fc3 = nn.DataParallel(self.fc3)
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
# if torch.cuda.device_count() > 1:
# posemlp = nn.DataParallel(posemlp)
posemlp.to(device)

criterion = nn.MSELoss()
def train(train_in, train_tgt):
    train_size = train_in.size
    tensor_x = torch.Tensor(train_in)
    tensor_y = torch.stack([torch.Tensor(i) for i in train_tgt])
    trainset = utils.TensorDataset(tensor_x, tensor_y)
    trainloader = utils.DataLoader(trainset)

    optimizer = optim.SGD(posemlp.parameters(), lr=0.001, momentum=0.9, nesterov=True)

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
                print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, running_loss / train_size))
                running_loss = 0.0

    print('Finished training')

def test(tst_src, gt, output, mean, std):
    tensor_x = torch.Tensor(tst_src)
    tensor_y = torch.stack([torch.Tensor(i) for i in gt])
    testset = utils.TensorDataset(tensor_x, tensor_y)
    testloader = utils.DataLoader(testset)
    loss = 0.0
    test_size = 0
    saved_data = gt

    with torch.no_grad():
        for data in testloader:
            # print(saved_data)
            inputs, targets = data[0].to(device), data[1].to(device)
            outputs = posemlp(inputs)
            loss += criterion(outputs, targets[0]).item()
            saved_data[test_size] = outputs.cpu().numpy()
            test_size += 1
    print(saved_data)
    print('Finished testing with average loss %.6f.', loss/test_size)
    # saved_data += mean
    saved_data = saved_data * std + mean

    print('Dumping data to ' + output)
    np.save(output, saved_data)

    """points_x = saved_data[...,0]
    points_y = saved_data[...,1]
    dirs_x = saved_data[..., 3]
    dirs_y = saved_data[..., 4]
    plt.figure(figsize=(12, 12))
    step = 1
    plt.quiver(points_x[::step], points_y[::step], dirs_x[::step], dirs_y[::step], scale=50, minshaft=1, headwidth=1.5, pivot='mid', color='b')
    plt.axis('equal')

    gt = gt * std + mean

    points_x = gt[...,0]
    points_y = gt[...,1]
    dirs_x = gt[..., 3]
    dirs_y = gt[..., 4]
    plt.quiver(points_x[::step], points_y[::step], dirs_x[::step], dirs_y[::step], scale=50, minshaft=1, headwidth=1.5, pivot='mid', color='g')
    
    gt = train_tgt
    gt = gt * std + mean
    points_x = gt[...,0]
    points_y = gt[...,1]
    dirs_x = gt[..., 3]
    dirs_y = gt[..., 4]
    plt.quiver(points_x[::step], points_y[::step], dirs_x[::step], dirs_y[::step], scale=50, minshaft=1, headwidth=1.5, pivot='mid', color='r')

    plt.title("Estimated and Actual Continuous Pose (smooth, arbitrary origin, not aligned to any map)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()"""

    # points_x = gt[...,0]
    # points_y = gt[...,1]
    # dirs_x = gt[..., 3]
    # dirs_y = gt[..., 4]
    # plt.figure(figsize=(12, 12))
    # step = 1
    # plt.quiver(points_x[::step], points_y[::step], dirs_x[::step], dirs_y[::step], scale=50, minshaft=0.5, headwidth=1.5, pivot='mid')
    # plt.axis('equal')

    # plt.title("Actual Continuous Pose (smooth, arbitrary origin, not aligned to any map)")
    # plt.xlabel("x (m)")
    # plt.ylabel("y (m)")
    # plt.show()


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
    """args = parse_args(argv)
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
    mean = np.mean(train_tgt, axis=0)
    std = np.std(train_tgt, axis=0)
    tr_src, tr_tgt, tst_src, tst_tgt = normalize(args.train_src, args.train_tgt, args.test_src, args.test_tgt)
    train(tr_src, tr_tgt)
    test(tr_tgt, tst_src, tst_tgt, os.path.join(args.output_dir, args.log_id+'_predictions.npy'), mean, std)
    """
    for lg in log_list:
        for j in range(loc_lst.size):
            tr_src = 'pose_data/' + loc_lst[j] + '/' + lg + '/' + lg + '_training_time.npy'   
            tr_tgt = 'pose_data/' + loc_lst[j] + '/' + lg + '/' + lg + '_training_pose.npy'
            tst_src = 'pose_data/' + loc_lst[j] + '/' + lg + '/' + lg + '_testing_time.npy'
            tst_tgt = 'pose_data/' + loc_lst[j] + '/' + lg + '/' + lg + '_testing_pose.npy'
            trsrc = np.load(tr_src)
            trtgt = np.load(tr_tgt)
            tssrc = np.load(tst_src)
            tstgt = np.load(tst_tgt)
            duration = trsrc.size
            tmp1 = (int)(spec_lst[j][0] * spec_lst[j][2])
            tmp2 = (int)(spec_lst[j][0] * spec_lst[j][3])
            tmp3 = (int)(spec_lst[j][1] * spec_lst[j][3])

            i = 2300

            while i <= 4036:
                # print(i)
                tr_ind = min((i+1)*tmp1, duration-1)
                tst_st = max(0, i*tmp2)
                tst_lst = min(tssrc.size-1, (i+1)*tmp2)
                fpred = '/mnt/yyz_data_2/users/jessicaz/pose_data/' + loc_lst[j] + '/' + lg + '/' + lg + '_%.6d_mlp2_predictions.npy' % i
                # print(fpred)
                pred = np.load(fpred)
                # print(pred)
                trs = trsrc[i*tmp1:tr_ind]
                trt = trtgt[i*tmp1:tr_ind]
                tss = tssrc[tst_st:tst_lst]
                tst = tstgt[tst_st:tst_lst]
                if np.array_equal(pred[0], pred[pred.size/6-1]) == False:
                    i += 1
                    continue
                # mean = trt[0]
                mean = np.mean(trt, axis=0)
                std = np.std(trt, axis=0)
                trs, trt, tss, tst = normalize(trs, trt, tss, tst)
                train(trs, trt)
                test(tss, tst, '/mnt/yyz_data_2/users/jessicaz/pose_data/' + loc_lst[j] + '/' + lg + '/' + lg + '_%.6d_mlp2_predictions.npy' % i, mean, std)
                i += 1
if __name__ == "__main__":
    main()
