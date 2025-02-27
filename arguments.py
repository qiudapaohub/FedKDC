import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='Semi decentralized FD')
    # federated arguments
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('--lr', nargs='+', type=float, default=[0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005],
                        help='A list of lr')
    parser.add_argument('--gamma', default=0.7, type=float, help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--epochs', default=150, type=int, help='number of total training rounds')
    parser.add_argument('--loc_epochs', default=5, type=int, help='number of local training rounds')
    parser.add_argument('--KD_epochs', default=10, type=int, help='number of KD training rounds')
    parser.add_argument('--num_clint', default=7, type=int, help='number of clint')
    parser.add_argument('--num_leader', default=3, type=int, help='number of leaders')
    parser.add_argument('--Fed_medium', default='logit', type=str, help='model or logit')
    parser.add_argument('--iid', default=1, type=int, help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--choose_classes', default=4, type=int, help='number of classes in non-IID')
    parser.add_argument('--unequal', default=0, type=int, help='whether to use unequal data splits for  \
                            non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--P_A_E', default=2, type=int, help='poverty alleviation epoch')
    parser.add_argument('--m', default=7, type=int, help='choose m clients to train every epoch')
    parser.add_argument('--alpha_d', default=0.2, type=float, help='data distribution')
    # model arguments
    parser.add_argument('--model1', default='cnn_mnist', type=str, help='cnn_mnist, cnn_cifar10')
    parser.add_argument('--model2', default='vgg11', type=str, help='model name')
    parser.add_argument('--model3', default='mobilenet', type=str, help='model name')
    parser.add_argument('--model4', default='mobilenetv2', type=str, help='model name')
    parser.add_argument('--model5', default='resnet18', type=str, help='model name')
    parser.add_argument('--model6', default='shufflenetv2', type=str, help='model name')
    parser.add_argument('--model7', default='EEGNet', type=str, help='model name')
    parser.add_argument('--model8', default='EEG_CNN_1', type=str, help='model name')
    parser.add_argument('--model9', default='EEG_CNN_2', type=str, help='model name')
    parser.add_argument('--model10', default='EEG_CNN_3', type=str, help='model name')
    parser.add_argument('--model11', default='dnn', type=str, help='model name')
    parser.add_argument('--optimizer', default='adam', type=str, help='[sgd,adam]')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
    # kd arguments
    parser.add_argument('--alpha', default='0.5', type=float, help='alpha in criterion')
    parser.add_argument('--temp', default=1, type=int, help='temperature in KD')
    parser.add_argument('--kd_mode', default='cse', type=str, help='cse or mse')
    # others
    parser.add_argument('--dataset', default='seediv', type=str, help="fmnist, mnist, cifar10, cifar100, seed, seediv, crocul1, crocul2, crocul3")
    parser.add_argument('--num_classes', default=4, type=int, help="cifar100:100")
    parser.add_argument('--input_channels', default=62, type=int, help="mnist:1  cifar:3")
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--save-model', default=False, action='store_true', help='For Saving the current Model')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    args = parser.parse_args()
    return args

