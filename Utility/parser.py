import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='!!')


    parser.add_argument('--node_dropout_flag', type=int, default=1, help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.5]', help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--regs', nargs='?', default='[1e-5]', help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')

    parser.add_argument('--epochs', type=int, default=1, help='epochs.')
    parser.add_argument('--patience', type=int, default=200, help='patience.')
    parser.add_argument('--node_emb_dim', type=int, default=64, help='node_emb_dim.')


    parser.add_argument('--ge_model', type=str, default="-1", help='ge_model, no need to use')
    parser.add_argument('--dataset', type=str, default="flickr", help='dataset')

    parser.add_argument('--meta_emb', type=bool, default=True, help='meta_emb.')
    parser.add_argument('--test_model', type=str, default="LightGCN", help='dataset')
    parser.add_argument('--ge_test', type=bool, default=False, help='dataset')
    parser.add_argument('--test_one', type=bool, default=False, help='dataset')
    parser.add_argument('--cutoff', type=int, default=6, help='dataset')
    parser.add_argument('--device', nargs='?', default='cuda:0', help='choose the device')

    return parser.parse_args()

args = parse_args()
