import argparse
import json

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='utk_face')
    parser.add_argument('--task', type=str, default='race')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--bias_rate', type=float, default=0.9)

    parser.add_argument('--bs', type=int, default=128, help='batch_size')
    parser.add_argument('--cbs', type=int, default=64, help='batch_size of dataloader for contrastive loss')
    parser.add_argument('--lr', type=float, default=1e-3)

    # hyperparameters
    parser.add_argument('--weight', type=float, default=0.01)
    parser.add_argument('--ratio', type=int, default=10)
    parser.add_argument('--aug', type=int, default=1)
    parser.add_argument('--bb', type=int, default=0)

    opt = parser.parse_args()

    return opt

def main():
    opt = parse_option()
    methods = ['ce', 'adv', 'dro', 'di', 'bc', 'os', 'uw', 'us', 'bm', 'bc+os', 'bc+us', 'bc+uw']
    bias_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.997, 0.999]
    result = []
    result.append("ce, adv, dro, di, bc, os, uw, us, bm, bc+os, bc+us, bc+uw\n")
    for bias_rate in bias_rates:
        row = ""
        for method in methods:
            if method == 'bc':
                exp_name = f'bc-bb{opt.bb}-{opt.dataset}_{opt.task}-{opt.exp_name}-{bias_rate}-lr{opt.lr}-bs{opt.bs}-cbs{opt.cbs}-w{opt.weight}-ratio{opt.ratio}-aug{opt.aug}-seed{opt.seed}'
            elif method.startswith('bc'):
                temp_method = method.split("+")[-1]
                if temp_method == 'us' and opt.task == 'race' and bias_rate >= 0.997:
                    exp_name = f'bc-bb{opt.bb}-{opt.dataset}_{opt.task}-us-{bias_rate}-lr{opt.lr}-bs2-cbs1-w{opt.weight}-ratio{opt.ratio}-aug{opt.aug}-seed{opt.seed}'
                elif temp_method == 'us' and opt.task == 'age' and bias_rate >= 0.99:
                    exp_name = f'bc-bb{opt.bb}-{opt.dataset}_{opt.task}-us-{bias_rate}-lr{opt.lr}-bs2-cbs1-w{opt.weight}-ratio{opt.ratio}-aug{opt.aug}-seed{opt.seed}'
                else:
                    exp_name = f'bc-bb{opt.bb}-{opt.dataset}_{opt.task}-{temp_method}-{bias_rate}-lr{opt.lr}-bs{opt.bs}-cbs{opt.cbs}-w{opt.weight}-ratio{opt.ratio}-aug{opt.aug}-seed{opt.seed}'
            else:
                exp_name = f'{method}-{opt.dataset}_{opt.task}-{opt.exp_name}-{bias_rate}-lr{opt.lr}-bs{opt.bs}-seed{opt.seed}'
            filename = f'exp_results/{exp_name}/train.log'
            with open(filename, 'r') as f:
                stats_dict = eval(f.readlines()[-2].strip()[12:])
                acc = stats_dict['best_test_test/acc']
                acc_unbiased = stats_dict['best_test_test/acc_unbiased']
                if 'best_test_test/diff' in stats_dict:
                    diff = stats_dict['best_test_test/diff']
                else:
                    diff = stats_dict['best_test_test/acc_diff']
                acc_skew = stats_dict['best_test_test/acc_skew']
                acc_align = stats_dict['best_test_test/acc_align']
                row += str(acc_align) + ","
        row += "\n"
        result.append(row)
    with open('result.csv', 'w+') as f:
        f.writelines(result)

if __name__ == '__main__':
    main()