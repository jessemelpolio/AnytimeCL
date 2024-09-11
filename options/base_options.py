import sys
import argparse
import os
import torch
import data
import pickle


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # seed setting
        parser.add_argument("--seed", type=int, default=42)
        # GPU and multi-GPU settings
        parser.add_argument("--device", nargs="+", type=int, default=[0])
        parser.add_argument("--num_workers", type=int, default=1)
        # Dataset settings
        parser.add_argument("--data_root", type=str, default="/data/owcl_data")
        parser.add_argument("--datasets", type=str, default="CIFAR100,SUN397,EuroSAT,OxfordIIITPet,Flowers102,FGVCAircraft,StanfordCars,Food101")
        parser.add_argument("--held_out_dataset", type=str, default="ImageNet,UCF101,DTD")
        # Continual learning settings
        parser.add_argument("--incremental", type=str, default="dataset")
        parser.add_argument("--num_classes", type=int, default=20)
        parser.add_argument("--randomize", type=bool, default=True)
        parser.add_argument("--perc", type=float, default=0.2)
        # Model settings
        parser.add_argument("--network_arc", type=str, default="clip")
        parser.add_argument("--backbone", type=str, default="ViT-B/32")
        parser.add_argument("--input_size", type=int, default=224)
        # Optimization settings
        parser.add_argument("--optimizer", type=str, default="adamw")
        parser.add_argument("--batch_size", type=int, default=2048)
        parser.add_argument("--lr", type=float, default=6e-4)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        # Training settings
        parser.add_argument("--exp_name", type=str, default="test")
        parser.add_argument("--results_dir", type=str, default="./results")
        parser.add_argument("--csv_file", type=str, default="acc.csv")
        parser.add_argument("--n_epochs", type=int, default=20)
        parser.add_argument("--log_interval", type=int, default=10)
        parser.add_argument("--save_interval", type=int, default=5)
        parser.add_argument("--eval_interval", type=int, default=5)
        parser.add_argument(
            "--eval_scenario", type=str, default="cumulative_cumulative"
        )
        # parser.add_argument("--train_scenario", type=str, default="single_cumulative")
        parser.add_argument("--save", action="store_true")
        parser.add_argument("--log_dir", type=str)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--resume_ckpt", type=str, default=None)
        parser.add_argument("--start_epoch", type=int, default=0)
        # Misc
        parser.add_argument("--load_from_opt_file", action="store_true")

        # compression related argument
        parser.add_argument("--need_compress", type=bool, default=False)
        parser.add_argument("--CLS_weight", type=bool, default=False)
        parser.add_argument("--per_instance", type=bool, default=True)
        parser.add_argument("--int_quantize", type=bool, default=False)
        parser.add_argument("--components", type=int, default=5)
        parser.add_argument("--int_range", type=int, default=255)
        
        self.initialized = True
        return parser

    def gather_options(self, module_list):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        # get the basic options
        args, unknown = parser.parse_known_args()

        for module in module_list:
            parser = module.modify_commandline_options(parser)

        args, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if args.load_from_opt_file:
            parser = self.update_options_from_file(parser, args)

        args = parser.parse_args()
        self.parser = parser
        return args

    def print_options(self, opt):
        message = "" + "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

    def option_file_path(self, opt, makedir=True):
        expr_dir = os.path.join(opt.results_dir, opt.exp_name)
        if makedir and not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        return os.path.join(expr_dir, "opt")

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(f"{file_name}.txt", "wt") as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ""
                default = self.parser.get_default(k)
                if v != default:
                    comment = "\t[default: %s]" % str(default)
                opt_file.write("{:>25}: {:<30}{}\n".format(str(k), str(v), comment))

        with open(f"{file_name}.pkl", "wb") as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        return pickle.load(open(f"{file_name}.pkl", "rb"))

    def parse(self, module_list, is_train=True):
        opt = self.gather_options(module_list)
        opt.isTrain = is_train  # train or test

        if not os.path.exists(opt.results_dir):
            os.makedirs(opt.results_dir, exist_ok=True)
        exp_dir = os.path.join(opt.results_dir, opt.exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)

        opt.results_dir = exp_dir

        # self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # set gpu ids
        opt.device = torch.device(
            f"cuda:{opt.device[0]}" if torch.cuda.is_available() else "cpu"
        )

        # str_ids = opt.device.split(',')
        # opt.device = []
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id >= 0:
        #         opt.device.append(id)
        # if len(opt.device) > 0:
        #     torch.cuda.set_device(opt.device[0])

        # assert len(opt.device) == 0 or opt.batchSize % len(opt.device) == 0, \
        #     "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
        #     % (opt.batchSize, len(opt.device))

        self.args = opt
        return self.args
