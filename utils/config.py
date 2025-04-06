import json
from argparse import ArgumentParser
def args2json(args,path):
    with open(path, mode = "w") as f:
        json.dump(args.__dict__, f, indent = 4)

def json2args(path):
    parser = ArgumentParser(description = 'VALID PyTorch')
    args = parser.parse_args()
    args_dict = vars(args)
    with open(path, 'rt') as f:
        args_dict.update(json.load(f))
    return args

def load_config(args,path):
    pass