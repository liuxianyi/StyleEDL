# -*- encoding: utf-8 -*-
import os
import yaml
from collections import OrderedDict


def read_yaml(cfg_path, isResume=False):
    if not isResume:
        path = os.path.join('configs', cfg_path)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                cfg = ordered_yaml_load(f.read())
        else:
            cfg = {}
    else:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = ordered_yaml_load(f.read())

    return cfg


def write_yaml(path, cfg):
    del_list = []

    for k, v in cfg.items():
        if 'scheduler_' in k and k != cfg['scheduler']:
            del_list.append(k)

    for k in del_list:
        del cfg[k]

    with open(os.path.join('logs', path, 'config.yaml'), 'w',
              encoding='utf-8') as f:
        ordered_yaml_dump(cfg,
                          f,
                          default_flow_style=False,
                          allow_unicode=True,
                          indent=4)


def write_result(path, epoch, acc, ldl, mode):
    if mode == 'train':
        name = 'train.txt'
    elif mode == 'test':
        name = 'test.txt'
    with open(os.path.join('logs', path, name), 'a+') as fid:
        if epoch == 0:
            fid.write(
                '{:^10},{:^10},{:^20},{:^20},{:^20},{:^20},{:^20},{:^20},{:^20},{:^20}\n'
                .format('epoch', 'acc', 'klDiv', 'cosine', 'intersection',
                        'chebyshev', 'squareChord', 'sorensendist', 'canberra',
                        'clark'))
        fid.write(
            '{:5}Epoch,{:10},{:20},{:20},{:20},{:20},{:20},{:20},{:20},{:20}\n'
            .format(epoch, acc, ldl['klDiv'], ldl['cosine'],
                    ldl['intersection'], ldl['chebyshev'], ldl['squareChord'],
                    ldl['sorensendist'], ldl['canberra'], ldl['clark']))


def ordered_yaml_load(stream,
                      Loader=yaml.SafeLoader,
                      object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def _construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)
    return yaml.load(stream, OrderedLoader)


def ordered_yaml_dump(data,
                      stream=None,
                      Dumper=yaml.SafeDumper,
                      object_pairs_hook=OrderedDict,
                      **kwds):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())

    OrderedDumper.add_representer(object_pairs_hook, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)
