# -*- encoding: utf-8 -*-
import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def create_summary(tag):
    name = None
    if tag != 'cache':
        name = datetime.now().strftime('%b-%d_%H:%M:%S') + '_' + tag
    else:
        name = 'cache'

    writer_dir = os.path.join("./runs", name)
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)

    writer = SummaryWriter(writer_dir)
    return writer, name


def create_logger(name):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)

    file_path = os.path.join('./logs', name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    handler = logging.FileHandler(os.path.join(file_path, 'train.log'),
                                  encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def clear_log(name):
    for path in ['runs', 'logs']:
        p = os.path.join(path, name)
        if os.path.isdir(p):
            command = 'rm -r ' + p
            os.system(command)


def save_del(name):
    while True:
        ans = input("\nWhether to save the results of this training? (yes/no)")
        if ans == 'no':
            clear_log(name)
            break
        elif ans == 'yes':
            break
