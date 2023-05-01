# -*- encoding: utf-8 -*-
import os
import traceback
from collections import OrderedDict
import torch
from trainer import get_trainer
from utils import parse_args, seed_torch
from utils import read_yaml, write_yaml
from utils import create_summary, create_logger, clear_log, save_del

if __name__ == "__main__":
    # ================1. config====================
    opt = OrderedDict(vars(parse_args()))

    if opt['resume_path']:
        config = read_yaml(os.path.join(opt['resume_path'], "config.yaml"),
                           isResume=True)
    else:
        config = read_yaml('base.yaml')
        if opt['specific_cfg']:
            config_train = read_yaml(opt['specific_cfg'])
        else:
            config_train = read_yaml('train.yaml')
        for k, v in config_train.items():
            config[k] = v

        config_model = read_yaml(os.path.join(config['model'] + '.yaml'))
        for k, v in config_model.items():
            config[k] = v

    for k, v in config.items():
        if k not in opt.keys() or opt[k] == None:
            opt[k] = v

    # ================2. log file ====================
    if opt['tag'] == 'cache':
        clear_log('cache')

    writer, path = create_summary(opt['tag'])
    logger = create_logger(path)
    logger.name = __name__

    opt['path'] = path

    # ================3. device====================
    seed_torch(opt['seed'])
    # device setting
    if torch.cuda.is_available():
        opt['device'] = 'cuda:' + str(opt['gpu_id'])
    else:
        opt['device'] = 'cpu'

    # save config
    write_yaml(opt['path'], opt)

    # ================4. start to train====================
    # 1. creater trainer
    trainer = get_trainer(opt["trainer"])(opt, logger, writer)

    # 2. data loader
    trainer.set_dataloader()

    # 3. model
    trainer.set_model()

    # 4. optimizer
    trainer.set_optimizer()

    # 5. lr
    trainer.set_scheduler()

    # 6. metric
    trainer.meters()

    # 7. loss
    trainer.set_loss()

    # 8. resume
    if opt["resume_path"]:
        trainer.load_checkpoint()

    # 9. training
    try:
        trainer.train()
    except KeyboardInterrupt:
        save_del(opt['path'])
    except Exception as e:
        traceback.print_exc()
        save_del(opt['path'])
