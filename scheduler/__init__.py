from torch import optim


def get_scheduler(opt, optimizer):
    if opt['scheduler'] == "scheduler_stepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=opt['scheduler_stepLR']['step_size'],
                                              gamma=opt['scheduler_stepLR']['gamma'])

    # MultiStepLR
    elif opt['scheduler'] == "scheduler_multi":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=opt['scheduler_multi']['milestones'],
            gamma=opt['scheduler_multi']['gamma'])

    # ExponentialLR
    elif opt['scheduler'] == "scheduler_exp":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=opt['scheduler_exp']['gamma'])

    # CosineAnnealingLR
    elif opt['scheduler'] == "scheduler_cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=opt['scheduler_cos']['T_max'],
            eta_min=opt['scheduler_cos']['eta_min'])

    # CyclicLR
    elif opt['scheduler'] == "scheduler_cyclic":
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=opt['lr'],
            max_lr=opt['scheduler_cyclic']['max_lr'],
            step_size_up=opt['scheduler_cyclic']['up'],
            step_size_down=opt['scheduler_cyclic']['down'])

    elif opt['scheduler'] == 'scheduler_lambda':
        if opt['scheduler_lambda']['lr_lambda'] == None:
            raise NotImplementedError("lr_lambda need define")
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, opt['scheduler_lambda']['lr_lambda'])

    else:
        scheduler = None

    return scheduler
