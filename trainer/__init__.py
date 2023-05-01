from .gmgtrainer import gmgTrainer
trainers = {"gmg": gmgTrainer}


def get_trainer(trainer: str):
    if trainer in trainers:
        return trainers[trainer]
    else:
        raise Exception('No such model: "%s", available: {%s}.' %
                        (trainer, '|'.join(trainers.keys())))
