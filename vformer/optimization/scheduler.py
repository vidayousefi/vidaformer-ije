import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, ConstantLR, CosineAnnealingLR, LambdaLR


class CustomScheduler:
    def __init__(self, optimizer, stages, total_epochs, iters_per_epoch, start_epoch=1):
        self.iter = (start_epoch - 1) * iters_per_epoch
        stage_idxs = []
        self.lr_history = []

        self._calc_durations(stages, stage_idxs, total_epochs, iters_per_epoch)

        self.simulate(stages, stage_idxs, total_epochs, iters_per_epoch)

        self.scheduler = LambdaLR(optimizer, lambda i: self.lr_history[self.iter])

    # ============================================== Duration =======================================

    @staticmethod
    def _calc_durations(stages, stage_idxs, total_epochs, iters_per_epoch):
        for idx, stage in enumerate(stages):
            next_stage = stages[idx + 1] if idx < len(stages) - 1 else SchedulerStage('constant', 0,
                total_epochs + 1)
            duration = (next_stage.start_epoch - stage.start_epoch)
            for i in range(duration):
                stage_idxs.append(idx)
            if not stage.epoch_based:
                duration *= iters_per_epoch
            stage.set_duration(duration)

    # ============================================== Simulation =======================================

    def simulate(self, stages, stage_idxs, total_epochs, iters_per_epoch):
        dummy_opt = None
        for e in range(total_epochs):
            stage = stages[stage_idxs[e]]
            if not stage.scheduler:
                dummy_opt = SGD([torch.tensor([1., 2.])], 1, 0.9)
                self.create_scheduler(stage, dummy_opt)

            for i in range(iters_per_epoch):
                if not stage.epoch_based:
                    dummy_opt.step()
                    stage.scheduler.step()
                self.lr_history.append(stage.scheduler.get_last_lr()[0])
            if stage.epoch_based:
                dummy_opt.step()
                stage.scheduler.step()

# ============================================== Schedulers =======================================

    @staticmethod
    def create_scheduler(stage, optimizer):
        if stage.sched_type == 'linear':
            sched = LinearLR(optimizer, start_factor=stage.factor[0], end_factor=stage.factor[1],
                total_iters=stage.duration)
        elif stage.sched_type == 'constant':
            sched = ConstantLR(optimizer, stage.factor, total_iters=stage.duration)
        elif stage.sched_type == 'cosine':
            sched = CosineAnnealingLR(optimizer, stage.duration)
        else:
            raise Exception()
        stage.set_scheduler(sched)

# ============================================== API =======================================

    def step(self):
        self.scheduler.step()
        self.iter += 1

    def get_last_lr(self):
        return self.scheduler.get_last_lr()


class SchedulerStage:
    def __init__(self, sched_type, factor, start_epoch, epoch_based=False):
        self.factor = factor
        self.sched_type = sched_type
        self.start_epoch = start_epoch
        self.epoch_based = epoch_based
        self.scheduler = None
        self.duration = 0

    def set_scheduler(self, sched):
        self.scheduler = sched

    def set_duration(self, duration):
        self.duration = duration
