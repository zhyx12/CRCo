#
# ----------------------------------------------
from mmcv.runner.hooks import Hook
from basicda.utils import get_root_logger, get_root_writer
from .cls_accuracy import ClsAccuracy
from basicda.hooks import HOOKS


@HOOKS.register_module()
class ClsBestAccuracyByVal(Hook):
    def __init__(self, runner, patience=100, pred_key='pred'):
        for ind, (key, _) in enumerate(runner.test_loaders.items()):
            if ind == 0:
                self.test_dataset_name = key
            elif ind == 1:
                self.val_dataset_name = key
        # assert self.val_dataset_name is not None, "you should specify val dataset"
        if not hasattr(self, 'val_dataset_name'):
            self.val_dataset_name = self.test_dataset_name
        self.best_val_acc = 0
        self.best_test_acc = 0
        self.best_iteration = 0
        self.counter = 0
        self.patience = patience
        self.pred_key = pred_key

    def after_val_iter(self, runner):
        pass

    def after_val_epoch(self, runner):
        logger = get_root_logger()
        writer = get_root_writer()
        #
        test_acc = None
        val_acc = None
        for hook in runner._hooks:
            if isinstance(hook, ClsAccuracy):
                if hook.pred_key == self.pred_key:
                    if hook.dataset_name == self.test_dataset_name:
                        test_acc = hook.current_acc
                    if hook.dataset_name == self.val_dataset_name:  # not elif,
                        val_acc = hook.current_acc
        assert val_acc is not None, "you should specify ClassAccuracy hook for val dataset"
        #
        if val_acc >= self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_test_acc = test_acc
            self.best_iteration = runner.iteration
            self.counter = 0
            runner.save_flag = True
        else:
            self.counter += 1
            if self.counter > self.patience:
                runner.early_stop_flag = True
        #
        logger.info(
            "Iteration {}, best test acc = {}, occured in {} iterations, with val acc {}".format(runner.iteration,
                                                                                                 self.best_test_acc,
                                                                                                 self.best_iteration,
                                                                                                 self.best_val_acc))
        writer.add_scalar('best_acc', self.best_test_acc, global_step=runner.iteration)
