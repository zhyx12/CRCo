#
# ----------------------------------------------
import torch
import torch.nn.functional as F
from basicda.runner import BaseTrainer, BaseValidator, TRAINER, VALIDATOR
from basicda.hooks import MetricsLogger
from ..hooks import ClsAccuracy, ClsBestAccuracyByVal
from basicda.utils import get_root_logger
from mmcv.runner import get_dist_info
from basicda.utils import concat_all_gather


@VALIDATOR.register_module(name='source_only')
class ValidatorSourceOnly(BaseValidator):
    def __init__(self, basic_parameters):
        super(ValidatorSourceOnly, self).__init__(**basic_parameters)

    def eval_iter(self, val_batch_data):
        # val_img, val_label, val_name = val_batch_data
        val_img = val_batch_data['img']
        val_label = val_batch_data['gt_label'].squeeze(1)
        val_metas = val_batch_data['img_metas']
        with torch.no_grad():
            feat, pred_unlabeled, _, target_logits = self.model_dict['base_model'](val_img)
        return {'img': val_img,
                'gt': val_label,
                'img_metas': val_metas,
                'feat': feat,
                'pred': pred_unlabeled,
                'target_pred': target_logits,
                }


@TRAINER.register_module('source_only')
class TrainerSourceOnly(BaseTrainer):
    def __init__(self, basic_parameters,
                 #
                 toalign=False, src_ce_type='weak', lambda_label_smooth=0.1,
                 ):
        super(TrainerSourceOnly, self).__init__(**basic_parameters)
        self.toalign = toalign
        self.num_class = self.train_loaders[0].dataset.n_classes
        assert src_ce_type in ['weak', 'strong'], 'wrong src_ce_type'
        self.src_ce_type = src_ce_type
        self.lambda_label_smooth = lambda_label_smooth
        #
        rank, world_size = get_dist_info()
        self.world_size = world_size
        # 增加记录
        if self.local_rank == 0:
            log_names = ['cls', ]
            loss_metrics = MetricsLogger(log_names=log_names, group_name='loss', log_interval=self.log_interval)
            self.register_hook(loss_metrics)
        #
        self.main_hist = None

    def train_iter(self, *args):
        src_img = args[0]['img']
        src_label = args[0]['gt_label'].squeeze(1)
        src_domain_label = args[0].get('domain_label', None)
        #
        src_labeled_size = src_img.shape[0]
        #
        batch_metrics = {}
        batch_metrics['loss'] = {}
        #
        base_model = self.model_dict['base_model']
        #
        self.zero_grad_all()
        #
        tmp_res = base_model(src_img,
                             src_labeled_size=src_labeled_size, toalign=self.toalign, src_labels=src_label,
                             train_iter=self.iteration)
        #
        if self.toalign:
            outputs = tmp_res['src_toalign_logits']
        else:
            outputs = tmp_res['logits']
        # softmax_out = F.softmax(outputs, dim=1)
        #
        loss = F.cross_entropy(outputs, src_label, label_smoothing=self.lambda_label_smooth)
        #
        loss.backward()
        self.step_grad_all()
        #
        batch_metrics['loss']['cls'] = loss.item()
        return batch_metrics

    def load_pretrained_model(self, weights_path):
        logger = get_root_logger()
        weights = torch.load(weights_path, map_location='cpu')
        weights = weights['base_model']
        self.model_dict['base_model'].load_state_dict(weights)
        logger.info('load pretrained model {}'.format(weights_path))
