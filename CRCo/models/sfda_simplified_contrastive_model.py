import random

import torch
from basicda.models import MODELS
from mmcv.runner import get_dist_info
from basicda.utils import concat_all_gather
from .basic_mix_contrastive_model import BasicMixContrastiveModel
import torch.nn.functional as F


@MODELS.register_module(name='sfda_simplified_contrastive_model')
class SFDASimplifiedContrastiveModel(BasicMixContrastiveModel):
    def __init__(self, model_dict, classifier_dict, num_class, low_dim=128,
                 model_moving_average_decay=0.99,
                 proto_moving_average_decay=0.99,
                 fusion_type='reconstruct_double_detach',
                 force_no_shuffle_bn=False, forward_twice=False,
                 ):
        super(SFDASimplifiedContrastiveModel, self).__init__(model_dict, classifier_dict, num_class, low_dim=low_dim,
                                                   model_moving_average_decay=model_moving_average_decay,
                                                   proto_moving_average_decay=proto_moving_average_decay,
                                                   fusion_type=fusion_type, normalize=True,
                                                   force_no_shuffle_bn=force_no_shuffle_bn,
                                                   )
        rank, world_size = get_dist_info()
        self.local_rank = rank
        self.world_size = world_size
        # 强弱图像分别前传
        self.forward_twice = forward_twice

    def test_forward(self, x1_img, **kwargs):
        # print('through test forward')
        feat = self.online_network(x1_img, **kwargs)
        online_pred = self.online_classifier(feat)
        target_feat = self.target_network(x1_img, **kwargs)
        target_pred = self.target_classifier(target_feat)
        return feat, online_pred, target_feat, target_pred

    def model_forward(self, x1_img, x2_img, model_type=None, shuffle_bn=False, **kwargs):
        if model_type == "online":
            feature_extractor = self.online_network
            classifier = self.online_classifier
        elif model_type == 'target':
            feature_extractor = self.target_network
            classifier = self.target_classifier
        else:
            raise RuntimeError('wrong model type specified')
        #
        x1_shape = x1_img.shape[0]
        img_concat = torch.cat((x1_img, x2_img))
        if shuffle_bn:
            img_concat, idx_unshuffle = self._batch_shuffle_ddp(img_concat)
        feat = feature_extractor(img_concat)
        if shuffle_bn:
            feat = self._batch_unshuffle_ddp(feat, idx_unshuffle)
        logits = classifier(feat)
        prob = F.softmax(logits, dim=-1)
        #
        strong_feat = feat[0:x1_shape]
        weak_feat = feat[x1_shape:]
        strong_logits = logits[0:x1_shape]
        weak_logits = logits[x1_shape:]
        strong_prob = prob[0:x1_shape]
        weak_prob = prob[x1_shape:]
        #
        output = {
            'strong_feat': strong_feat,
            'weak_feat': weak_feat,
            'strong_logits': strong_logits,
            'weak_logits': weak_logits,
            'strong_prob': strong_prob,
            'weak_prob': weak_prob,
        }
        return output

    def single_input_forward(self, x1_img, **kwargs):
        if self.training:
            feature_extractor = self.online_network
            classifier = self.online_classifier
            feat = feature_extractor(x1_img)
            logits = classifier(feat)
            output = {
                'feat': feat,
                'logits': logits,
            }
            return output
        else:
            return self.test_forward(x1_img,**kwargs)
