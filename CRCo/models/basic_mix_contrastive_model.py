import torch
import torch.nn as nn

import torch.nn.functional as F
from basicda.models import MODELS
from .model_utils import EMA, update_moving_average
from mmcv.runner import get_dist_info
from basicda.utils import concat_all_gather
from basicda.models import build_models


@MODELS.register_module(name='basic_mix_contrastive_model')
class BasicMixContrastiveModel(nn.Module):
    def __init__(self, model_dict, classifier_dict, num_class, low_dim=512,
                 model_moving_average_decay=0.99,
                 proto_moving_average_decay=0.99,
                 fusion_type='reconstruct_double_detach', normalize=True, all_normalize=False,
                 force_no_shuffle_bn=False, mixup_sample_type='low', select_src_by_tgt_similarity=False,
                 src_keep_ratio=0.5,
                 ):
        super(BasicMixContrastiveModel, self).__init__()
        self.fusion_type = fusion_type
        self.online_network = build_models(model_dict)
        self.target_network = build_models(model_dict)
        self.target_ema_updater = EMA(model_moving_average_decay)
        self.proto_moving_average_decay = proto_moving_average_decay
        self.normalize = normalize
        self.num_class = num_class
        self.low_dim = low_dim
        self.all_normalize = all_normalize
        self.force_no_shuffle_bn = force_no_shuffle_bn
        self.mixup_sample_type = mixup_sample_type
        self.select_src_by_tgt_similarity = select_src_by_tgt_similarity
        self.src_keep_ratio = src_keep_ratio
        rank, _ = get_dist_info()
        #
        self.online_classifier = build_models(classifier_dict)
        self.target_classifier = build_models(classifier_dict)
        self.gpu_device = 'cuda:{}'.format(rank)
        #
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.online_classifier.parameters(), self.target_classifier.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    # TODO: 模型输入需要考虑多个输入的情况
    def forward(self, x1, x2=None, src_mixup=False, **kwargs):
        threshold = kwargs.get('threshold', None)
        if 'threshold' in kwargs:
            kwargs.pop('threshold')
        src_labeled_size = kwargs.get('src_labeled_size', None)
        if 'src_labeled_size' in kwargs:
            kwargs.pop('src_labeled_size')
        tgt_labeled_size = kwargs.get('tgt_labeled_size', 0)  # 对于无监督域适应，这里的值为0
        if 'tgt_labeled_size' in kwargs:
            kwargs.pop('tgt_labeled_size')
        #
        x1_img = x1
        x2_img = x2
        if x2 is not None:
            all_labeled_size = src_labeled_size + tgt_labeled_size
            with torch.no_grad():
                # TODO:挪到后面去
                self.update_moving_average()
                #
                tmp_shuffle_bn_flag = True if not self.force_no_shuffle_bn else False
                target_res = self.model_forward(x1_img, x2_img, model_type='target', shuffle_bn=tmp_shuffle_bn_flag,
                                                **kwargs)
            mix_feat_list = None
            lambda_mixup = None
            #
            online_res = self.model_forward(x1_img.clone(), x2_img.clone(), model_type='online', **kwargs)
            #
            return online_res, target_res, lambda_mixup, mix_feat_list
        else:
            return self.single_input_forward(x1_img, **kwargs)

    def update_moving_average(self):
        update_moving_average(self.target_ema_updater, self.target_network, self.online_network)
        update_moving_average(self.target_ema_updater, self.target_classifier, self.online_classifier)

    def optim_parameters(self, lr):
        params = []
        if hasattr(self.online_network, 'optim_parameters'):
            tmp_params = self.online_network.optim_parameters(lr)
        else:
            raise RuntimeError('not supported')
        params.extend(tmp_params)
        #
        if hasattr(self.online_classifier, 'optim_parameters'):
            tmp_params = self.online_classifier.optim_parameters(lr * 10)
        else:
            raise RuntimeError('not supported')
        params.extend(tmp_params)
        return params

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all, device=self.gpu_device)
        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this]

    def feat_prob_fusion(self, feat, prob, model='online'):
        if self.fusion_type == 'prob':
            return prob
        elif self.fusion_type == 'feat':
            return feat
        elif self.fusion_type == 'outer_product':
            return self.random_layer((feat, prob))
        elif self.fusion_type == 'reconstruct':
            if model == 'online':
                fc_weight = self.online_classifier.fc2.weight
            else:
                fc_weight = self.target_classifier.fc2.weight
            return prob.mm(fc_weight)
        elif self.fusion_type == 'reconstruct_detach':
            if model == 'online':
                fc_weight = self.online_classifier.fc2.weight.detach()
            else:
                fc_weight = self.target_classifier.fc2.weight.detach()
            fc_weight = F.normalize(fc_weight, dim=1) if (not self.normalize) or self.all_normalize else fc_weight
            return prob.mm(fc_weight)
        elif self.fusion_type == 'reconstruct_double_detach':
            if model == 'online':
                fc_weight = self.online_classifier.fc2.weight.detach()
            else:
                fc_weight = self.target_classifier.fc2.weight.detach()
            fc_weight = F.normalize(fc_weight, dim=1) if (not self.normalize) or self.all_normalize else fc_weight
            #
            new_prob = F.softmax(feat.mm(fc_weight.detach().t()) / self.online_classifier.temp, dim=1)
            return new_prob.mm(fc_weight)
        elif self.fusion_type == 'l1_reconstruct_detach':
            if model == 'online':
                fc_weight = self.online_classifier.fc2.weight
            else:
                fc_weight = self.target_classifier.fc2.weight
            # 计算系数
            coeff = feat.mm(fc_weight.t())
            coeff = coeff / torch.sum(coeff, dim=1, keepdim=True)
            return coeff.mm(fc_weight.detach())
        else:
            raise RuntimeError("need fusion_type")

    def test_forward(self, x1_img, **kwargs):
        feat = self.online_network(x1_img, **kwargs)
        logits = self.online_classifier(feat)
        prob = F.softmax(logits, dim=-1)
        fusion_feat = self.feat_prob_fusion(feat, prob, model='online')
        online_pred = self.online_classifier(feat)
        target_feat = self.target_network(x1_img, **kwargs)
        target_pred = self.target_classifier(target_feat)
        return feat, fusion_feat, online_pred, target_pred

    def model_forward(self, x1_img, x2_img, model_type=None, shuffle_bn=False, **kwargs):
        if model_type == "online":
            feature_extractor = self.online_network
            classifier = self.online_classifier
        elif model_type == 'target':
            feature_extractor = self.target_network
            classifier = self.target_classifier
        else:
            raise RuntimeError('wrong model type specified')
        x1_shape = x1_img.shape[0]
        img_concat = torch.cat((x1_img, x2_img))
        if shuffle_bn:
            img_concat, idx_unshuffle = self._batch_shuffle_ddp(img_concat)
        feat = feature_extractor(img_concat)
        if shuffle_bn:
            feat = self._batch_unshuffle_ddp(feat, idx_unshuffle)
        logits = classifier(feat)
        prob = F.softmax(logits, dim=-1)
        contrastive_feat = self.feat_prob_fusion(feat, prob, model=model_type)
        contrastive_feat = F.normalize(contrastive_feat, dim=1) if self.normalize else contrastive_feat
        #
        strong_logits = logits[0:x1_shape]
        weak_logits = logits[x1_shape:]
        strong_prob = prob[0:x1_shape]
        weak_prob = prob[x1_shape:]
        strong_contrastive_feat = contrastive_feat[0:x1_shape]
        weak_contrastive_feat = contrastive_feat[x1_shape:]
        output = {
            'strong_logits': strong_logits,
            'weak_logits': weak_logits,
            'strong_prob': strong_prob,
            'weak_prob': weak_prob,
            'strong_contrastive_feat': strong_contrastive_feat,
            'weak_contrastive_feat': weak_contrastive_feat,
        }
        return output

    def single_input_forward(self, x1_img, **kwargs):
        if self.training:
            feat = self.online_network(x1_img, **kwargs)
            pred = self.online_classifier(feat, reverse=True, eta=1.0)
            return pred
        else:
            res = self.test_forward(x1_img, **kwargs)
            return res
