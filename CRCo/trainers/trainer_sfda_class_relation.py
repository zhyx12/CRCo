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
import numpy as np
from scipy.spatial.distance import cdist
from basicda.utils import concat_all_gather
import pickle


@VALIDATOR.register_module(name='sfda_class_relation')
class ValidatorSFDAClassRelation(BaseValidator):
    def __init__(self, basic_parameters):
        super(ValidatorSFDAClassRelation, self).__init__(**basic_parameters)

    def eval_iter(self, val_batch_data):
        # val_img, val_label, val_name = val_batch_data
        val_img = val_batch_data['img']
        val_label = val_batch_data['gt_label'].squeeze(1)
        val_metas = val_batch_data['img_metas']
        with torch.no_grad():
            feat, pred_unlabeled, target_feat, target_logits = self.model_dict['base_model'](val_img)
        return {'img': val_img,
                'gt': val_label,
                'img_metas': val_metas,
                'feat': feat,
                'pred': pred_unlabeled,
                'target_pred': target_logits,
                }


@TRAINER.register_module('sfda_class_relation')
class TrainerSFDAClassRelation(BaseTrainer):
    def __init__(self, basic_parameters,
                 #
                 baseline_type='IM',
                 lambda_ent=1.0, lambda_div=1.0, lambda_aad=1.0, fix_classifier=True,
                 prob_threshold=0.95,
                 feat_dim=256,
                 pseudo_update_interval=100000, threshold=0,
                 num_k=4, num_m=3, lambda_near=0.0,
                 lambda_fixmatch=0.0, fixmatch_start=0, fixmatch_type='class_relation',
                 use_cluster_label_for_fixmatch=False, lambda_fixmatch_temp=0.07,
                 bank_size=512, lambda_nce=1.0, lambda_temp=0.07,
                 non_diag_alpha=1.0, add_current_data_for_instance=False,
                 use_only_current_batch_for_instance=False, max_iters=15000, beta=0.0,
                 ):
        super(TrainerSFDAClassRelation, self).__init__(**basic_parameters)
        self.num_class = self.train_loaders[0].dataset.n_classes
        self.baseline_type = baseline_type
        self.lambda_ent = lambda_ent
        self.lambda_div = lambda_div
        self.lambda_aad = lambda_aad
        self.prob_threshold = prob_threshold
        self.lambda_nce = lambda_nce
        self.lambda_temp = lambda_temp
        self.pseudo_update_interval = pseudo_update_interval
        self.threshold = threshold
        self.num_k = num_k
        self.num_m = num_m
        self.lambda_near = lambda_near
        self.lambda_fixmatch = lambda_fixmatch
        self.fixmatch_start = fixmatch_start
        self.fixmatch_type = fixmatch_type
        self.use_cluster_label_for_fixmatch = use_cluster_label_for_fixmatch
        self.lambda_fixmatch_temp = lambda_fixmatch_temp
        self.bank_size = bank_size
        self.non_diag_alpha = non_diag_alpha
        self.class_contrastive_simmat = None
        self.instance_contrastive_simmat = None
        self.add_current_data_for_instance = add_current_data_for_instance
        self.use_only_current_batch_for_instance = use_only_current_batch_for_instance
        self.max_iters = max_iters
        self.beta = beta
        #
        rank, world_size = get_dist_info()
        self.world_size = world_size
        # 增加记录
        if self.local_rank == 0:
            log_names = ['info_nce', 'mean_max_prob', 'mask', 'mask_acc', 'cluster_mask_acc']
            if baseline_type == "IM":
                log_names.extend(['ent', 'div'])
            elif baseline_type == 'AaD':
                log_names.extend(['aad_pos', 'aad_neg'])
            loss_metrics = MetricsLogger(log_names=log_names, group_name='loss', log_interval=self.log_interval)
            self.register_hook(loss_metrics)
        #
        if fix_classifier:
            base_model = self.model_dict['base_model']
            for param in base_model.module.online_classifier.parameters():
                param.requires_grad = False
        #
        num_image = len(self.train_loaders[0].dataset)
        self.weak_feat_bank = torch.randn(num_image, feat_dim).to('cuda:{}'.format(rank))
        self.weak_score_bank = torch.randn(num_image, self.num_class).to('cuda:{}'.format(rank))
        self.label_bank = torch.zeros((num_image,), dtype=torch.long).to('cuda:{}'.format(rank))
        self.pseudo_label_bank = torch.zeros((num_image,), dtype=torch.long).to('cuda:{}'.format(rank))
        self.class_prototype_bank = torch.randn(self.num_class, feat_dim).to('cuda:{}'.format(rank))
        self.strong_feat_bank = torch.randn(num_image, feat_dim).to('cuda:{}'.format(rank))
        self.strong_score_bank = torch.randn(num_image, self.num_class).to('cuda:{}'.format(rank))
        self.aad_weak_feat_bank = torch.randn(num_image, feat_dim).to('cuda:{}'.format(rank))
        self.aad_weak_score_bank = torch.randn(num_image, self.num_class).to('cuda:{}'.format(rank))
        #
        self.weak_negative_bank = torch.randn(bank_size, self.num_class).to('cuda:{}'.format(rank))
        self.weak_negative_bank_ptr = torch.zeros(1, dtype=torch.long).to('cuda:{}'.format(rank))
        self.strong_negative_bank = torch.randn(bank_size, self.num_class).to('cuda:{}'.format(rank))
        self.strong_negative_bank_ptr = torch.zeros(1, dtype=torch.long).to('cuda:{}'.format(rank))
        self.ngative_img_ind_bank = torch.zeros((bank_size,), dtype=torch.long).to('cuda:{}'.format(rank))

    def train_iter(self, *args):
        tgt_unlabeled_img_weak = args[0][0]['img'].squeeze(0)
        tgt_unlabeled_img_strong = args[0][1]['img'].squeeze(0)
        tgt_unlabeled_img_strong_2 = args[0][2]['img'].squeeze(0)
        tgt_domain_label = args[0][0].get('domain_label', None)
        tgt_img_ind = args[0][0]['image_ind']
        tgt_unlabeled_label = args[0][0]['gt_label'].squeeze(0)
        # 
        tgt_unlabeled_size = tgt_unlabeled_img_weak.shape[0]
        #
        batch_metrics = {}
        batch_metrics['loss'] = {}
        #
        base_model = self.model_dict['base_model']
        #
        if self.iteration % self.pseudo_update_interval == 0:
            self.update_bank()
            self.obtain_all_label()
        #
        if self.iteration == 0:
            self.class_contrastive_simmat = self.obtain_sim_mat(usage='class_contrastive')
            self.instance_contrastive_simmat = self.obtain_sim_mat(usage='instance_contrastive')
        #
        self.zero_grad_all()
        #
        all_weak_img = tgt_unlabeled_img_weak
        all_strong_img = torch.cat((tgt_unlabeled_img_strong, tgt_unlabeled_img_strong_2), dim=0)
        #
        tmp_res = base_model(all_strong_img, all_weak_img, src_labeled_size=0, tgt_unlabeled_size=tgt_unlabeled_size,
                             train_iter=self.iteration)
        online_output, target_output, _, _ = tmp_res
        online_strong_logits = online_output['strong_logits']
        target_strong_logits = target_output['strong_logits']
        target_weak_logits = target_output['weak_logits']
        target_weak_feat = target_output['weak_feat']
        online_weak_logits = online_output['weak_logits']
        online_weak_feat = online_output['weak_feat']
        online_strong_prob = F.softmax(online_strong_logits, dim=-1)
        target_strong_prob = F.softmax(target_strong_logits, dim=-1)
        online_weak_prob = F.softmax(online_weak_logits, dim=-1)
        target_weak_prob = F.softmax(target_weak_logits, dim=-1)
        #
        # timely updated weak bank
        self.update_weak_bank_timely(target_weak_feat, target_weak_prob, tgt_img_ind)
        # baseline
        loss = self.baseline_loss(online_weak_prob, target_weak_feat, batch_metrics, online_weak_logits)
        #
        # fixmatch损失
        pseudo_label = torch.softmax(target_weak_logits.detach(), dim=-1)
        max_probs, tgt_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.prob_threshold).float().detach()
        pred_right = torch.sum((tgt_u == tgt_unlabeled_label.squeeze(1)) * mask) / torch.sum(mask)
        if self.use_cluster_label_for_fixmatch:
            tgt_u = self.obtain_batch_label(online_weak_feat, None)
            # tgt_u = self.obtain_batch_label(target_weak_feat, None)
            cluster_acc = torch.sum((tgt_u == tgt_unlabeled_label.squeeze(1)) * mask) / torch.sum(mask)
        else:
            cluster_acc = torch.tensor(0)
        mask_val = torch.sum(mask).item() / mask.shape[0]
        self.high_ratio = mask_val
        if self.iteration >= self.fixmatch_start:
            ###########
            if self.fixmatch_type == 'orig':
                strong_aug_pred = online_strong_logits[0:tgt_unlabeled_size]
                loss_consistency = (F.cross_entropy(strong_aug_pred, tgt_u, reduction='none') * mask).mean()
                loss += loss_consistency * self.lambda_fixmatch * 0.5
                strong_aug_pred = online_strong_logits[tgt_unlabeled_size:]
                loss_consistency = (F.cross_entropy(strong_aug_pred, tgt_u, reduction='none') * mask).mean()
                loss += loss_consistency * self.lambda_fixmatch * 0.5
            #
            elif self.fixmatch_type == 'class_relation':
                #
                loss_1 = self.class_contrastive_loss(online_strong_prob[0:tgt_unlabeled_size], tgt_u, mask)
                loss_2 = self.class_contrastive_loss(online_strong_prob[tgt_unlabeled_size:], tgt_u, mask)
                loss += (loss_1 + loss_2) * self.lambda_fixmatch * 0.5
            else:
                raise RuntimeError('wrong fixmatch type')
        # #
        # # constrastive loss
        all_k_strong = target_strong_prob
        all_k_weak = target_weak_prob
        weak_feat_for_backbone = online_weak_prob
        k_weak_for_backbone = all_k_weak
        k_strong_for_backbone = all_k_strong[0:tgt_unlabeled_size]
        strong_feat_for_backbone = online_strong_prob[0:tgt_unlabeled_size]
        k_strong_2 = all_k_strong[tgt_unlabeled_size:]
        feat_strong_2 = online_strong_prob[tgt_unlabeled_size:]
        if self.use_only_current_batch_for_instance:
            tmp_weak_negative_bank = online_weak_prob
            tmp_strong_negative_bank = strong_feat_for_backbone
            neg_ind = tgt_img_ind
            self.num_k = 1
        else:
            if self.add_current_data_for_instance:
                tmp_weak_negative_bank = torch.cat((self.weak_negative_bank, online_weak_prob), dim=0)
                tmp_strong_negative_bank = torch.cat((self.strong_negative_bank, strong_feat_for_backbone), dim=0)
                neg_ind = torch.cat((self.ngative_img_ind_bank, tgt_img_ind))
            else:
                tmp_weak_negative_bank = self.weak_negative_bank
                tmp_strong_negative_bank = self.strong_negative_bank
                neg_ind = self.ngative_img_ind_bank
        #
        info_nce_loss_1 = self.instance_contrastive_loss(strong_feat_for_backbone, k_weak_for_backbone,
                                                         tmp_weak_negative_bank,
                                                         self_ind=tgt_img_ind, neg_ind=neg_ind)
        info_nce_loss_3 = self.instance_contrastive_loss(strong_feat_for_backbone, k_strong_2,
                                                         tmp_strong_negative_bank,
                                                         self_ind=tgt_img_ind, neg_ind=neg_ind)
        info_nce_loss_2 = self.instance_contrastive_loss(weak_feat_for_backbone, k_strong_for_backbone,
                                                         tmp_strong_negative_bank,
                                                         self_ind=tgt_img_ind, neg_ind=neg_ind)
        info_nce_loss = (info_nce_loss_1 + info_nce_loss_2 + info_nce_loss_3) / 3.0
        #
        loss += info_nce_loss * self.lambda_nce
        loss.backward()
        #
        self.step_grad_all()
        #
        #
        self.update_negative_bank(target_weak_prob, target_strong_prob[0:tgt_unlabeled_size, :], tgt_img_ind)
        #
        # batch_metrics['loss']['kld'] = kld_loss.item()
        batch_metrics['loss']['info_nce'] = info_nce_loss.item() if isinstance(info_nce_loss,
                                                                               torch.Tensor) else info_nce_loss
        batch_metrics['loss']['mean_max_prob'] = torch.mean(max_probs).item()
        batch_metrics['loss']['mask'] = mask_val
        batch_metrics['loss']['mask_acc'] = pred_right.item()
        batch_metrics['loss']['cluster_mask_acc'] = cluster_acc.item()
        return batch_metrics

    def load_pretrained_model(self, weights_path):
        logger = get_root_logger()
        weights = torch.load(weights_path, map_location='cpu')
        weights = weights['base_model']
        for key in weights:
            key_split = key.split('.')
            if key_split[1] in ['target_network', 'target_classifier']:
                key_split[1] = key_split[1].replace('target', 'online')
                online_key = ('.').join(key_split)
                weights[key] = weights[online_key]
        self.model_dict['base_model'].load_state_dict(weights, strict=False)
        logger.info('load pretrained model {}'.format(weights_path))

    def update_bank(self):
        self.set_eval_state()
        base_model = self.model_dict['base_model']
        shape = 0
        with torch.no_grad():
            for data in self.train_loaders[0]:
                img = data[0]['img']
                img_ind = data[0]['image_ind']
                img_label = data[0]['gt_label']
                tmp_res = base_model(img)
                feat, logits, target_feat, target_logits = tmp_res
                #
                tmp_feat = feat
                tmp_score = F.softmax(logits, dim=-1)
                feat = concat_all_gather(tmp_feat)
                score = concat_all_gather(tmp_score)
                img_ind = concat_all_gather(img_ind.to('cuda:{}'.format(self.local_rank)))
                img_label = concat_all_gather(img_label.to('cuda:{}'.format(self.local_rank)))
                self.weak_feat_bank[img_ind] = F.normalize(feat, dim=-1)
                self.weak_score_bank[img_ind] = score
                self.label_bank[img_ind] = img_label.squeeze(1).to('cuda:{}'.format(self.local_rank))
                #
                if self.iteration == 0:
                    target_feat = concat_all_gather(target_feat)
                    target_score = concat_all_gather(F.softmax(target_logits, dim=-1))
                    self.aad_weak_feat_bank[img_ind] = F.normalize(target_feat, dim=-1)
                    self.aad_weak_score_bank[img_ind] = target_score
                #
                shape += img.shape[0]
        print('rank {}, shape {}'.format(self.local_rank, shape))
        self.set_train_state()

    def obtain_all_label(self):
        all_output = self.weak_score_bank
        all_fea = self.weak_feat_bank
        all_label = self.label_bank
        #
        _, predict = torch.max(all_output, 1)

        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        print('orig acc is {}'.format(accuracy))
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        all_fea = all_fea.float().cpu().numpy()
        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        #
        predict = predict.cpu().numpy()
        for _ in range(2):
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            cls_count = np.eye(K)[predict].sum(axis=0)
            labelset = np.where(cls_count > self.threshold)
            labelset = labelset[0]

            dd = cdist(all_fea, initc[labelset], 'cosine')
            pred_label = dd.argmin(axis=1)
            predict = labelset[pred_label]

            aff = np.eye(K)[predict]

        acc = np.sum(predict == all_label.cpu().float().numpy()) / len(all_fea)
        # log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
        print('acc is {}'.format(acc))
        self.pseudo_label_bank = torch.from_numpy(predict).to("cuda:{}".format(self.local_rank))
        prototype = torch.from_numpy(initc).to("cuda:{}".format(self.local_rank)).to(torch.float32)
        self.class_prototype_bank = F.normalize(prototype)

    def obtain_batch_label(self, feat, score, prototype=None):
        feat_1 = F.normalize(feat.detach())
        if prototype is None:
            prototype = F.normalize(self.class_prototype_bank.detach())
        else:
            prototype = F.normalize(prototype.detach())
        cos_similarity = torch.mm(feat_1, prototype.t())
        pred_label = torch.argmax(cos_similarity, dim=1)
        return pred_label

    def my_sim_compute(self, prob_1, prob_2, sim_mat, expand=True):
        """
        prob_1: B1xC
        prob_2: B2xC
        sim_mat: CxC
        expand: True, computation conducted between every element in prob_2 and prob_1; Fasle, need B1=B2
        """
        b1 = prob_1.shape[0]
        b2 = prob_2.shape[0]
        cls_num = prob_1.shape[1]
        if expand:
            prob_1 = prob_1.unsqueeze(2).unsqueeze(1).expand(-1, b2, -1, -1)  # B1xB2xCx1
            prob_2 = prob_2.unsqueeze(1).unsqueeze(0).expand(b1, -1, -1, -1)  # B1xB2x1xC
            prob_1 = prob_1.reshape(b1 * b2, cls_num, 1)
            prob_2 = prob_2.reshape(b1 * b2, 1, cls_num)
            sim = torch.sum(torch.sum(torch.bmm(prob_1, prob_2) * sim_mat, -1), -1)
            sim = sim.reshape(b1, b2)
        else:
            prob_1 = prob_1.unsqueeze(2)  # BxCx1
            prob_2 = prob_2.unsqueeze(1)  # Bx1xC
            sim = torch.sum(torch.sum(torch.bmm(prob_1, prob_2) * sim_mat, -1), -1)
            sim = sim.reshape(b1, 1)
        return sim

    def my_sim_compute_2(self, query_prob, pos_prob, neg_prob, sim_mat):
        pos_logits = my_sim_compute(query_prob, pos_prob, sim_mat, expand=False)
        neg_logits = my_sim_compute(query_prob, neg_prob, sim_mat, expand=True)
        all_logits = torch.cat((pos_logits, neg_logits), dim=1)
        return all_logits

    def IM_loss(self, score):
        softmax_out = score
        loss_ent = -torch.mean(torch.sum(softmax_out * torch.log(softmax_out + 1e-5), 1)) * 0.5
        tensors_gather = [torch.ones_like(softmax_out) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, softmax_out, async_op=False)
        self_ind = self.local_rank
        msoftmax = 0
        for i in range(len(tensors_gather)):
            if i == self_ind:
                msoftmax += softmax_out.mean(dim=0)
            else:
                msoftmax += tensors_gather[i].mean(dim=0)
        loss_div = torch.sum(msoftmax * torch.log(msoftmax + 1e-5)) * 0.5
        return loss_ent, loss_div

    def AaD_loss(self, score, feat):
        with torch.no_grad():
            normalized_feat = F.normalize(feat, dim=-1)
            normalized_feat = concat_all_gather(normalized_feat)
            distance = normalized_feat @ self.aad_weak_feat_bank.T
            _, idx_near = torch.topk(distance,
                                     dim=-1,
                                     largest=True,
                                     k=self.num_k + 1)
            idx_near = idx_near[:, 1:]  # batch x K
            score_near = self.aad_weak_score_bank[idx_near]  # batch x K x C
        #
        tensors_gather = [torch.ones_like(score) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, score, async_op=False)
        tensors_gather[self.local_rank] = score
        outputs = torch.cat(tensors_gather, dim=0)
        softmax_out_un = outputs.unsqueeze(1).expand(-1, self.num_k, -1)  # batch x K x C
        #
        mask = torch.ones((normalized_feat.shape[0], normalized_feat.shape[0]))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        #####################
        loss_1 = torch.mean(
            (F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1)).sum(1))
        copy = outputs.T  # .detach().clone()#
        dot_neg = outputs @ copy  # batch x batch
        dot_neg = (dot_neg * mask.to("cuda:{}".format(self.local_rank))).sum(-1)  # batch
        neg_pred = torch.mean(dot_neg)
        return loss_1, neg_pred

    def baseline_loss(self, score, feat, batch_metrics, logits):
        if self.baseline_type == "IM":
            loss_ent, loss_div = self.IM_loss(score)
            batch_metrics['loss']['ent'] = loss_ent.item()
            batch_metrics['loss']['div'] = loss_div.item()
            return loss_ent * self.lambda_ent + loss_div * self.lambda_div
        elif self.baseline_type == 'AaD':
            loss_aad_pos, loss_aad_neg = self.AaD_loss(score, feat)
            batch_metrics['loss']['aad_pos'] = loss_aad_pos.item()
            batch_metrics['loss']['aad_neg'] = loss_aad_neg.item()
            tmp_lambda = (1 + 10 * self.iteration / self.max_iters) ** (-self.beta)
            return (loss_aad_pos + loss_aad_neg * tmp_lambda) * self.lambda_aad
        else:
            raise RuntimeError('wrong type of baseline')

    def class_contrastive_loss(self, score, label, mask):
        all_other_prob = torch.eye(self.num_class).to("cuda:{}".format(self.local_rank))
        new_logits = self.my_sim_compute(score, all_other_prob, self.class_contrastive_simmat,
                                         expand=True) / self.lambda_fixmatch_temp
        loss_consistency = (F.cross_entropy(new_logits, label, reduction='none') * mask).mean()
        return loss_consistency

    def instance_contrastive_loss(self, query_feat, key_feat, neg_feat, self_ind, neg_ind):
        pos_logits = self.my_sim_compute(query_feat, key_feat, self.instance_contrastive_simmat, expand=False) * 0.5
        neg_logits = self.my_sim_compute(query_feat, neg_feat, self.instance_contrastive_simmat, expand=True) * 0.5
        all_logits = torch.cat((pos_logits, neg_logits), dim=1) / self.lambda_temp
        #
        constrastive_labels = self.get_contrastive_labels(query_feat)
        info_nce_loss = F.cross_entropy(all_logits, constrastive_labels) * 0.5
        return info_nce_loss

    def get_contrastive_labels(self, query_feat):
        current_batch_size = query_feat.shape[0]
        constrastive_labels = torch.zeros((current_batch_size,), dtype=torch.long,
                                          device='cuda:{}'.format(self.local_rank))
        return constrastive_labels

    def obtain_neg_mask(self, self_ind, neg_ind):
        self_size = self_ind.shape[0]
        neg_size = neg_ind.shape[0]
        final_mask = torch.ones((self_size, neg_size)).to('cuda:{}'.format(self.local_rank))
        # 获取self_ind对应的特征
        self_feat = self.aad_weak_feat_bank[self_ind, :]
        # 计算self_feat的近邻
        distance = self_feat @ self.aad_weak_feat_bank.T
        _, near_ind = torch.topk(distance,
                                 dim=-1,
                                 largest=True,
                                 k=self.num_k + 1)
        #
        neg_ind = neg_ind.unsqueeze(0).unsqueeze(2).expand(self_size, -1, 1)
        near_ind = near_ind.unsqueeze(1)
        #
        mask_ind = (neg_ind == near_ind).sum(-1)
        final_mask[mask_ind > 0] = 0
        return final_mask

    def update_negative_bank(self, weak_score, strong_score, img_ind):
        """
        update score bank in trainer
        :param weak_score: weak score output by teacher model
        :param strong_score: strong score output by teacher model
        :img_ind: image index
        :return: None
        """

        def update_bank(new_score, bank, ptr, img_ind=None):
            all_score = concat_all_gather(new_score)
            batch_size = all_score.shape[0]
            start_point = int(ptr)
            end_point = min(start_point + batch_size, self.bank_size)
            real_size = end_point - start_point
            bank[start_point:end_point, :] = all_score[0:(end_point - start_point), :]
            if img_ind is not None:
                img_ind = concat_all_gather(img_ind)
                self.ngative_img_ind_bank[start_point:end_point] = img_ind[0:(end_point - start_point)]
            if end_point == self.bank_size:
                ptr[0] = 0
            else:
                ptr += batch_size

        update_bank(weak_score, self.weak_negative_bank, self.weak_negative_bank_ptr, img_ind)
        update_bank(strong_score, self.strong_negative_bank, self.strong_negative_bank_ptr)
        # print(self.weak_negative_bank_ptr, self.strong_negative_bank_ptr)

    def update_weak_bank_timely(self, feat, score, ind):
        with torch.no_grad():
            single_output_f_ = F.normalize(feat).detach().clone()
            tmp_softmax_out = score
            tmp_img_ind = ind
            output_f_ = concat_all_gather(single_output_f_)
            tmp_softmax_out = concat_all_gather(tmp_softmax_out)
            tmp_img_ind = concat_all_gather(tmp_img_ind)
            #
            self.aad_weak_feat_bank[tmp_img_ind] = output_f_.detach().clone()
            self.aad_weak_score_bank[tmp_img_ind] = tmp_softmax_out.detach().clone()

    def obtain_sim_mat(self, usage):
        base_model = self.model_dict['base_model']
        fc_weight = base_model.module.online_classifier.fc.weight_v.detach()
        normalized_fc_weight = F.normalize(fc_weight)
        sim_mat_orig = normalized_fc_weight @ normalized_fc_weight.T
        eye_mat = torch.eye(self.num_class).to("cuda:{}".format(self.local_rank))
        non_eye_mat = 1 - eye_mat
        sim_mat = (eye_mat + non_eye_mat * sim_mat_orig * self.non_diag_alpha).clone()
        return sim_mat
