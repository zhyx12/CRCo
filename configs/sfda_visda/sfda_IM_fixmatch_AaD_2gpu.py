_base_ = [
    '../_base_/cls_datasets/sfda_visda/sfda_visda_target_2gpu.py',
    '../_base_/cls_models/sfda_visda/resnet_101_sfda_visda_target_model.py',
]

# models['find_unused_parameters']=False

log_interval = 100
val_interval = 1000

control = dict(
    log_interval=log_interval,
    max_iters=15000,
    val_interval=val_interval,
    cudnn_deterministic=True,
    save_interval=1000,
    max_save_num=1,
    seed=2022,
    pretrained_model="fill path_to_source_models here"
)

train = dict(
    baseline_type='AaD',
    lambda_nce=1.0,
    lambda_ent=0.0,
    lambda_div=0.0,
    fix_classifier=True,
    pseudo_update_interval=500,
    lambda_fixmatch=1.0,
    prob_threshold=0.95,
    use_cluster_label_for_fixmatch=True,
    max_iters=15000,
    beta=5.0,
    num_k=5,
)

test = dict(
custom_hooks=[
        dict(type='ClsAccuracy', dataset_index=0, pred_key='pred',class_acc=True),
        # dict(type='ClsAccuracy', dataset_index=0, pred_key='target_pred'),
        # dict(type='ClsAccuracy', dataset_index=1, pred_key='pred'),
        # dict(type='ClsAccuracy', dataset_index=0, pred_key='target_pred'),
        dict(type='ClsBestAccuracyByVal', patience=100, priority="LOWEST")
    ]
)
