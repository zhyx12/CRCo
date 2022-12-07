_base_ = [
    '../_base_/cls_datasets/sfda_officehome/sfda_officehome_target_4gpu_AC.py',
    '../_base_/cls_models/sfda_officehome/ssrt_vit_b_sfda_simplified_officehome_fixlr_4e4.py'
]

# models = dict(find_unused_parameters=False)

log_interval = 100
val_interval = 300

control = dict(
    log_interval=log_interval,
    max_iters=5000,
    val_interval=val_interval,
    cudnn_deterministic=True,
    save_interval=500,
    max_save_num=1,
    seed=2,
    pretrained_model='fill path_to_source_models here'
)

train = dict(
    baseline_type='IM',
    lambda_nce=1.0,
    lambda_ent=1.0,
    lambda_div=1.0,
    fix_classifier=True,
    pseudo_update_interval=50,
    lambda_fixmatch=1.0,
    prob_threshold=0.95,
    use_cluster_label_for_fixmatch=True,
    lambda_fixmatch_temp=0.07,
)

test = dict(
custom_hooks=[
        dict(type='ClsAccuracy', dataset_index=0, pred_key='pred'),
        # dict(type='ClsAccuracy', dataset_index=0, pred_key='target_pred'),
        dict(type='ClsAccuracy', dataset_index=1, pred_key='pred'),
        # dict(type='ClsAccuracy', dataset_index=0, pred_key='target_pred'),
        dict(type='ClsBestAccuracyByTest', patience=1000, priority="LOWEST")
    ]
)
