

def get_model_name_scope_by_name(name):


    scope = 'mmdet'
    if name == 'frcn_r50':
        model_name = 'faster-rcnn_r50_fpn_1x_coco'
    elif name == 'frcn_r101':
        model_name = 'faster-rcnn_r101_fpn_1x_coco'
    elif name == 'mrcn_r50':
        model_name = 'mask-rcnn_r50_fpn_1x_coco'
    elif name == 'mrcn_r101':
        model_name = 'mask-rcnn_r101_fpn_1x_coco'
    elif name == 'mrcn_swin':
        model_name = 'mask-rcnn_swin-t-p4-w7_fpn_1x_coco'
    elif name == 'yolov3_608':
        model_name = 'yolov3_d53_mstrain-608_273e_coco'
    elif name == 'yolov3_416':
        model_name = 'yolov3_d53_mstrain-416_273e_coco'
    elif name == 'ddetr_r50':
        model_name = 'deformable-detr_r50_16xb2-50e_coco'
    elif name == 'dino_swin':
        model_name = 'dino-5scale_swin-l_8xb2-12e_coco'
    elif name == 'dino_r50':
        model_name = 'dino-4scale_r50_8xb2-12e_coco'
    elif name == 'autoassign_r50':
        model_name = 'autoassign_r50-caffe_fpn_1x_coco'
    elif name == 'sparse_rcnn_r50':
        model_name = 'sparse-rcnn_r50_fpn_1x_coco'
    elif name == 'sparse_rcnn_r101':
        model_name = 'sparse-rcnn_r101_fpn_ms-480-800-3x_coco'
    elif name == 'ssd_vgg16_512':
        model_name = 'ssd512_coco'
    elif name == 'diffusiondet':
        model_name = 'diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco'
    
    elif name == 'grounding_dino':
        model_name = 'grounding_dino_swin-t_pretrain_obj365_goldg_cap4m'
    elif name == 'grounding_dino_t':
        model_name = 'grounding_dino_swin-t_pretrain_obj365_goldg_cap4m'
    elif name == 'grounding_dino_tf':
        model_name = 'grounding_dino_swin-t_finetune_16xb2_1x_coco'
    elif name == 'grounding_dino_b':
        model_name = 'grounding_dino_swin-b_pretrain_mixeddata'
    elif name == 'grounding_dino_bf':
        model_name = 'grounding_dino_swin-b_finetune_16xb2_1x_coco'
    elif name == 'grounding_dino_r50':
        model_name = 'grounding_dino_r50_scratch_8xb2_1x_coco'


    elif name == 'mm_grounding_dino_t':
        model_name = 'grounding_dino_swin-t_pretrain_obj365_goldg_v3det'
    elif name == 'mm_grounding_dino_b':
        model_name = 'grounding_dino_swin-b_pretrain_obj365_goldg_v3det'



    elif name == 'mask2former_r50':
        model_name = 'mask2former_r50_8xb2-lsj-50e_coco'
    elif name == 'mask2former_swin':
        model_name = 'mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco'
    elif name == 'mask-rcnn_convnext':
        model_name = 'mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco'
    elif name == 'dab-detr_r50':
        model_name = 'dab-detr_r50_8xb2-50e_coco'
    elif name == 'frcn-dcnv2':
        model_name = 'faster-rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco'


    elif name == 'yolov8_s':
        model_name = 'yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco'
        scope = 'mmyolo'
    elif name == 'yolov5_s':
        model_name = 'yolov5_s_mask-refine-v61_syncbn_fast_8xb16-300e_coco'
        scope = 'mmyolo'

    else:
        raise ValueError(f'unknow model name: {name}')

    


    return model_name, scope


def get_all_name():
    name_all = [
        'frcn_r50',
        'frcn_r101',
        'mrcn_r50',
        'mrcn_r101',
        'mrcn_swin',
        'yolov3_608',
        'yolov3_416',
        'ddetr_r50',
        'dino_swin',
        'dino_r50',
        'autoassign_r50',
        'sparse_rcnn_r50',
        'sparse_rcnn_r101',
        'ssd_vgg16_512',
        'diffusiondet',
        'grounding_dino',
        'grounding_dino_t',
        'grounding_dino_tf',
        'grounding_dino_b',
        'grounding_dino_bf',
        'grounding_dino_r50',
        'mm_grounding_dino_t',
        'mm_grounding_dino_b',
        'mask2former_r50',
        'mask2former_swin',
        'mask-rcnn_convnext',
        'dab-detr_r50',
        'frcn-dcnv2',
        'yolov8_s',
        'yolov5_s',
    ]
    return name_all