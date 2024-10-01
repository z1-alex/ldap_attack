from mmdet.structures.bbox import bbox_overlaps

def get_success_flag(bbox_pred, label_pred, gt_bbox, gt_label, iou_thr=0.5, score_thr=0.3):
    iou = bbox_overlaps(bbox_pred[:,:4], gt_bbox[None])[:,0]

    label_mask = (label_pred == gt_label)
    iou_mask = (iou>iou_thr)
    score_mask = (bbox_pred[:,4] > score_thr)
    detect_mask = label_mask * iou_mask * score_mask
    detect_success = detect_mask.sum() > 0
    return detect_success

def get_success_flag_and_bbox(bbox_pred, label_pred, gt_bbox, gt_label, iou_thr=0.5, score_thr=0.3):
    iou = bbox_overlaps(bbox_pred[:,:4], gt_bbox[None])[:,0]

    label_mask = (label_pred == gt_label)
    iou_mask = (iou>iou_thr)
    score_mask = (bbox_pred[:,4] > score_thr)
    detect_mask = label_mask * iou_mask * score_mask
    detect_success = detect_mask.sum() > 0
    detect_success_bbox = bbox_pred[detect_mask]
    return detect_success, detect_success_bbox