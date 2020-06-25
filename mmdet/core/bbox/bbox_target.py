import torch
import numpy as np
from ..utils import multi_apply
from .transforms import bbox2delta
class_weight = np.array([1, 0.928, 1, 2, 1.785, 3.74, 0.51, 0.3])


def bbox_target(pos_bboxes_list,
                neg_bboxes_list,
                pos_gt_bboxes_list,
                pos_gt_labels_list,
                cfg,
                reg_classes=1,
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
                concat=True):
    labels, label_weights, bbox_targets, bbox_weights, pos_boxes = multi_apply(
        bbox_target_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_gt_bboxes_list,
        pos_gt_labels_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
        pos_boxes = torch.cat(pos_boxes,  0)
    return labels, label_weights, bbox_targets, bbox_weights, pos_boxes


def bbox_target_single(pos_bboxes,
                       neg_bboxes,
                       pos_gt_bboxes,
                       pos_gt_labels,
                       cfg,
                       reg_classes=1,
                       target_means=[.0, .0, .0, .0],
                       target_stds=[1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    bbox_weights = pos_bboxes.new_zeros(num_samples)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
#         weight = class_weight[pos_gt_labels.cpu().data.numpy()]
#         weight = torch.Tensor(weight).cuda()
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
#         label_weights[:num_pos] = weight
        label_weights[:num_pos] = pos_weight
        # pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means,
        #                               target_stds)
        # bbox_targets[:num_pos, :] = pos_bbox_targets
#         weight_bbox = class_weight[pos_gt_labels.cpu().data.numpy()]
#         weight_bbox = torch.Tensor(weight_bbox).cuda()
        bbox_targets[:num_pos, :] = pos_gt_bboxes
        bbox_weights[:num_pos] = 1.0
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0
    return labels, label_weights, bbox_targets, bbox_weights, pos_bboxes

def bbox_target_IoU(pos_bboxes_list,
                    neg_bboxes_list,
                    pos_gt_bboxes_list,
                    pos_gt_labels_list,
                    pos_gt_ious_list,
                    neg_gt_ious_list,
                    cfg,
                    reg_classes=1,
                    target_means=[.0, .0, .0, .0],
                    target_stds=[1.0, 1.0, 1.0, 1.0],
                    concat=True):
    labels, label_weights, bbox_targets, bbox_weights, IoU_targets, IoU_weights = multi_apply(
        bbox_target_single_IoU,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_gt_bboxes_list,
        pos_gt_labels_list,
        pos_gt_ious_list,
        neg_gt_ious_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
        IoU_targets = torch.cat(IoU_targets, 0)
        IoU_weights = torch.cat(IoU_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights, IoU_targets, IoU_weights


def bbox_target_single_IoU(pos_bboxes,
                           neg_bboxes,
                           pos_gt_bboxes,
                           pos_gt_labels,
                           pos_gt_iou,
                           neg_gt_iou,
                           cfg,
                           reg_classes=1,
                           target_means=[.0, .0, .0, .0],
                           target_stds=[1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
    IoU_targets = pos_bboxes.new_zeros(num_samples)
    IoU_weights = pos_bboxes.new_zeros(num_samples)

    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        IoU_targets[:num_pos] = pos_gt_iou
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        IoU_weights[:num_pos] = pos_weight
        pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means,
                                      target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0
        IoU_targets[-num_neg:] = neg_gt_iou

    return labels, label_weights, bbox_targets, bbox_weights, IoU_targets, IoU_weights


def expand_target(bbox_targets, bbox_weights, labels, num_classes):
    bbox_targets_expand = bbox_targets.new_zeros(
        (bbox_targets.size(0), 4 * num_classes))
    bbox_weights_expand = bbox_weights.new_zeros(
        (bbox_weights.size(0), 4 * num_classes))
    for i in torch.nonzero(labels > 0).squeeze(-1):
        start, end = labels[i] * 4, (labels[i] + 1) * 4
        bbox_targets_expand[i, start:end] = bbox_targets[i, :]
        bbox_weights_expand[i, start:end] = bbox_weights[i, :]
    return bbox_targets_expand, bbox_weights_expand

def intersect(box_a, box_b):

    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].expand(A,  2),
                       box_b[:, 2:].expand(B,  2))
    min_xy = torch.max(box_a[:, :2].expand(A,  2),
                       box_b[:, :2].expand(B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]


def jaccard(box_a, box_b):

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]