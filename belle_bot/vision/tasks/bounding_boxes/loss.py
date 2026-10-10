import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_iou(box1, box2, eps=1e-7):
    """
    Computes IoU between two sets of bounding boxes [cx, cy, w, h].
    """
    b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
    b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
    b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
    b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    w1, h1 = box1[..., 2], box1[..., 3]
    w2, h2 = box2[..., 2], box2[..., 3]
    union_area = (w1 * h1) + (w2 * h2) - inter_area + eps

    return inter_area / union_area


class YOLODistillationLoss(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_box=2.0, lambda_obj=1.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj

        self.bce_logits = nn.BCEWithLogitsLoss(reduction="none")
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

    def forward(self, pred_student: torch.Tensor, target_teacher: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        pred_student:   [Batch, Num_Boxes, 4 + 1 + Num_Classes] -> [cx, cy, w, h, obj_logit, class_logits...]
        target_teacher: [Batch, Num_Boxes, 6]                  -> [cx, cy, w, h, obj_logit, class_id]
        """
        # 1. Sort candidate boxes by teacher confidence descending to align predictions with targets
        teacher_obj_logits = target_teacher[..., 4]
        sorted_indices = torch.argsort(teacher_obj_logits, dim=-1, descending=True)
        batch_indices = torch.arange(pred_student.size(0), device=pred_student.device).unsqueeze(-1)

        aligned_student = pred_student[batch_indices, sorted_indices]
        aligned_teacher = target_teacher[batch_indices, sorted_indices]

        # 2. Unpack aligned components
        student_boxes = aligned_student[..., :4]
        student_obj_logits = aligned_student[..., 4]
        student_cls_logits = aligned_student[..., 5:]  # Shape: [Batch, Num_Boxes, Num_Classes]

        teacher_boxes = aligned_teacher[..., :4]
        teacher_obj_logits = aligned_teacher[..., 4]
        teacher_class_ids = aligned_teacher[..., 5].long()  # Shape: [Batch, Num_Boxes]

        # Teacher confidence score mask for weighting
        teacher_obj_prob = torch.sigmoid(teacher_obj_logits)

        # 3. Objectness Loss
        loss_obj = self.bce_logits(student_obj_logits, teacher_obj_prob).mean()

        # 4. Bounding Box Loss (Weighted by teacher objectness)
        iou = bbox_iou(student_boxes, teacher_boxes)
        loss_iou = 1.0 - iou
        loss_coord = self.smooth_l1(student_boxes, teacher_boxes).sum(dim=-1)
        loss_box = ((loss_iou + loss_coord) * teacher_obj_prob).sum() / (teacher_obj_prob.sum() + 1e-7)

        # 5. One-Hot Classification Loss (Weighted by teacher objectness)
        num_classes = student_cls_logits.shape[-1]

        # Create a mask for valid classes (ignoring background/invalid IDs like -1)
        valid_class_mask = (teacher_class_ids >= 0) & (
                teacher_class_ids < num_classes
        )
        # Convert scalar class indices [Batch, Num_Boxes] -> One-Hot [Batch, Num_Boxes, Num_Classes]
        clamped_class_ids = teacher_class_ids.clamp(min=0, max=num_classes - 1)
        teacher_cls_one_hot = F.one_hot(clamped_class_ids, num_classes=num_classes).float()

        # Compute element-wise BCE across classes
        bce_cls = self.bce_logits(student_cls_logits, teacher_cls_one_hot)

        # Mask class loss by teacher confidence so background box class IDs don't contaminate training
        teacher_obj_mask = teacher_obj_prob.unsqueeze(-1)  # Shape: [Batch, Num_Boxes, 1]
        loss_cls = (bce_cls * teacher_obj_mask).sum() / (teacher_obj_mask.sum() * num_classes + 1e-7)

        # 6. Total Combined Loss
        total_loss = (
                self.lambda_cls * loss_cls +
                self.lambda_box * loss_box +
                self.lambda_obj * loss_obj
        )

        return {
            "loss": total_loss,
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss_obj": loss_obj,
        }
