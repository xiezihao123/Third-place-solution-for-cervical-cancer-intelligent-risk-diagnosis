from .bbox_head import BBoxHead
from .convfc_bbox_head import ConvFCBBoxHead, SharedFCBBoxHead, IoU_head, reg_head
from .double_bbox_head import DoubleConvFCBBoxHead

__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'SharedFCBBoxHead', 'DoubleConvFCBBoxHead', 'IoU_head', 'reg_head'
]
