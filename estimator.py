import os

import torch
from torch.autograd import Variable

from .lib.network import PoseNet, PoseRefineNet
from .lib.loss import Loss
from .lib.loss_refiner import Loss_refine


_PREFIX = os.path.dirname(os.path.realpath(__file__))

class DenseFusionEstimator:
    """
    Warning: currently specialized for Linemod only.
    """

    def __init__(self, model_prefix=_PREFIX + "/trained_checkpoints/linemod"):

        # Linemod Dataset info
        self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.sym_list = [7, 8]
        self.num_objects = len(self.objlist)
        self.num_points = 500
        self.iteration = 2

        # Initialize estimator
        self.estimator = PoseNet(num_points = self.num_points, num_obj = self.num_objects)
        self.estimator.cuda()
        self.refiner = PoseRefineNet(num_points = self.num_points, num_obj = self.num_objects)
        self.refiner.cuda()

        # Load models
        model = os.path.join(model_prefix, "pose_model_9_0.01310166542980859.pth")
        refine_model = os.path.join(model_prefix, "pose_refine_model_493_0.006761023565178073.pth")
        self.estimator.load_state_dict(torch.load(model))
        self.refiner.load_state_dict(torch.load(refine_model))
        self.estimator.eval()
        self.refiner.eval()

        # Initialize criterion
        self.criterion = Loss(self.num_points, self.sym_list)
        self.criterion_refine = Loss_refine(self.num_points, self.sym_list)

    # def estimate(self, points, choose, img, target, model_points, idx):
    def pose(self, points, choose, img, model_points, idx):

        points_ = Variable(points).cuda()
        choose_ = Variable(choose).cuda()
        img_ = Variable(img).cuda()
        target_ = Variable(torch.zeros(model_points.size())).cuda()
        model_points_ = Variable(model_points).cuda()
        idx_ = Variable(idx).cuda()

        pred_r, pred_t, pred_c, emb = self.estimator(img_, points_, choose_, idx_)
        _, _, new_points, new_target = self.criterion(pred_r, pred_t, pred_c, target_, model_points_, idx_, points_, 0.0, False)

        for ite in range(0, self.iteration):
            pred_r, pred_t = self.refiner(new_points, emb, idx_)
            _, new_points, new_target = self.criterion_refine(pred_r, pred_t, new_target, model_points_, idx_, new_points)

        return pred_r, pred_t, pred_c
