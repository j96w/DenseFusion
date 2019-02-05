import os

import torch
from torch.autograd import Variable

from .datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from .lib.network import PoseNet, PoseRefineNet
from .lib.loss import Loss
from .lib.loss_refiner import Loss_refine


_PREFIX = os.path.dirname(os.path.abspath(__file__))

class DenseFusionEstimator:

    def __init__(
        self,
        dataset=_PREFIX + "/datasets/linemod/Linemod_preprocessed",
        model=_PREFIX + "/trained_checkpoints/linemod/pose_model_9_0.01310166542980859.pth",
        refine_model=_PREFIX + "/trained_checkpoints/linemod/pose_refine_model_493_0.006761023565178073.pth"
    ):


        self.num_objects = 13
        self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.num_points = 500
        self.iteration = 2

        # Initialize estimator
        self.estimator = PoseNet(num_points = self.num_points, num_obj = self.num_objects)
        self.estimator.cuda()
        self.refiner = PoseRefineNet(num_points = self.num_points, num_obj = self.num_objects)
        self.refiner.cuda()
        self.estimator.load_state_dict(torch.load(model))
        self.refiner.load_state_dict(torch.load(refine_model))
        self.estimator.eval()
        self.refiner.eval()

        # Initialize criterion
        self.testdataset = PoseDataset_linemod('eval', self.num_points, False, dataset, 0.0, True)
        self.sym_list = self.testdataset.get_sym_list()
        self.num_points_mesh = self.testdataset.get_num_points_mesh()
        self.criterion = Loss(self.num_points_mesh, self.sym_list)
        self.criterion_refine = Loss_refine(self.num_points_mesh, self.sym_list)

    # def estimate(self, points, choose, img, target, model_points, idx):
    def estimate(self, points, choose, img, model_points, idx):

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
