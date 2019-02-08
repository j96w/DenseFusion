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

    def __init__(self, model_prefix=os.path.join(_PREFIX, "trained_checkpoints", "linemod")):

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

    @staticmethod
    def _quatm(q1, q0):
        """Return multiplication of two quaternions.

        >>> q = _quatm([4, 1, -2, 3], [8, -5, 6, 7])
        >>> numpy.allclose(q, [28, -44, -14, 48])
        True

        """
        w0, x0, y0, z0 = q0
        w1, x1, y1, z1 = q1
        return torch.Tensor([
            -x1*x0 - y1*y0 - z1*z0 + w1*w0,
            x1*w0 + y1*z0 - z1*y0 + w1*x0,
            -x1*z0 + y1*w0 + z1*x0 + w1*y0,
            x1*y0 - y1*x0 + z1*w0 + w1*z0])

    @staticmethod
    def _update_pose(pose_original, update):

        q, t = pose_original
        qu, tu = update

        quc = torch.Tensor([qu[0], -qu[1], -qu[2], -qu[3]])
        q_final = self._quatm(qu, self._quatm(q, quc))

        tq = torch.Tensor([0, t[0], t[1], t[2]])
        tuq = torch.Tensor([0, tu[0], tu[1], tu[2]])
        t_final = self._quatm(qu, self._quatm(tq, quc)) + tuq

        return q_final, t_final[1:]


    # def estimate(self, points, choose, img, target, model_points, idx):
    def pose(self, points, choose, img, model_points, idx):

        points_ = Variable(points).cuda()
        choose_ = Variable(choose).cuda()
        img_ = Variable(img).cuda()
        target_ = Variable(torch.zeros(model_points.size())).cuda()
        model_points_ = Variable(model_points).cuda()
        idx_ = Variable(idx).cuda()

        pred_r, pred_t, pred_c, emb = self.estimator(img_, points_, choose_, idx_)
        _, _, new_points, new_target, max_idx = self.criterion(pred_r, pred_t, pred_c, target_, model_points_, idx_, points_, 0.0, False)

        # flatten things
        pred_r = pred_r[0, max_idx]
        pred_t = pred_t[max_idx, 0]

        for _ in range(self.iteration):
            pred_r_inc, pred_t_inc = self.refiner(new_points, emb, idx_)
            # TODO Needs to be validated
            pred_r, pred_t = self._update_pose((pred_r, pred_t), (pred_r_inc, pred_t_inc))
            _, new_points, new_target = self.criterion_refine(pred_r_inc, pred_t_inc, new_target, model_points_, idx_, new_points)

        return pred_r, pred_t, pred_c[0, max_idx]
