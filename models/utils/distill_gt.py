import yaml
from easydict import EasyDict as edict
import torch

from models.networks import build_network
from .warp import WarpModel



class DistillGT:
    """
    用一个大模型对GT进行修改
    """

    def __init__(self, config_file, model_path, warp="warpnet", type="model", alpha_in=0.5, alpha_gt=0.5) -> None:
        """
        warp: warpnet | flow | no_warp
        type: model | real | mix_in | mix_gt
        alpha_in: used when type is 'mix_in'
        alpha_gt: used when type is 'mix_gt'
        """
        self.type = type
        self.warp = warp
        self.warp_model = WarpModel(warp_type=self.warp)
        self.alpha_in = alpha_in
        self.alpha_gt = alpha_gt

        with open(config_file, 'r') as f:
            config = edict(yaml.load(f, Loader=yaml.SafeLoader))
        self.net = build_network(config.RunnerConfig.ModelConfig.Network.GNet)
        self.net.load_pretrain_model(model_path)
        self.net.eval()

    def wrap(self, parallel):
        self.warp_model.wrap(parallel)
        self.net = parallel.wrapper(self.net)

    @torch.no_grad()
    def __call__(self, d_img_y, ref_img_y, gt_img_y, scores, d_img_rgb=None, ref_img_rgb=None):
        """
        if warp_type is warpnet:
            d_img_y: torch.Tensor, [NCHW]
            ref_img_y: torch.Tensor, [NCHW]
            gt_img_y: torch.Tensor, [NCHW]
            scores: torch.Tensor, [NCHW]
            d_img_rgb, ref_img_rgb: warp使用warpnet的时候需要用到
        
        """

        gt_distill = []
        for i in range(d_img_y.shape[0]):
            if scores[i][0] > 0.55 or scores[i][1] > 0.45 or scores[i][2] > 0.65 or scores[i][3] > 0.65:
                d_y = d_img_y[i].unsqueeze(0)
                ref_y = ref_img_y[i].unsqueeze(0)
                gt_y = gt_img_y[i].unsqueeze(0)
                warp_img = self.warp_model(d_y, ref_y, d_img_rgb, ref_img_rgb)
                if self.type == "model":
                    rec_img = self.net(d_y, warp_img)
                    gt_distill.append(rec_img)
                elif self.type == "real":
                    if scores[i][2] > 0.65 or scores[i][3] > 0.65:
                        gt_distill.append(d_y)
                    else:
                        rec_img = self.net(d_y, warp_img)
                        gt_distill.append(rec_img)
                elif self.type == "mix_in":
                    rec_img = self.net(d_y, warp_img)
                    rec_img = (1 - self.alpha_in) * rec_img + self.alpha_in * d_y
                    gt_distill.append(rec_img)
                elif self.type == "mix_gt":
                    rec_img = self.net(d_y, warp_img)
                    rec_img = (1 - self.alpha_gt) * rec_img + self.alpha_gt * gt_y
                    gt_distill.append(rec_img)
                else:
                    raise NotImplementedError("distill type: %s is not supported." % self.type)
            else:
                gt_distill.append(gt_y)
        
        gt_distill = torch.cat(gt_distill, dim=0)

        return gt_distill



