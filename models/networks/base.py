import torch.nn as nn
import torch


class BaseNet(nn.Module):

    def __init__(self, **args):
        """
        args: easydict, parameters
        """
        super(BaseNet, self).__init__()


    def _set_args(self, args):
        for key, value in args.items():
            if key != 'self' and key != '__class__':
                setattr(self, key, value)


    def load_pretrain_model(self, model_path):
        if model_path is not None and model_path != "":
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))

            state_dict_tmp = self.state_dict()
            key_tmp = set(list(state_dict_tmp.keys()))

            for n, p in state_dict.items():

                if n in key_tmp:
                    key_tmp.remove(n)
                else:
                    print('%s not exist, pass!' % n)
                
                pretrain_weight = p.data
                if state_dict_tmp[n].shape != pretrain_weight.shape:
                    print("%s size mismatch, loading selected kernel!" % n)
                    pretrain_weight = self.select_pretrain_kernel(pretrain_weight, state_dict_tmp[n].data)
                
                state_dict_tmp[n].copy_(pretrain_weight)
            
            if len(key_tmp) != 0:
                for k in key_tmp:
                    print("param %s not found in pretrain model!" % k)

            self.load_state_dict(state_dict)
            print("Load checkpoint {} successfully!".format(model_path))


    def select_pretrain_kernel(self, pertrain_weight, cur_weight):
        """
        如果pretrain_weight大于cur_weight，则从pretrain_weight中选择一些channel加载到cur_weight中
        """
        assert len(pertrain_weight.shape) == len(cur_weight.shape)
        if len(pertrain_weight.shape) == 4:  # conv weight
            from_oc, from_ic, from_h, from_w = pertrain_weight.size()
            to_oc, to_ic, to_h, to_w = cur_weight.size()
            assert from_oc >= to_oc and from_ic >= to_ic
            return pertrain_weight[0:to_oc, 0:to_ic, :, :]
        elif len(pertrain_weight.shape) == 2:  # fc weight
            from_oc, from_ic = pertrain_weight.size()
            to_oc, to_ic = cur_weight.size()
            assert from_oc >= to_oc and from_ic >= to_ic
            return pertrain_weight[0:to_oc, 0:to_ic]
        elif len(pertrain_weight.shape) == 1:   # bias
            from_oc = pertrain_weight.size()[0]
            to_oc = cur_weight.size()[0]
            assert from_oc >= to_oc
            return pertrain_weight[0:to_oc]
        else:
            raise NotImplementedError("Unsupported weight shape.")

