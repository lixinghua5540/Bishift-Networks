import torch.nn as nn
import torch
import util.util as util

from models.shift_net.InnerShiftTripleFunction import InnerShiftTripleFunction


class InnerResShiftTriple(nn.Module):
    def __init__(self, inner_nc, shift_sz=1, stride=1, mask_thred=1, triple_weight=1, layer_to_last=3):
        super(InnerResShiftTriple, self).__init__()

        self.shift_sz = shift_sz
        self.stride = stride
        self.mask_thred = mask_thred
        self.triple_weight = triple_weight
        self.show_flow = False # default false. Do not change it to be true, it is computation-heavy.
        self.flow_srcs = None # Indicating the flow src(pixles in non-masked region that will shift into the masked region)
        self.layer_to_last = layer_to_last

        # Additional for ResShift.
        self.inner_nc = inner_nc
        self.res_net = nn.Sequential(
            nn.Conv2d(inner_nc*2, inner_nc, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(inner_nc),
            nn.ReLU(True),
            nn.Conv2d(inner_nc, inner_nc, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(inner_nc)
        )

    def set_mask(self, mask_global):
        mask = util.cal_feat_mask(mask_global, self.layer_to_last)
        self.mask = mask
        return self.mask

    # If mask changes, then need to set cal_fix_flag true each iteration.
    def forward(self, input):
        #print(input.shape)
        _, self.c, self.h, self.w = input.size()
        self.flag = util.cal_flag_given_mask_thred(self.mask, self.shift_sz, self.stride, self.mask_thred)
        shift_out = InnerShiftTripleFunction.apply(input, self.shift_sz, self.stride, self.triple_weight, self.flag, self.show_flow)

        c_out = shift_out.size(1)
        # get F_c, F_s, F_shift
        F_c = shift_out.narrow(1, 0, c_out//3)
        F_s = shift_out.narrow(1, c_out//3, c_out//3)
        F_shift = shift_out.narrow(1, c_out*2//3, c_out//3)
        F_fuse = F_c * F_shift
        F_com = torch.cat([F_c, F_fuse], dim=1)

        res_out = self.res_net(F_com)
        F_c = F_c + res_out

        final_out = torch.cat([F_c, F_s], dim=1)

        if self.show_flow:
            self.flow_srcs = InnerShiftTripleFunction.get_flow_src()
        return final_out

    def get_flow(self):
        return self.flow_srcs

    def set_flow_true(self):
        self.show_flow = True

    def set_flow_false(self):
        self.show_flow = False

    def __repr__(self):
        return self.__class__.__name__+ '(' \
              + ' ,triple_weight ' + str(self.triple_weight) + ')'
