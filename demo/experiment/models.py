import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from spn.modules import SoftProposal, SpatialSumOverMap
from spn.utils import localize_with_map
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '..', '..'))
from rcnnlib.model.roi_align.modules.roi_align import RoIAlignAvg
class SPNetWSL(nn.Module):
    """
    This module will merge the previous step into one step
    """
    def __init__(self, model, num_classes, num_maps, pooling):
        super(SPNetWSL, self).__init__()

        self.features = nn.Sequential(*list(model.features.children())[:-1])
        self.spatial_pooling = pooling

        # classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_maps, num_classes)
        )

        # image normalization
        self.image_normalization_mean = [103.939, 116.779, 123.68]

    def forward(self, x): 
        x = self.features(x)
        x = self.spatial_pooling(x)
        x = x.view(x.size(0), -1)
        # so a sum over spatial and send to classifier
        x = self.classifier(x)
        # x (batch, num_classes)
        return x

class Feat_offset(nn.Module):
    """get the offset and offsetted feat

    :param feat: image feature, (batch, c, h, w)
    :param dets: bboxes (batch, n_bbox, 5), (x1, y1, x2, y2, s)
    :param img_size: (h, w)
    :return: offset (batch, n_bbox, 4)
    :return: offseted feat
    """
    def __init__(self, bbox_num, featsize, img_size):
        super(Feat_offset, self).__init__()
        self.batch_num, self.c, self.h, self.w = featsize
        self.roi_align = RoIAlignAvg(7, 7, 1.0/16.0)
        self.bbox_num = bbox_num
        self.batch_ind = torch.arange(self.batch_num).repeat(self.bbox_num, 1).view(-1)
        self.fc_offset = nn.Linear(self.c, 4)
        self.img_h, self.img_w = img_size
        self.avg_pool = nn.AvgPool2d(self.h, self.w)

    def forward(self, feat, dets):
        """
        [warning]: only support batch size 1!
        because the spn cannot simultaneously support img and box level interpolate

        :param feat: (batch, c, h, w)
        :param dets: (batch, box_num, 5) (x1, y1, x2, y2, score)
        :return: offset_feats: (batch, box_num, c)
        :return: offset: (batch, box_num, 4)
        """
        # see how to crop feature and predict offset
        # assume the number of proposals in each image is fixed.
        # transfer dets to shape (all_num, 5) (batch_ind, x1, y1, x2, y2)
        # get box-level feat
        rois = dets.view(-1, 5)
        # rois = Variable(torch.zeros_like(rois_temp), requires_grad=True)
        rois.data[:, 1:] = rois.data[:, :4]
        rois.data[:, 0] = self.batch_ind
        # bbox_feats (batch_num*bbox_num, c)
        bbox_feats = self.roi_align(feat, rois)
        bbox_feats = torch.mean(bbox_feats.view(self.batch_num*self.bbox_num, self.c, -1), dim=-1)
        # offsets (batch_num*bbox_num, 4) [tx, ty, tw, th]
        offsets = self.fc_offset(bbox_feats)
        tx = offsets[:, 0]
        ty = offsets[:, 1]
        tw = offsets[:, 2]
        th = offsets[:, 3]
        # interpolate the offsets by feature
        dets = dets.view(-1, 5)
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        scores.data = scores.data.contiguous()
        xa = (x1 + x2)/2.0
        ya = (y1 + y2)/2.0
        wa = x2 - x1
        ha = y2 - y1
        xar = xa + tx*wa
        yar = ya + ty*ha
        war = wa*torch.exp(tw)
        har = ha*torch.exp(th)
        # get new bbox axis
        x1r = xar - war/2
        y1r = yar - har/2
        x2r = xar + war/2
        y2r = yar + har/2
        # get offset dets
        offset_dets = torch.cat((x1r.view(-1, 1), y1r.view(-1, 1), x2r.view(-1, 1), y2r.view(-1, 1), scores.view(-1, 1)), dim=1)
        offset_dets = offset_dets.view(self.batch_num, -1, 5)
        zeros1 = Variable(torch.zeros(self.bbox_num*self.batch_num).cuda(), requires_grad=False)
        zeros3 = Variable(torch.zeros(self.bbox_num*self.batch_num).cuda(), requires_grad=False)
        affine_list = [war/self.img_w, zeros1, 2*xar/self.img_w - 1, zeros3, har/self.img_h, 2*yar/self.img_h - 1]
        affine_mat = torch.stack(affine_list, dim=1)
        affine_mat = affine_mat.view(-1, 2, 3)
        feat_rep = feat.repeat(self.bbox_num, 1, 1, 1)
        # get functional grid
        grid = F.affine_grid(affine_mat, feat_rep.size())
        # offset_feat (batch*box_num, c, h, w)
        offset_feats = F.grid_sample(feat_rep, grid)
        offset_feats = self.avg_pool(offset_feats)
        offset_feats = torch.squeeze(offset_feats)
        offset_feats = offset_feats.view(self.batch_num, self.bbox_num, -1)
        return offset_feats, offset_dets

class WSLDetPred(nn.Module):
    def __init__(self, bbox_num, cls_num, feat_size):
        super(WSLDetPred, self).__init__()
        self.batch_num, self.c, self.h, self.w = feat_size
        self.bbox_num = bbox_num
        self.cls_num = cls_num
        self.fc1 = nn.Linear(self.c, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc_cls = nn.Linear(1024, self.cls_num)
        self.fc_box = nn.Linear(1024, self.cls_num)


    def forward(self, feats):
        """
        :param feats: (batch, box, c)
        :return: pred: (batch, class_num)
        """
        feats = feats.view(-1, self.c)
        feats = F.relu(self.fc1(feats))
        feats = F.relu(self.fc2(feats))
        feat_clses = F.relu(self.fc_cls(feats))
        feat_clses = feat_clses.view(self.batch_num, self.bbox_num, -1)
        feat_boxes = F.relu(self.fc_box(feats))
        feat_boxes = feat_boxes.view(self.batch_num, self.bbox_num, -1)
        clses_score = F.softmax(feat_clses, dim=1)
        boxes_score = F.softmax(feat_boxes, dim=1)
        pred = torch.sum(clses_score * boxes_score, dim=1)
        print 'bbox num', self.bbox_num
        print 'pred shape', pred.size()
        print 'pred', pred
        return pred


def _sp_hook(self, input, output):
    self.parent_modules[0].class_response_maps = output

class WSLDetPipe(nn.Module):
    def __init__(self, pretrained, num_classes, num_maps, num_boxes, img_size, feat_size):
        super(WSLDetPipe, self).__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        model = models.vgg16(pretrained=False)
        if pretrained:
            model_path = '/home/jshi31/SPN.pytorch/demo/models/vgg16_official.pth'
            # model_path = './models/vgg16_official.pth'
            print 'syspath', sys.path
            print 'env', os.path.isfile(model_path)
            if os.path.isfile(model_path):
                print 'loading pretrained model...'
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
            else:
                print('Please download the pretrained VGG16 into ./models')
        # model.feature[28] is the last conv layer in VGG16
        num_features = model.features[28].out_channels
        # nn.Sequntial is the success of module class. So add_module is the method in module
        # pooling : conv->relu->sp->sum
        """ Whether should I pool?"""
        pooling = nn.Sequential()
        pooling.add_module('adconv', nn.Conv2d(num_features, num_maps, kernel_size=3, stride=1, padding=1, groups=2, bias=True))
        pooling.add_module('maps', nn.ReLU())
        sp_layer = SoftProposal()

        model.sp_hook = sp_layer.register_forward_hook(_sp_hook)
        pooling.add_module('sp', sp_layer)

        self.summing = nn.Sequential(SpatialSumOverMap())

        # the output shape of sp is batch, num_maps, 7, 7, I guess it is coupled with the conv layer, so it keeps shape
        self.features = nn.Sequential(*list(model.features.children())[:-1])
        self.spatial_pooling = pooling
        # classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_maps, num_classes)
        )
        # image normalization
        self.image_normalization_mean = [103.939, 116.779, 123.68]
        """at last you need to return the class level prediction, but do not know how to output it."""
        self.feat_offset = Feat_offset(num_boxes, feat_size, img_size[2:])
        self.WSLdetpred = nn.Sequential(WSLDetPred(num_boxes, num_classes, feat_size))
        self.batch_num, _, self.h, self.w = img_size
        """write the hook function, that can hook the variable spn"""
        self.hook_spn()
        self.class_response_maps = torch.zeros([self.batch_num, self.num_classes, feat_size[2], feat_size[3]])

    def hook_spn(self):
        if not (hasattr(self, 'sp_hook') and hasattr(self, 'fc_hook')):
            def _sp_hook(self, input, output):
                self.parent_modules[0].class_response_maps = output
            def _fc_hook(self, input, output):
                if hasattr(self.parent_modules[0], 'class_response_maps'):
                    self.parent_modules[0].class_response_maps = F.conv2d(self.parent_modules[0].class_response_maps, self.weight.unsqueeze(-1).unsqueeze(-1))
                else:
                    raise RuntimeError('The SPN is broken, please recreate it.')

            sp_layer = None
            for mod in self.modules():
                if isinstance(mod, SoftProposal):
                    sp_layer = mod
            fc_layer = self.classifier[1]

            if sp_layer is None or fc_layer is None:
                raise RuntimeError('Invalid SPN model')
            else:
                sp_layer.parent_modules = [self]
                fc_layer.parent_modules = [self]
                self.sp_hook = sp_layer.register_forward_hook(_sp_hook)
                self.fc_hook = fc_layer.register_forward_hook(_fc_hook)

    def forward(self, x):
        feat = self.features(x)
        feat = self.spatial_pooling(feat)
        # img level
        x = self.summing(feat)
        x = x.view(x.size(0), -1)
        # so a sum over spatial and send to classifier
        img_score = self.classifier(x)

        # box level dets (batch, n, 6)
        dets = localize_with_map(F.upsample(self.class_response_maps, size=(self.h, self.w), mode='bilinear'), 0.7, self.num_boxes, 1)
        det_var = Variable(torch.from_numpy(dets[:, :, 1:]).type(torch.FloatTensor).cuda(), requires_grad=False)
        det_var.data = det_var.data.contiguous()
        offset_feats, offset_dets = self.feat_offset(feat, det_var)
        box_score = self.WSLdetpred(offset_feats)

        # return box_score, img_score
        return 0.5*(img_score + box_score)


def vgg16_sp(num_classes, pretrained=True, num_maps=1024):
    model = models.vgg16(pretrained=False)
    print model
    if pretrained:
        model_path = './models/vgg16_official.pth'
        if os.path.isfile(model_path):
            print 'loading pretrained model...'
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
        else:
            print('Please download the pretrained VGG16 into ./models')
    # model.feature[28] is the last conv layer in VGG16
    num_features = model.features[28].out_channels
    # nn.Sequntial is the success of module class. So add_module is the method in module
    # pooling : conv->relu->sp->sum
    pooling = nn.Sequential()
    pooling.add_module('adconv', nn.Conv2d(num_features, num_maps, kernel_size=3, stride=1, padding=1, groups=2, bias=True))
    pooling.add_module('maps', nn.ReLU())
    pooling.add_module('sp', SoftProposal())
    # the output shape of sp is batch, num_maps, 7, 7, I guess it is coupled with the conv layer, so it keeps shape
    pooling.add_module('sum', SpatialSumOverMap())
    # output shape of sum (batch, num_maps)
    # model: VGG16, the last conv5 layer.
    # num_classes: the number of output class
    # num_maps: output channel
    # pooling: the tail module, containing the sp module and sum function.
    return SPNetWSL(model, num_classes, num_maps, pooling)


