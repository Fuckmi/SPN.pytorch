import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cPickle
from PIL import Image
from tqdm import tqdm

from spn import object_localization
import experiment.util as utils

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
plt.rcParams["figure.figsize"] = (8,8)

DATA_ROOT = '../data/voc/VOCdevkit/VOC2007'
print "load ground truth ..."
" load ground truth with cache "
if not os.path.isdir('../data/cache'):
    os.makedirs('../data/cache')
gt_cache_file = os.path.join('../data/cache', 'voc2007_trainval_gt.pkl')
if os.path.exists(gt_cache_file):
    with open(gt_cache_file, 'rb') as fid:
        ground_truth = cPickle.load(fid)
else:
    ground_truth = utils.load_ground_truth_voc(DATA_ROOT, 'trainval')
    with open(gt_cache_file, 'wb') as f:
        cPickle.dump(ground_truth, f, protocol=cPickle.HIGHEST_PROTOCOL)

model_path = './logs/voc2007/model.pth.tar'
print "load model ..."
model_dict = utils.load_model_voc(model_path, True)

"""Extract bboxes and evaluate"""
print "start testing ..."
predictions = []
# for img_idx in tqdm(range(len(ground_truth['image_list']))):
#     image_name = os.path.join(DATA_ROOT, 'JPEGImages', ground_truth['image_list'][img_idx] + '.jpg')
#     _, input_var = utils.load_image_voc(image_name)
#     preds, labels = object_localization(model_dict, input_var, location_type='bbox', gt_labels=(ground_truth['gt_labels'][img_idx] == 1).nonzero()[0], nms_threshold=0.7)
#     predictions += [(img_idx,) + p for p in preds]
# print("Corloc: {:.2f}".format(utils.corloc(np.array(predictions), ground_truth) * 100.))

"""visualization"""
image_name = './evaluation/sample3.jpg'
_, input_var = utils.load_image_voc(image_name)
preds, labels = object_localization(model_dict, input_var, location_type='bbox', gt_labels=[ground_truth['class_names'].index(c) for c in ['person', 'horse']])
print 'preds', preds
img = Image.open(image_name)
img_draw = utils.draw_bboxes(img, np.array(preds), ground_truth['class_names'])
img_draw.save('demo_sample3.jpg')

"""problem
even if we just derive from the sf map, objects should not overlap. But now they overlapped
The answer is, for each 

"""

