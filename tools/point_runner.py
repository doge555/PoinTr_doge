##############################################################
# % Author: Jianlin Dou
# % Date:1/11/2025
###############################################################
import os
import numpy as np
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from tools import builder
from utils.config import cfg_from_yaml_file
from utils.point_processing import preprocess_for_pointtr, postprocessing_from_pointr
from datasets.data_transforms import Compose


def inference_single(model, pcd, device, config):
    # preprocess input point cloud
    point_partial, centroid_orginal, scale_orginal = preprocess_for_pointtr(pcd)
    # read refined single point cloud
    pc_ndarray = np.asarray(point_partial.points, dtype=np.float32)
    # transform it according to the model 
    if config.dataset.train._base_['NAME'] == 'ShapeNet':
        # normalize it to fit the model on ShapeNet-55/34
        centroid = np.mean(pc_ndarray, axis=0)
        pc_ndarray = pc_ndarray - centroid
        m = np.max(np.sqrt(np.sum(pc_ndarray**2, axis=1)))
        pc_ndarray = pc_ndarray / m

    transform = Compose([{
        'callback': 'UpSamplePoints',
        'parameters': {
            'n_points': 2048
        },
        'objects': ['input']
    }, {
        'callback': 'ToTensor',
        'objects': ['input']
    }])
    
    pc_ndarray_normalized = transform({'input': pc_ndarray})
    # inference
    ret = model(pc_ndarray_normalized['input'].unsqueeze(0).to(device.lower()))
    dense_points = ret[-1].squeeze(0).detach().cpu().numpy()

    if config.dataset.train._base_['NAME'] == 'ShapeNet':
        # denormalize it to adapt for the original input
        dense_points = dense_points * m
        dense_points = dense_points + centroid
    
    # postprocess the output point cloud
    point_complete = postprocessing_from_pointr(dense_points, centroid_orginal, scale_orginal)
    
    return point_complete

def point_pip_runner(pcd, model_config = "cfgs/PCN_models/PoinTr.yaml", model_checkpoint = "ckpts/PoinTr_PCN_FT.pth", device='cuda:0'):
    # init config
    config = cfg_from_yaml_file(model_config)
    # build model
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, model_checkpoint)
    base_model.to(device.lower())
    base_model.eval()

    inference_single(base_model, pcd, device, config)