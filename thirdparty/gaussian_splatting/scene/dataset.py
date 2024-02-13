from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov,getProjectionMatrix,getWorld2View2
from kornia import create_meshgrid
from helper_model import pix2ndc
from PIL import Image
import glob
import cv2
from torchvision import transforms as T
import os
from typing import NamedTuple

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args
    ):
        self.dataset = dataset
        self.args = args
    def __getitem__(self, index):
        caminfo = self.dataset[index]
        image = caminfo.image
        image = PILtoTorch(image,None)

        cameradirect = caminfo.hpdirecitons
        camerapose = caminfo.pose 
        
        loaded_mask = None
        
        if camerapose is not None:
            rays_o, rays_d = 1, cameradirect # TODO always True
        else :
            rays_o = None
            rays_d = None
        return Camera(colmap_id=caminfo.uid, R=caminfo.R, T=caminfo.T, 
                FoVx=caminfo.FovX, FoVy=caminfo.FovY, 
                image=image, gt_alpha_mask=loaded_mask,
                image_name=caminfo.image_name, uid=id, data_device=torch.device("cuda"), #있던거
                near=caminfo.near, far=caminfo.far, 
                timestamp=caminfo.timestamp, 
                rayo=rays_o, rayd=rays_d,cxr=caminfo.cxr,cyr=caminfo.cyr)
    def __len__(self):
        
        return len(self.dataset)

class COLMAP_Dataset(Dataset):#follow scene.neural_3D-dataset_NDC.Neural3D_NDC_Dataset
    def __init__(
            self,
            cam_extrinsics, 
            cam_intrinsics, 
            images_folder, #colmap_0/images
            near, 
            far, 
            startime=0, 
            duration=50, # follow scene.dataset_readers.readColmapCameras
            split="train"
    ):
        self.cam_extrinsics = cam_extrinsics        
        self.cam_intrinsics = cam_intrinsics        
        self.start_time = startime
        self.duration = duration

        originnumpy = os.path.join(os.path.dirname(os.path.dirname(images_folder)), "poses_bounds.npy")
        with open(originnumpy, 'rb') as numpy_file:
            poses_bounds = np.load(numpy_file)
            bounds = poses_bounds[:, -2:]
            self.near = bounds.min() * 0.95
            self.far = bounds.max() * 1.05
            

        self.images_paths = []
        self.camera_params = []
        for key in sorted(cam_extrinsics.keys()):
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            height = intr.height
            width = intr.width
            uid = intr.id
            Rotation = np.transpose(qvec2rotmat(extr.qvec))
            Translation = np.array(extr.tvec)
            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            self.camera_params.append([uid,Rotation,Translation,FovY,FovX,width,height])

            for j in range(startime, startime+ int(duration)):
                image_path = os.path.join(images_folder, os.path.basename(extr.name))
                image_path = image_path.replace("colmap_"+str(startime), "colmap_{}".format(j), 1)
                assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
                self.images_paths.append(image_path)
        
        if split == "train":
            self.images_paths = self.images_paths[duration:]
        elif split == "test":
            self.images_paths = self.images_paths[:duration]
        else:
            assert False, "split only train or test"

    def __len__(self):
        return len(self.images_paths)
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        image_name = os.path.basename(image_path).split(".")[0]
        frame_num = int(image_path.split('/')[-3].split('_')[-1])
        cam_num = int(image_name[3:])

        image = Image.open(self.images_paths[index])
        uid,Rotation,Translation,FovY,FovX,width,height = self.camera_params[cam_num]

        if True:#frame_num == self.start_time:
            cam_info = CameraInfo(uid=uid, R=Rotation, T=Translation, FovY=FovY, FovX=FovX, image=image, image_path=image_path, 
                                  image_name=image_name, width=width, height=height, near=self.near, far=self.far, 
                                  timestamp=(frame_num-self.start_time)/self.duration, pose=1, hpdirecitons=1,cxr=0.0, cyr=0.0)
        else: # TODO pose and hpdirections
            cam_info = CameraInfo(uid=uid, R=Rotation, T=Translation, FovY=FovY, FovX=FovX, image=image, image_path=image_path, 
                                  image_name=image_name, width=width, height=height, near=self.near, far=self.far, 
                                  timestamp=(frame_num-self.start_time)/self.duration, pose=None, hpdirecitons=None, cxr=0.0, cyr=0.0)
        return cam_info
    def get_rays(Rotation,Translation,FoVy,FoVx,width,height):
        world_view_transform = torch.tensor(getWorld2View2(Rotation, Translation, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        projectinverse = projection_matrix.T.inverse()
        camera2wold = world_view_transform.T.inverse()
        pixgrid = create_meshgrid(height, width, normalized_coordinates=False, device="cpu")[0]
        #pixgrid = pixgrid#.cuda()  # H,W,
        
        xindx = pixgrid[:,:,0] # x 
        yindx = pixgrid[:,:,1] # y
    
        
        ndcy, ndcx = pix2ndc(yindx, height), pix2ndc(xindx, width)
        ndcx = ndcx.unsqueeze(-1)
        ndcy = ndcy.unsqueeze(-1)# * (-1.0)
        
        ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4 

        projected = ndccamera @ projectinverse.T 
        diretioninlocal = projected / projected[:,:,3:] #v 


        direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T 
        rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)

        
        rayo = camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0)                                     #rayo.permute(2, 0, 1).unsqueeze(0)
        rayd = rays_d.permute(2, 0, 1).unsqueeze(0)
        rays = torch.cat([rayo, rayd], dim=1)
        return world_view_transform,full_proj_transform,camera_center,rays
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    near: float
    far: float
    timestamp: float
    pose: np.array 
    hpdirecitons: np.array
    cxr: float
    cyr: float
