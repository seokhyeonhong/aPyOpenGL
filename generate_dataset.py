from OpenGL.GL import *
import os
import pickle
import time

import numpy as np

from pymovis.motion.ops import npmotion
from pymovis.motion.core import Motion

from pymovis.utils import util
from pymovis.learning.rbf import RBF

from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import MotionApp
from pymovis.vis.render import Render
from pymovis.vis.heightmap import Heightmap
from pymovis.vis.glconst import INCH_TO_METER

""" Global variables for the dataset """
MOTION_DIR    = "./data/animations/processed"
WINDOW_SIZE   = 50
WINDOW_OFFSET = 20
FPS           = 30

HEIGHTMAP_DIR = "./data/heightmaps/processed"
SPARSITY      = 15
SIZE          = 200
TOP_K_SAMPLES = 10
H_SCALE       = 2 * INCH_TO_METER
V_SCALE       = INCH_TO_METER

DATASET_DIR   = "./data/dataset"

""" Load from saved files """
def load_processed_motions(split):
    # skeleton
    with open(os.path.join(MOTION_DIR, "skeleton.pkl"), "rb") as f:
        skeleton = pickle.load(f)

    # features
    motion_path = os.path.join(MOTION_DIR, f"motion_{split}_size{WINDOW_SIZE}_offset{WINDOW_OFFSET}_fps{FPS}.npy")
    features = np.load(motion_path)

    # make motion
    local_R6, root_p = features[..., :-3], features[..., -3:]
    local_R = npmotion.R.from_R6(local_R6.reshape(-1, 6)).reshape(-1, WINDOW_SIZE, skeleton.num_joints, 3, 3)
    motions = [Motion.from_numpy(skeleton, R, p, fps=FPS) for R, p in zip(local_R, root_p)]

    return motions, skeleton, features

def load_processed_heightmaps():
    heightmap_path = os.path.join(HEIGHTMAP_DIR, f"heightmap_sparsity{SPARSITY}_size{SIZE}.npy")
    heightmaps = np.load(heightmap_path)
    return heightmaps

""" Data processing """
def get_contact_info(motion: Motion, vel_factor=2e-4):
    def _get_contact_info_by_jid(jid):
        global_p_jid = motion.global_p[:, jid]
        contact = np.sum((global_p_jid[1:] - global_p_jid[:-1]) ** 2, axis=-1) < vel_factor
        contact = np.concatenate([contact[0:1], contact], axis=0).astype(np.float32)
        return global_p_jid, contact
    
    contact_left_foot  = _get_contact_info_by_jid(motion.skeleton.idx_by_name["LeftFoot"])
    contact_left_toe   = _get_contact_info_by_jid(motion.skeleton.idx_by_name["LeftToe"])
    contact_right_foot = _get_contact_info_by_jid(motion.skeleton.idx_by_name["RightFoot"])
    contact_right_toe  = _get_contact_info_by_jid(motion.skeleton.idx_by_name["RightToe"])

    feet_p  = np.stack([contact_left_foot[0], contact_left_toe[0], contact_right_foot[0], contact_right_toe[0]], axis=-2)
    contact = np.stack([contact_left_foot[1], contact_left_toe[1], contact_right_foot[1], contact_right_toe[1]], axis=-1)
    return feet_p, contact

def sample_height(heightmap, x_, z_):
    """
    heightmap: (noh, w, h)
    x, z: (nof, noj)
    """
    h, w = heightmap.shape[1:]

    x = x_ / H_SCALE + w / 2
    z = z_ / H_SCALE + h / 2

    a0 = np.fmod(x, 1)
    a1 = np.fmod(z, 1)

    x0, x1 = np.floor(x).astype(np.int32), np.ceil(x).astype(np.int32)
    z0, z1 = np.floor(z).astype(np.int32), np.ceil(z).astype(np.int32)

    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    z0 = np.clip(z0, 0, h - 1)
    z1 = np.clip(z1, 0, h - 1)

    s0 = heightmap[:, z0, x0]
    s1 = heightmap[:, z0, x1]
    s2 = heightmap[:, z1, x0]
    s3 = heightmap[:, z1, x1]

    return V_SCALE * ((s0 * (1 - a0) + s1 * a0) * (1 - a1) + (s2 * (1 - a0) + s3 * a0) * a1)

def sample_top_patches(motion_contact, heightmaps, num_samples=TOP_K_SAMPLES):
    motion, feet_p, contact = motion_contact
    
    # get contact info
    feet_up          = feet_p[contact == 0]
    feet_down        = feet_p[contact == 1]
    feet_up_y        = feet_up[:, 1]
    feet_down_y      = feet_down[:, 1]
    feet_down_y_mean = np.mean(feet_down_y, axis=0, keepdims=True)

    # get terrain heights at feet positions
    terr_up_y        = sample_height(heightmaps, feet_up[..., 0], feet_up[..., 2])
    terr_down_y      = sample_height(heightmaps, feet_down[..., 0], feet_down[..., 2])
    terr_down_y_mean = np.mean(terr_down_y, axis=1, keepdims=True)

    # measure error
    err_down = 0.1 * np.sum(((terr_down_y - terr_down_y_mean) - (feet_down_y - feet_down_y_mean)) ** 2, axis=-1)
    err_up   = np.sum(np.maximum((terr_up_y - terr_down_y_mean) - (feet_up_y - feet_down_y_mean), 0) ** 2, axis=-1)
    err      = err_down + err_up

    # best fitting terrains
    terr_ids = np.argsort(err)[:num_samples]
    terr_patches = heightmaps[terr_ids]

    # terrain fit editing
    terr_residuals = (feet_down_y - feet_down_y_mean) - (terr_down_y[terr_ids] - terr_down_y_mean[terr_ids])
    terr_fine_func = [RBF(smooth=0.01, function="gaussian", eps=1e-8) for _ in range(num_samples)]
    edits = []
    for i in range(num_samples):
        h, w = terr_patches[i].shape
        x, z = np.meshgrid(np.arange(w) - w / 2, np.arange(h) - h / 2)
        terr = sample_height(terr_patches[i:i+1], x * H_SCALE, z * H_SCALE).reshape(h, w)

        terr_fine_func[i].fit(feet_down[..., (0, 2)], terr_residuals[i])
        edit = terr_fine_func[i].forward(np.stack([x, z], axis=-1).reshape(-1, 2)).reshape(h, w)
        
        terr_patches[i] = (terr - terr_down_y_mean[terr_ids[i]] + feet_down_y_mean + edit) / V_SCALE
        edits.append(edit / V_SCALE)
    
    # sample environment maps
    grid_x, grid_z = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
    grid_y = np.zeros_like(grid_x)
    grid = np.stack([grid_x, grid_y, grid_z], axis=-1)
    
    forward = np.stack([pose.forward for pose in motion.poses], axis=0)
    left    = np.stack([pose.left for pose in motion.poses], axis=0)
    up      = np.stack([pose.up for pose in motion.poses], axis=0)
    base    = np.stack([pose.base for pose in motion.poses], axis=0)
    R       = np.stack([left, up, forward], axis=-1)
    
    env_map = np.einsum("ijk,abk->iabj", R, grid) + base[:, None, None, :]
    env_map = env_map.reshape(env_map.shape[0], -1, 3)
    
    env_y = []
    for i in range(num_samples):
        env_y.append(sample_height(terr_patches[i:i+1], env_map[..., 0], env_map[..., 2]))

    env_map = np.repeat(env_map[None, ...], num_samples, axis=0)
    env_map[..., 1] = np.concatenate(env_y, axis=0)
    
    return terr_patches, env_map

""" Main functions """
def generate_dataset(split="train"):
    # load processed data
    motions, skeleton, features = load_processed_motions(split)
    contact_info = [get_contact_info(motion) for motion in motions]
    feet_p, contact = zip(*contact_info)

    heightmaps = load_processed_heightmaps()

    # create directories
    feature_dir = os.path.join(DATASET_DIR, split, "features")
    vis_dir = os.path.join(DATASET_DIR, split, "vis")

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    # generate and save dataset
    start = time.perf_counter()
    for idx, motion_contact in enumerate(zip(motions, feet_p, contact)):
        print("Processing motion %d / %d" % (idx + 1, len(motions)), end="\r")
        patch, envmap = sample_top_patches(motion_contact, heightmaps, num_samples=TOP_K_SAMPLES)
        for idx2, (p, e) in enumerate(zip(patch, envmap)):
            with open(os.path.join(vis_dir, f"{idx}_{idx2}.pkl"), "wb") as f:
                pickle.dump([motion_contact[0], p, e, motion_contact[2]], f)
            np.save(os.path.join(feature_dir, f"{idx}_{idx2}.npy"), np.concatenate([features[idx], e[..., 1]], axis=-1))
    
    print(f"Processed {len(motions)} motions in {time.perf_counter() - start:.2f} seconds")

def visualize(split):
    vis_dir = os.path.join(DATASET_DIR, split, "vis")
    for idx, file in enumerate(os.listdir(vis_dir)):
        print("Visualizing %d / %d" % (idx + 1, len(os.listdir(vis_dir))), end="\r")
        with open(os.path.join(vis_dir, file), "rb") as f:
            motion, patch, envmap, contact = pickle.load(f)

        app_manager = AppManager()
        app = MyApp(motion, contact, patch, envmap)
        app_manager.run(app)

class MyApp(MotionApp):
    def __init__(self, motion, contact, heightmap, envmap):
        super().__init__(motion)
        self.contact = contact
        self.sphere = Render.sphere().set_material([0, 1, 0])
        self.grid.set_visible(False)
        self.axis.set_visible(False)
        
        jid_left_foot  = self.motion.skeleton.idx_by_name["LeftFoot"]
        jid_left_toe   = self.motion.skeleton.idx_by_name["LeftToe"]
        jid_right_foot = self.motion.skeleton.idx_by_name["RightFoot"]
        jid_right_toe  = self.motion.skeleton.idx_by_name["RightToe"]
        self.jid       = [jid_left_foot, jid_left_toe, jid_right_foot, jid_right_toe]

        self.heightmap_mesh = Render.mesh(Heightmap(heightmap, h_scale=H_SCALE, v_scale=V_SCALE, offset=0).mesh).set_texture("grid.png").set_uv_repeat(0.1)

        self.envmap = envmap
    
    def render(self):
        super().render()
        contact = self.contact[self.frame]
        envmap = self.envmap[self.frame]

        glDisable(GL_DEPTH_TEST)
        for idx, jid in enumerate(self.jid):
            self.sphere.set_position(self.motion.global_p[self.frame, jid]).set_material([0, 1, 0]).set_scale(0.1 * contact[idx]).draw()
        glEnable(GL_DEPTH_TEST)

        self.heightmap_mesh.draw()

        for p in envmap:
            self.sphere.set_position(p).set_scale(0.1).set_material([1, 1, 0]).draw()

def main():
    # dataset generation
    # generate_dataset("train")

    # dataset visualization
    visualize("train")

if __name__ == "__main__":
    main()