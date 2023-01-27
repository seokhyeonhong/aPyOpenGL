import sys
sys.path.append(".")
sys.path.append("..")

import os
import pickle

from OpenGL.GL import *
from tqdm import tqdm
import numpy as np

from pymovis.motion.core import Motion
from pymovis.learning.rbf import RBF
from pymovis.vis.const import INCH_TO_METER

""" Global variables for the dataset """
WINDOW_SIZE     = 50
WINDOW_OFFSET   = 20
FPS             = 30
MOTION_DIR      = f"./data/dataset/motion"
MOTION_FILENAME = f"length{WINDOW_SIZE}_offset{WINDOW_OFFSET}_fps{FPS}.pkl"

SPARSITY        = 15
SIZE            = 140
TOP_K_SAMPLES   = 10
H_SCALE         = 2 * INCH_TO_METER
V_SCALE         = INCH_TO_METER
HEIGHTMAP_DIR   = f"./data/dataset/heightmap"
HEIGHT_FILENAME = f"sparsity{SPARSITY}_mapsize{SIZE}.pkl"

ENVMAP_DIR      = f"./data/dataset/envmap/"
VIS_DIR         = f"./data/dataset/vis/"
SAVE_FILENAME   = f"length{WINDOW_SIZE}_offset{WINDOW_OFFSET}_fps{FPS}_sparsity{SPARSITY}_mapsize{SIZE}_top{TOP_K_SAMPLES}.pkl"

""" Load processed data """
def load_processed_motions(split):
    with open(os.path.join(MOTION_DIR, f"{split}_{MOTION_FILENAME}"), "rb") as f:
        dict = pickle.load(f)
    return dict["windows"], dict["features"]

def load_processed_heightmaps():
    with open(os.path.join(HEIGHTMAP_DIR, HEIGHT_FILENAME), "rb") as f:
        heightmaps = pickle.load(f)
    return heightmaps

""" Data processing """
def get_contact_info(motion: Motion, vel_factor=2e-4):
    def _get_contact_info_by_jid(jid):
        global_p_jid = np.stack([pose.global_p[jid] for pose in motion.poses], axis=0)
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

def sample_top_patches(mfc, heightmaps):
    motion, feet_p, contact = mfc
    
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
    if motion.type == "jumpy":
        terr_over_minh = 0.3
        err_jump = np.sum((np.maximum(((feet_up_y - feet_down_y_mean) - terr_over_minh) - (terr_up_y - terr_down_y_mean), 0.0) ** 2), axis=-1)
    else:
        err_jump = 0.0

    err = err_down + err_up + err_jump

    # best fitting terrains
    terr_ids = np.argsort(err)[:TOP_K_SAMPLES]
    terr_patches = heightmaps[terr_ids]

    # terrain fit editing
    terr_residuals = (feet_down_y - feet_down_y_mean) - (terr_down_y[terr_ids] - terr_down_y_mean[terr_ids])
    terr_fine_func = [RBF(smooth=0.01, function="linear", eps=5e-3) for _ in range(TOP_K_SAMPLES)]
    edits = []
    for i in range(TOP_K_SAMPLES):
        terr_fine_func[i].fit(feet_down[..., (0, 2)], terr_residuals[i])

        h, w = terr_patches[i].shape
        x, z = np.meshgrid(np.arange(w, dtype=np.float32) - w / 2, np.arange(h, dtype=np.float32) - h / 2)
        x *= H_SCALE
        z *= H_SCALE
        
        terr = sample_height(terr_patches[i:i+1], x, z).reshape(h, w)
        edit = terr_fine_func[i].forward(np.stack([x, z], axis=-1).reshape(-1, 2)).reshape(h, w).astype(np.float32)
        
        terr_patches[i] = (terr - terr_down_y_mean[terr_ids[i]] + feet_down_y_mean + edit) / V_SCALE
        edits.append(edit / V_SCALE)
    
    # sample environment maps
    grid_x, grid_z = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
    grid_y         = np.zeros_like(grid_x)
    grid           = np.stack([grid_x, grid_y, grid_z], axis=-1)
    
    forward = np.stack([pose.forward for pose in motion.poses], axis=0)
    left    = np.stack([pose.left for pose in motion.poses], axis=0)
    up      = np.stack([pose.up for pose in motion.poses], axis=0)
    base    = np.stack([pose.base for pose in motion.poses], axis=0)
    R       = np.stack([left, up, forward], axis=-1)
    
    env_map = np.einsum("ijk,abk->iabj", R, grid) + base[:, None, None, :]
    env_map = env_map.reshape(env_map.shape[0], -1, 3)
    
    env_y = []
    for i in range(TOP_K_SAMPLES):
        env_y.append(sample_height(terr_patches[i:i+1], env_map[..., 0], env_map[..., 2]))

    env_map = np.repeat(env_map[None, ...], TOP_K_SAMPLES, axis=0)
    env_map[..., 1] = np.concatenate(env_y, axis=0)

    root_traj = np.concatenate([base, forward], axis=-1)
    root_traj = np.repeat(root_traj[None, ...], TOP_K_SAMPLES, axis=0)
    
    return {
        "patches": terr_patches,
        "edits": np.stack(edits, axis=0),
        "env_map": env_map,
        "root_traj": root_traj
    }

""" Main functions """
def generate_dataset(split="train"):
    # load processed data
    motions, features = load_processed_motions(split)
    feet_p, contact = zip(*[get_contact_info(motion) for motion in motions])
    heightmaps = load_processed_heightmaps()

    # extract envmap dataset
    env_maps, vis_data = [], []
    # for motion_contact in tqdm(zip(motions[::512], feet_p[::512], contact[::512]), total=len(motions), desc="Generating envmap dataset"):
    for motion_contact in tqdm(zip(motions, feet_p, contact), total=len(motions), desc="Generating envmap dataset"):
        dict = sample_top_patches(motion_contact, heightmaps)
        patch = dict["patches"]
        edit = dict["edits"]
        env_map = dict["env_map"]
        root_traj = dict["root_traj"]

        env_data = np.concatenate([root_traj, env_map[..., 1]], axis=-1).astype(np.float32)
        env_maps.append(env_data)
        vis_data.append([motion_contact[0], patch, edit, env_map, motion_contact[2]])

    env_maps = np.stack(env_maps, axis=0)
    print("Envmap dataset shape:", env_maps.shape)

    # create directory
    if not os.path.exists(ENVMAP_DIR):
        os.makedirs(ENVMAP_DIR)
    if not os.path.exists(VIS_DIR):
        os.makedirs(VIS_DIR)

    # save
    print("Saving envmap dataset")
    envmap_path = os.path.join(ENVMAP_DIR, f"{split}_{SAVE_FILENAME}")
    with open(envmap_path, "wb") as f:
        pickle.dump(env_maps, f)
    
    vis_path = os.path.join(VIS_DIR, f"{split}_{SAVE_FILENAME}")
    with open(vis_path, "wb") as f:
        pickle.dump(vis_data, f)

def main():
    generate_dataset("train")
    generate_dataset("test")

if __name__ == "__main__":
    main()