import sys
sys.path.append(".")
sys.path.append("..")

import os
import pickle

from tqdm import tqdm
import numpy as np

from pymovis.motion.core import Motion
from pymovis.learning.rbf import RBF
from pymovis.utils.config import DatasetConfig
from pymovis.vis.heightmap import sample_height

""" Load data """
def load_windows(split, dir, filename):
    with open(os.path.join(dir, f"{split}_{filename}"), "rb") as f:
        dict = pickle.load(f)
    return dict["windows"]

def load_heightmaps(dir, filename):
    with open(os.path.join(dir, filename), "rb") as f:
        heightmaps = pickle.load(f)
    return heightmaps

""" Preprocessing functions """
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

def sample_top_patches(mfc, heightmaps, top_k, h_scale, v_scale, sensor_size):
    motion, feet_p, contact = mfc
    
    # get contact info
    feet_up          = feet_p[contact == 0]
    feet_down        = feet_p[contact == 1]
    feet_up_y        = feet_up[:, 1]
    feet_down_y      = feet_down[:, 1]
    feet_down_y_mean = np.mean(feet_down_y, axis=0, keepdims=True)

    # get terrain heights at feet positions
    terr_up_y        = sample_height(heightmaps, feet_up[..., 0], feet_up[..., 2], h_scale, v_scale)
    terr_down_y      = sample_height(heightmaps, feet_down[..., 0], feet_down[..., 2], h_scale, v_scale)
    terr_down_y_mean = np.mean(terr_down_y, axis=1, keepdims=True)

    # measure error
    err_down = 0.1 * np.sum(((terr_down_y - terr_down_y_mean) - (feet_down_y - feet_down_y_mean)) ** 2, axis=-1)
    err_up   = np.sum(np.maximum((terr_up_y - terr_down_y_mean) - (feet_up_y - feet_down_y_mean), 0) ** 2, axis=-1)

    if motion.type == "jumpy":
        terr_over_minh = 0.3
        err_jump = np.sum((np.maximum(((feet_up_y - terr_over_minh) - feet_down_y_mean) - (terr_up_y - terr_down_y_mean), 0.0) ** 2), axis=-1)
    else:
        err_jump = 0.0
    
    if motion.type == "beam":
        beam_min_height = 2.0

        base    = np.stack([pose.base for pose in motion.poses], axis=0)
        base_xz = base[:, (0, 2)]
        base_y  = sample_height(heightmaps, base_xz[:, 0], base_xz[:, 1], h_scale, v_scale)

        base_v = base[1:] - base[:-1]
        base_v = np.concatenate([base_v[:1], base_v], axis=0)
        base_v = base_v / np.linalg.norm(base_v, axis=-1, keepdims=True)

        perpend = np.cross(base_v, np.array([0, 1, 0], dtype=np.float32))
        perpend = perpend / np.linalg.norm(perpend, axis=-1, keepdims=True)

        disp = np.random.randn(len(base_xz), 2)
        far_enough = np.linalg.norm(disp, axis=-1) > 0.5

        base_xz = base_xz[far_enough]
        base_y  = base_y[:, far_enough]
        disp    = disp[far_enough]
        perpend = perpend[far_enough]

        disp_xz = base_xz + perpend[:, (0, 2)] * disp
        disp_y  = sample_height(heightmaps, disp_xz[:, 0], disp_xz[:, 1], h_scale, v_scale)

        err_beam = 0.001 * np.sum((np.maximum(disp_y - (base_y - beam_min_height), 0.0)) ** 2, axis=-1)
    else:
        err_beam = 0.0

    err = err_down + err_up + err_jump + err_beam

    # best fitting terrains
    terr_ids = np.argsort(err)[:top_k]
    terr_patches = heightmaps[terr_ids]

    # terrain fit editing
    terr_residuals = (feet_down_y - feet_down_y_mean) - (terr_down_y[terr_ids] - terr_down_y_mean[terr_ids])
    terr_fine_func = [RBF(smooth=0.01, function="linear", eps=5e-3) for _ in range(top_k)]

    h, w = terr_patches[0].shape
    x, z = np.meshgrid(np.arange(w, dtype=np.float32) - w / 2, np.arange(h, dtype=np.float32) - h / 2)
    xz = np.stack([x * h_scale, z * h_scale], axis=-1).reshape(-1, 2)
    terr = sample_height(terr_patches, xz[..., 0], xz[..., 1], h_scale, v_scale).reshape(top_k, h, w)
    for i in range(top_k):
        terr_fine_func[i].fit(feet_down[..., (0, 2)], terr_residuals[i])
        edit = terr_fine_func[i].forward(xz).reshape(h, w).astype(np.float32)
        terr_patches[i] = (terr[i] - terr_down_y_mean[terr_ids[i]] + feet_down_y_mean + edit) / v_scale

    # environment state (T: number of frames)
    base    = np.stack([pose.base for pose in motion.poses], axis=0) # (T, 3)
    up      = np.stack([pose.up for pose in motion.poses], axis=0)
    forward = np.stack([pose.forward for pose in motion.poses], axis=0)
    left    = np.cross(up, forward, axis=-1)
    R       = np.stack([left, up, forward], axis=-1) # (T, 3, 3)
    
    sensor_x, sensor_z = np.linspace(-1, 1, sensor_size, dtype=np.float32), np.linspace(-1, 1, sensor_size, dtype=np.float32)
    sensor_x, sensor_z = np.meshgrid(sensor_x, sensor_z)
    sensor_y = np.zeros_like(sensor_x)
    sensor = np.stack([sensor_x, sensor_y, sensor_z], axis=-1).reshape(-1, 3) # (S*S, 3)
    sensor = np.einsum("Tij,Sj->TSi", R, sensor) + base[:, None, :] # (T, S*S, 3)

    sensor_y = sample_height(terr_patches, sensor[..., 0], sensor[..., 2], h_scale, v_scale) # (K, T, S*S)
    base_y   = sample_height(terr_patches, base[:, 0], base[:, 2], h_scale, v_scale) # (K, T)

    env_state = np.concatenate([base, forward], axis=-1) # (T, 6)
    env_state = np.repeat(env_state[None], top_k, axis=0) # (K, T, 6)
    env_state[..., 1] = base_y
    env_state = np.concatenate([env_state, sensor_y], axis=-1) # (K, T, 6 + S*S)

    return terr_patches, env_state # (K, mapsize, mapsize), (K, T, 6 + S*S)

""" Main functions """
def generate_dataset(split, config):
    # load processed data
    motions = load_windows(split, config.dataset_dir, config.motion_pklname)
    feet_p, contact = zip(*[get_contact_info(motion) for motion in motions])
    heightmaps = load_heightmaps(config.dataset_dir, config.heightmap_pklname)

    # extract environment dataset
    patches, env_states = [], []
    # for motion_contact in tqdm(zip(motions[::2048], feet_p[::2048], contact[::2048]), total=len(motions), desc="Generating environment dataset"):
    for motion_contact in tqdm(zip(motions, feet_p, contact), total=len(motions), desc="Generating environment dataset"):
        k_patches, env_state = sample_top_patches(motion_contact, heightmaps, config.top_k, config.h_scale, config.v_scale, config.sensor_size)
        patches.append(k_patches)
        env_states.append(env_state)

    # dataset (N: number of motion windows, T: number of frames)
    patches    = np.concatenate(patches, axis=0) # (N * top_k, mapsize, mapsize)
    env_states = np.concatenate(env_states, axis=0) # (N * top_k, T, 6 + sensor_size*sensor_size)
    print("Patch samples:", patches.shape)
    print("Environment states:", env_states.shape)

    # save
    env_dset = {"patches": patches, "env": env_states}
    print("Saving environment dataset")
    env_path = os.path.join(config.dataset_dir, f"{split}_{config.env_pklname}")
    with open(env_path, "wb") as f:
        pickle.dump(env_dset, f, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    # config
    config = DatasetConfig.load("configs/config.json")

    # generate dataset
    generate_dataset("train", config)
    generate_dataset("test", config)

if __name__ == "__main__":
    main()