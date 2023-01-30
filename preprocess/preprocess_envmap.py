import sys
sys.path.append(".")
sys.path.append("..")

import os
import pickle

from tqdm import tqdm
import numpy as np

from pymovis.motion.core import Motion
from pymovis.learning.rbf import RBF
from pymovis.vis.const import INCH_TO_METER
from pymovis.utils import util

""" Load processed data """
def load_processed_motions(split, dir, filename):
    with open(os.path.join(dir, f"{split}_{filename}"), "rb") as f:
        dict = pickle.load(f)
    return dict["windows"], dict["features"]

def load_processed_heightmaps(dir, filename):
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

def sample_height(heightmap, x_, z_, h_scale, v_scale):
    """
    heightmap: (noh, w, h)
    x, z: (nof, noj)
    """
    h, w = heightmap.shape[1:]

    x = x_ / h_scale + w / 2
    z = z_ / h_scale + h / 2

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

    return v_scale * ((s0 * (1 - a0) + s1 * a0) * (1 - a1) + (s2 * (1 - a0) + s3 * a0) * a1)

def sample_top_patches(mfc, heightmaps, top_k, h_scale, v_scale):
    motion, feet_p, contact = mfc
    
    # get contact info
    feet_up          = feet_p[contact == 0]
    feet_down        = feet_p[contact == 1]
    feet_up_y        = feet_up[:, 1]
    feet_down_y      = feet_down[:, 1]
    feet_down_y_mean = np.mean(feet_down_y, axis=0, keepdims=True)
    
    # get motion info
    forward = np.stack([pose.forward for pose in motion.poses], axis=0)
    left    = np.stack([pose.left for pose in motion.poses], axis=0)
    up      = np.stack([pose.up for pose in motion.poses], axis=0)
    base    = np.stack([pose.base for pose in motion.poses], axis=0)
    R       = np.stack([left, up, forward], axis=-1)

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

        xz = base_xz + perpend[:, (0, 2)] * disp
        perp_y1 = sample_height(heightmaps, xz[:, 0], xz[:, 1], h_scale, v_scale)

        err_beam = 0.001 * np.sum((np.maximum(perp_y1 - (base_y - beam_min_height), 0.0)) ** 2, axis=-1)
    else:
        err_beam = 0.0

    err = err_down + err_up + err_jump + err_beam

    # best fitting terrains
    terr_ids = np.argsort(err)[:top_k]
    terr_patches = heightmaps[terr_ids]

    # terrain fit editing
    terr_residuals = (feet_down_y - feet_down_y_mean) - (terr_down_y[terr_ids] - terr_down_y_mean[terr_ids])
    terr_fine_func = [RBF(smooth=0.01, function="linear", eps=5e-3) for _ in range(top_k)]
    edits = []

    h, w = terr_patches[0].shape
    x, z = np.meshgrid(np.arange(w, dtype=np.float32) - w / 2, np.arange(h, dtype=np.float32) - h / 2)
    x *= h_scale
    z *= h_scale
    terr = sample_height(terr_patches, x, z, h_scale, v_scale).reshape(top_k, h, w)
    xz = np.stack([x, z], axis=-1).reshape(-1, 2)
    for i in range(top_k):
        terr_fine_func[i].fit(feet_down[..., (0, 2)], terr_residuals[i])
        edit = terr_fine_func[i].forward(xz).reshape(h, w).astype(np.float32)
        terr_patches[i] = (terr[i] - terr_down_y_mean[terr_ids[i]] + feet_down_y_mean + edit) / v_scale
        edits.append(edit / v_scale)
    
    # sample environment maps
    grid_x, grid_z = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
    grid_y         = np.zeros_like(grid_x)
    grid           = np.stack([grid_x, grid_y, grid_z], axis=-1)
    
    env_map = np.einsum("ijk,abk->iabj", R, grid) + base[:, None, None, :]
    env_map = env_map.reshape(env_map.shape[0], -1, 3)
    
    env_y = sample_height(terr_patches, env_map[..., 0], env_map[..., 2], h_scale, v_scale)
    env_map = np.repeat(env_map[None, ...], top_k, axis=0)
    env_map[..., 1] = env_y

    # root trajectory (xz position and forward direction)
    root_traj = np.concatenate([base[:, (0, 2)], forward], axis=-1)
    root_traj = np.repeat(root_traj[None, ...], top_k, axis=0)
    
    return {
        "patches": terr_patches,
        "edits": np.stack(edits, axis=0),
        "env_map": env_map,
        "root_traj": root_traj
    }

""" Main functions """
def generate_dataset(split, motion_config, hmap_config):
    # load processed data
    motions, features = load_processed_motions(split, motion_config["save_dir"], motion_config["save_filename"])
    feet_p, contact = zip(*[get_contact_info(motion) for motion in motions])
    heightmaps = load_processed_heightmaps(hmap_config["save_dir"], hmap_config["save_filename"])

    # extract envmap dataset
    env_maps, vis_data = [], []
    # for motion_contact in tqdm(zip(motions[::32], feet_p[::32], contact[::32]), total=len(motions), desc="Generating envmap dataset"):
    for motion_contact in tqdm(zip(motions, feet_p, contact), total=len(motions), desc="Generating envmap dataset"):
        dict = sample_top_patches(motion_contact, heightmaps, hmap_config["top_k"], hmap_config["h_scale"] * INCH_TO_METER, hmap_config["v_scale"] * INCH_TO_METER)
        patch = dict["patches"]
        edit = dict["edits"]
        env_map = dict["env_map"]
        root_traj = dict["root_traj"]

        env_data = np.concatenate([root_traj, env_map[..., 1]], axis=-1).astype(np.float32)
        env_maps.append(env_data)

        # NOTE: don't have to save this data, just for visualization
        vis_data.append([motion_contact[0], patch, edit, env_map, motion_contact[2]])

    env_maps = np.stack(env_maps, axis=0)
    print("Envmap dataset shape:", env_maps.shape)

    # create directory
    envmap_dir = "./data/dataset/envmap/"
    vis_dir = "./data/dataset/vis/"
    save_filename = f"length{motion_config['window_length']}_offset{motion_config['window_offset']}_fps{motion_config['fps']}_sparsity{hmap_config['sparsity']}_mapsize{hmap_config['mapsize']}_topk{hmap_config['top_k']}.pkl"
    if not os.path.exists(envmap_dir):
        os.makedirs(envmap_dir)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    # save
    print("Saving envmap dataset")
    envmap_path = os.path.join(envmap_dir, f"{split}_{save_filename}")
    with open(envmap_path, "wb") as f:
        pickle.dump(env_maps, f)
    
    # NOTE: don't have to save this data, just for visualization
    vis_path = os.path.join(vis_dir, f"{split}_{save_filename}")
    with open(vis_path, "wb") as f:
        pickle.dump(vis_data, f)

def main():
    # config
    motion_config, hmap_config = util.config_parser()

    # generate dataset
    generate_dataset("train", motion_config, hmap_config)
    generate_dataset("test", motion_config, hmap_config)

if __name__ == "__main__":
    main()