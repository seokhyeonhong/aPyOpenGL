import torch
import glfw
import glm
import copy

import numpy as np

from pymovis.motion.data import bvh
from pymovis.motion.core import Motion
from pymovis.motion.ops import npmotion

from pymovis.vis.core import Vertex, VAO, Mesh
from pymovis.vis.render import Render
from pymovis.vis.app import MotionApp
from pymovis.vis.appmanager import AppManager

class Heightmap:
    def __init__(self, filename, h_scale=1.0, v_scale=1.0, offset=None):
        """
        filename : Path to the heightmap file
        h_scale : Horizontal scale of the heightmap (xz plane)
        v_scale : Vertical scale of the heightmap (y axis)
        offset : Offset of the heightmap (y axis)
        """
        self.filename = filename
        self.h_scale = h_scale
        self.v_scale = v_scale
        self.offset = offset

        self.load()
    
    def load(self):
        """ Load the heightmap data and create the vertices """
        self.data = np.loadtxt(self.filename, dtype=np.float32)
        w = len(self.data)
        h = len(self.data[0])
        self.offset = np.sum(self.data) / (w * h) if self.offset is None else 0

        vertices = [Vertex() for _ in range(w * h)]

        """ Calculate and set the positions of the vertices """
        cw = self.h_scale * w
        ch = self.h_scale * h
        cx = self.h_scale * np.arange(w)
        cy = self.h_scale * np.arange(h)
        cx, cy = np.meshgrid(cx, cy)

        x_pos = cx - cw / 2
        z_pos = cy - ch / 2
        y_pos = self.sample(x_pos, z_pos)
        positions = np.stack([x_pos, y_pos, z_pos], axis=-1)

        print(f"Heightmap size: {x_pos.max() - x_pos.min():.4f}m x {z_pos.max() - z_pos.min():.4f}m")

        for i, pos in enumerate(positions.reshape(-1, 3)):
            vertices[i].set_position(pos)
        
        """ Calculate and set the normals of the vertices """
        normals = np.empty((h, w, 3), dtype=np.float32)

        cross1 = np.cross(positions[2:, 1:-1] - positions[1:-1, 1:-1], positions[1:-1, 2:] - positions[1:-1, 1:-1])
        cross2 = np.cross(positions[:-2, 1:-1] - positions[1:-1, 1:-1], positions[1:-1, :-2] - positions[1:-1, 1:-1])
        cross = (cross1 + cross2) * 0.5
        cross = cross / (np.linalg.norm(cross, axis=-1, keepdims=True) + 1e-8)

        normals[1:-1, 1:-1] = cross
        normals[0, :] = normals[-1, :] = np.array([0, 1, 0])
        normals[:, 0] = normals[:, -1] = np.array([0, 1, 0])

        for i, normal in enumerate(normals.reshape(-1, 3)):
            vertices[i].set_normal(normal)
        
        """ Set vertex indices """
        indices = np.empty((h-1, w-1, 6), dtype=np.int32)
        indices[..., 0] = np.arange((h-1) * (w-1)).reshape(h-1, w-1)
        indices[..., 1] = indices[..., 0] + w
        indices[..., 2] = indices[..., 0] + 1
        indices[..., 3] = indices[..., 0] + 1
        indices[..., 4] = indices[..., 0] + w
        indices[..., 5] = indices[..., 0] + w + 1
        indices = indices.flatten()

        """ Generate VAO and Mesh """
        vao = VAO.from_vertex_array(vertices, indices)
        self.mesh = Mesh(vao, vertices, indices)

    def sample(self, x, z):
        w = len(self.data)
        h = len(self.data[0])

        x = (x / self.h_scale) + (w / 2)
        z = (z / self.h_scale) + (h / 2)

        a0 = np.fmod(x, 1)
        a1 = np.fmod(z, 1)

        x0, x1 = np.floor(x).astype(np.int32), np.ceil(x).astype(np.int32)
        z0, z1 = np.floor(z).astype(np.int32), np.ceil(z).astype(np.int32)
        
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)
        z0 = np.clip(z0, 0, h - 1)
        z1 = np.clip(z1, 0, h - 1)

        s0 = self.v_scale * (self.data[x0, z0] - self.offset)
        s1 = self.v_scale * (self.data[x1, z0] - self.offset)
        s2 = self.v_scale * (self.data[x0, z1] - self.offset)
        s3 = self.v_scale * (self.data[x1, z1] - self.offset)

        return (s0 * (1 - a0) + s1 * a0) * (1 - a1) + (s2 * (1 - a0) + s3 * a0) * a1

class MyApp(MotionApp):
    def __init__(self, motion: Motion, vel_factor):
        super().__init__(motion)
        self.motion = motion
        self.vel_factor = vel_factor

        self.left_leg_idx = self.motion.skeleton.idx_by_name["LeftUpLeg"]
        self.left_foot_idx = self.motion.skeleton.idx_by_name["LeftFoot"]
        self.right_leg_idx = self.motion.skeleton.idx_by_name["RightUpLeg"]
        self.right_foot_idx = self.motion.skeleton.idx_by_name["RightFoot"]

        heightmap = Heightmap("./data/heightmaps/hmap_010_smooth.txt", 0.0254, 0.0254)
        self.heightmap = Render.mesh(heightmap.mesh).set_material(0.5)

        self.motion.two_bone_ik(self.left_leg_idx, self.left_foot_idx, self.motion.root_p)

        # self.scaled_motion = self.get_scaled_motion()
    
    def get_scaled_motion(self):
        # TODO: parallelize this using numpy array functions
        
        motions = []
        for v in self.vel_factor:
            poses = []
            for i in range(len(self.motion)):
                pose = copy.deepcopy(self.motion.poses[i])

                _, global_p = npmotion.R.fk(pose.local_R, pose.root_p, pose.skeleton)
                pose.root_p *= np.array([v, 1, v])
                
                left_foot_p = global_p[self.left_foot_idx] * np.array([v, 1, v])
                pose.two_bone_ik(self.left_leg_idx, self.left_foot_idx, left_foot_p)

                right_foot_p = global_p[self.right_foot_idx] * np.array([v, 1, v])
                pose.two_bone_ik(self.right_leg_idx, self.right_foot_idx, right_foot_p)
                
                poses.append(pose)
            motions.append(Motion(f"scaled_by_{v}", self.motion.skeleton, poses, fps=self.motion.fps))
        return motions
        # _, global_p = npmotion.R.fk(self.motion.local_R, self.motion.root_p, self.motion.skeleton)
        
        # gt_root_p = global_p[:, self.hip_idx]
        # gt_left_foot = global_p[:, self.left_foot_idx]
        # gt_right_foot = global_p[:, self.right_foot_idx]

        # root_p = [global_p[0, self.hip_idx]]
        # left_foot_p = [global_p[0, self.left_foot_idx]]
        # right_foot_p = [global_p[0, self.right_foot_idx]]
        
        # for i in range(1, len(self.motion)):
        #     # root_v = gt_root_p[i] - gt_root_p[i-1]
        #     # root_p.append(root_p[-1] + self.vel_factor * root_v)

        #     # curr_root2foot = npmotion.normalize(gt_left_foot[i] - gt_root_p[i])
        #     # prev_root2foot = npmotion.normalize(gt_left_foot[i-1] - gt_root_p[i-1])
        #     # angle = np.arccos(np.dot(prev_root2foot, curr_root2foot))[..., np.newaxis]
        #     # axis = np.cross(prev_root2foot, curr_root2foot)
        #     # R = npmotion.R.from_A(angle * self.vel_factor, axis)

        #     # left_foot_pos = root_p[-1] + R @ (gt_left_foot[i-1] - gt_root_p[i-1])
        #     # left_foot_pos[1] = gt_left_foot[i, 1]
        #     # left_foot_p.append(left_foot_pos)
        #     root_p.append(gt_root_p[i] * np.array((self.vel_factor, 1, self.vel_factor)))
        #     left_foot_p.append(gt_left_foot[i] * np.array((self.vel_factor, 1, self.vel_factor)))
        #     right_foot_p.append(gt_right_foot[i] * np.array((self.vel_factor, 1, self.vel_factor)))

        # return np.array(root_p, dtype=np.float32), np.array(left_foot_p, dtype=np.float32), np.array(right_foot_p, dtype=np.float32)

    def render(self):
        super().render()
        self.heightmap.draw()
        # for m in self.scaled_motion:
        #     m.render_by_frame(self.frame, 1.0)

if __name__ == "__main__":
    motion = bvh.load("D:/data/LaFAN1/aiming1_subject1.bvh", v_forward=[0, 1, 0], v_up=[1, 0, 0], to_meter=0.01)
    motion.align_by_frame(0)

    app_manager = AppManager()
    app = MyApp(motion, [1.2])
    app_manager.run(app)