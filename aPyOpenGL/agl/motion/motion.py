from __future__ import annotations
import numpy as np
import copy
import os

from .pose import Pose

from aPyOpenGL.transforms import n_quat


def _global_xforms_to_skeleton_xforms(global_xforms, parent_idx):
    nof, noj = global_xforms.shape[:2]

    skeleton_xforms = np.stack([np.identity(4, dtype=np.float32) for _ in range(noj - 1)], axis=0)
    skeleton_xforms = np.stack([skeleton_xforms for _ in range(nof)], axis=0)

    for i in range(1, noj):
        parent_pos = global_xforms[:, parent_idx[i], :3, 3]
        
        target_dir = global_xforms[:, i, :3, 3] - parent_pos
        target_dir = target_dir / (np.linalg.norm(target_dir, axis=-1, keepdims=True) + 1e-8)

        quat = n_quat.between_vecs(np.array([0, 1, 0], dtype=np.float32), target_dir)

        skeleton_xforms[:, i-1, :3, :3] = n_quat.to_rotmat(quat)
        skeleton_xforms[:, i-1, :3,  3] = (parent_pos + global_xforms[:, i, :3, 3]) / 2

    return skeleton_xforms

class Motion:
    """
    Motion class that contains the skeleton and its sequence of poses.

    Attributes:
        poses     (list[Pose]): A sequence of poses.
        fps       (float)     : The number of frames per second.
        name      (str)       : The name of the motion.
    """
    def __init__(
        self,
        poses : list[Pose],
        fps   : float = 30.0,
        name  : str   = "default",
    ):
        self.__poses : list[Pose] = poses
        self.__name  : str        = name
        self.fps   : float      = fps

        self.update_global_xform(verbose=True)

    def __len__(self):
        return len(self.__poses)
    
    @property
    def num_frames(self):
        return len(self.__poses)
    
    
    @property
    def skeleton(self):
        return self.__poses[0].skeleton
    

    @property
    def poses(self):
        return self.__poses.copy()
    

    @property
    def name(self):
        return str(self.__name)
    

    @poses.setter
    def poses(self, value: list[Pose]):
        self.__poses = value.copy()
    
    
    def remove_joint_by_name(self, joint_name):
        remove_indices = self.skeleton.remove_joint_by_name(joint_name)
        for pose in self.__poses:
            pose.skeleton = self.skeleton
            pose.local_quats = np.delete(pose.local_quats, remove_indices, axis=0)


    def export_as_bvh(self, filename, rot_order="XYZ"):
        self.__save(filename, rot_order=rot_order)

    
    def update_global_xform(self, verbose=False):
        local_quats = np.stack([pose.local_quats for pose in self.__poses], axis=0) # (T, J, 4)
        root_pos = np.stack([pose.root_pos for pose in self.__poses], axis=0) # (T, 3)

        # fk
        gq, gp = n_quat.fk(local_quats, root_pos, self.skeleton)
        gr = n_quat.to_rotmat(gq)
        gx = np.stack([np.identity(4, dtype=np.float32) for _ in range(self.skeleton.num_joints)], axis=0)
        gx = np.stack([gx for _ in range(len(self.__poses))], axis=0)
        gx[..., :3, :3] = gr
        gx[..., :3,  3] = gp

        # skeleton xforms
        sx = _global_xforms_to_skeleton_xforms(gx, self.skeleton.parent_idx)

        for i in range(len(self.__poses)):
            self.__poses[i].set_global_xform(gx[i], sx[i])

        if verbose:
            print(f" > Global transformations of {self.name} updated.")


    def __save(self, filename, scale=100.0, rot_order="ZXY", verbose=False):
        if verbose:
            print(" >  >  Save BVH file: %s" % filename)
        with open(filename, "w") as f:
            """ Write hierarchy """
            if verbose:
                print(" >  >  >  >  Write BVH hierarchy")
            f.write("HIERARCHY\n")
            joint_order = self._write_hierarchy(
                f, self.skeleton, 0, scale, rot_order
            )
            """ Write data """
            if verbose:
                print(" >  >  >  >  Write BVH data")
            t_start = 0
            dt = 1.0 / self.fps
            num_frames = self.num_frames
            f.write("MOTION\n")
            f.write("Frames: %d\n" % num_frames)
            f.write("Frame Time: %f\n" % dt)
            t = t_start

            for i in range(num_frames):
                if verbose and i % self.fpsjoint.local_pos == 0:
                    print(
                        "\r >  >  >  >  %d/%d processed (%d FPS)"
                        % (i + 1, num_frames, self.fps),
                        end=" ",
                    )
                pose = self.__poses[i]

                p = self.__poses[i].root_pos
                p *= scale
                f.write("%f %f %f " % (p[0], p[1], p[2]))
                for quat in self.__poses[i].local_quats:
                    # R = self._Q2E(quat, rot_order)
                    R = n_quat.to_euler(quat, rot_order, radians=False)
                    f.write("%f %f %f " % (R[0], R[1], R[2]))
                f.write("\n")
                t += dt
                if verbose and i == num_frames - 1:
                    print(
                        "\r >  >  >  >  %d/%d processed (%d FPS)"
                        % (i + 1, num_frames, self.fps)
                    )
            f.close()

    def _write_hierarchy(self, file, skeleton, joint_idx, scale=1.0, rot_order="XYZ", tab=""):
        def rot_order_to_str(order):
            if order == "xyz" or order == "XYZ":
                return "Xrotation Yrotation Zrotation"
            elif order == "zyx" or order == "ZYX":
                return "Zrotation Yrotation Xrotation"
            elif order == "zxy" or order == "ZXY":
                return "Zrotation Xrotation Yrotation"
            else:
                raise NotImplementedError

        joint = skeleton.joints[joint_idx]
        child_joints = skeleton.children_idx[joint_idx]
        joint_order = [joint.name]
        is_root_joint = (skeleton.parent_idx[joint_idx] == -1)
        if is_root_joint:
            file.write(tab + "ROOT %s\n" % joint.name)
        else:
            file.write(tab + "JOINT %s\n" % joint.name)
        file.write(tab + "{\n")
        p = joint.local_pos
        p *= scale
        file.write(tab + "\tOFFSET %f %f %f\n" % (p[0], p[1], p[2]))
        if is_root_joint:
            file.write(
                tab
                + "\tCHANNELS 6 Xposition Yposition Zposition %s\n"
                % rot_order_to_str(rot_order)
            )
        else:
            file.write(tab + "\tCHANNELS 3 %s\n" % rot_order_to_str(rot_order))
        for child_joint_idx in child_joints:
            child_joint_order = self._write_hierarchy(
                file, skeleton, child_joint_idx, scale, rot_order, tab + "\t"
            )
            joint_order.extend(child_joint_order)
        if len(child_joints) == 0:
            file.write(tab + "\tEnd Site\n")
            file.write(tab + "\t{\n")
            file.write(tab + "\t\tOFFSET %f %f %f\n" % (0.0, 0.0, 0.0))
            file.write(tab + "\t}\n")
        file.write(tab + "}\n")
        return joint_order

    # def _Q2E(self, Q, order="yxz", degrees=True):
    #     w = Q[0]
    #     x = Q[1]
    #     y = Q[2]
    #     z = Q[3]
    #     modifiedQ = [x,y,z,w]

    #     return Rotation.from_quat(modifiedQ).as_euler(order, degrees=degrees)
    
    def mirror(self, pair_indices, sym_axis=None):
        local_quats = np.stack([pose.local_quats for pose in self.__poses], axis=0)
        root_pos = np.stack([pose.root_pos for pose in self.__poses], axis=0)

        # swap joint indices
        local_quats = local_quats[:, pair_indices]

        # mirror by symmetry axis
        if sym_axis is None:
            sym_axis = self.skeleton.find_symmetry_axis(pair_indices)
        else:
            assert sym_axis in ["x", "y", "z"], f"Invalid axis {sym_axis} for symmetry axis, must be one of ['x', 'y', 'z']"

        idx = {"x": 0, "y": 1, "z": 2}[sym_axis]
        local_quats[:, :, (0, idx+1)] *= -1
        root_pos[:, idx] *= -1

        mirrored_poses = []
        for i in range(len(self.__poses)):
            mirrored_poses.append(Pose(self.skeleton, local_quats[i], root_pos[i]))
        return Motion(mirrored_poses, self.fps, str(self.__name) + "_mirrored")
