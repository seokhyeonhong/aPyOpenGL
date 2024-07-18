from __future__ import annotations
import numpy as np
import copy
import os

from scipy.spatial.transform import Rotation
from .pose import Pose
from aPyOpenGL.transforms import n_quat
from aPyOpenGL import agl

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
        self.poses : list[Pose] = poses
        self.fps   : float      = fps
        self.name  : str        = name

    def __len__(self):
        return len(self.poses)
    
    @property
    def num_frames(self):
        return len(self.poses)
    
    @property
    def skeleton(self):
        return self.poses[0].skeleton
    
    def remove_joint_by_name(self, joint_name):
        remove_indices = self.skeleton.remove_joint_by_name(joint_name)
        for pose in self.poses:
            pose.skeleton = self.skeleton
            pose.local_quats = np.delete(pose.local_quats, remove_indices, axis=0)

    def export_as_bvh(self):
        total_frames = self.num_frames
        filename = os.path.join(agl.AGL_PATH, "data/bvh/ybot_capoeira_export.bvh")
        self._save(filename)
        

    def _save(self, filename, scale=100.0, rot_order="ZXY", verbose=False):
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
                pose = self.poses[i]

                p = self.poses[i].root_pos
                p *= scale
                f.write("%f %f %f " % (p[0], p[1], p[2]))
                for quat in self.poses[i].local_quats:
                    R = self._Q2E(quat, rot_order)
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

    def _Q2E(self, Q, order="yxz", degrees=True):
        w = Q[0]
        x = Q[1]
        y = Q[2]
        z = Q[3]
        modifiedQ = [x,y,z,w]

        return Rotation.from_quat(modifiedQ).as_euler(order, degrees=degrees)