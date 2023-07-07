import os

from pymovis import AppManager, MotionApp, FBX

if __name__ == "__main__":
    # app cycle manager
    app_manager = AppManager()

    # load data
    model_fbx = FBX(os.path.join(os.path.dirname(__file__), "../data/fbx/model/ybot.fbx"))
    motion_fbx = FBX(os.path.join(os.path.dirname(__file__), "../data/fbx/motion/ybot_capoeira.fbx"))

    # joint relation dict: identity
    joint_dict = {}
    for joint in model_fbx.skeleton().joints:
        joint_dict[joint.name] = joint.name

    # create app
    app = MotionApp(motion_fbx.motion()[0], model_fbx.model(), joint_dict)

    # run app
    app_manager.run(app)