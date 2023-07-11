import os

from pymovis import vis

class MyApp(vis.App):
    def __init__(self, fbx_filename):
        super().__init__()
        self.fbx_filename = fbx_filename

    def start(self):
        super().start()
        self.fbx_model = vis.FBX(self.fbx_filename).model()

    def render(self):
        super().render()
        vis.Render.plane(150, 150).floor(True).albedo(0.2).draw()
        vis.Render.model(self.fbx_model).draw()
    
    def render_xray(self):
        super().render_xray()
        vis.Render.skeleton(self.fbx_model).draw()

if __name__ == "__main__":
    fbx_filename = os.path.join(vis.PYMOVIS_PATH, "../../data/fbx/model/ybot.fbx")
    vis.AppManager.start(MyApp(fbx_filename))