import os

from pymovis import vis

class MyApp(vis.App):
    def __init__(self, fbx_filename):
        super().__init__()
        self.fbx_filename = fbx_filename

    def start(self):
        super().start()
        self.plane     = vis.Render.plane(150, 150).floor(True).albedo(0.2)
        self.fbx_model = vis.FBX(self.fbx_filename).model()

        self.model    = vis.Render.model(self.fbx_model)
        self.skeleton = vis.Render.skeleton(self.fbx_model)
    
    def update(self):
        super().update()
        
    def render(self):
        super().render()
        self.plane.draw()
        self.model.draw()
    
    def render_xray(self):
        super().render_xray()
        self.skeleton.draw()

if __name__ == "__main__":
    fbx_filename = os.path.join(vis.PYMOVIS_PATH, "../../data/fbx/model/ybot.fbx")
    vis.AppManager.start(MyApp(fbx_filename))