import os

from aPyOpenGL import agl

class MyApp(agl.App):
    def __init__(self, fbx_filename):
        super().__init__()
        self.fbx_filename = fbx_filename

    def start(self):
        super().start()
        self.fbx_model = agl.FBX(self.fbx_filename).model()

    def render(self):
        super().render()
        agl.Render.plane(150, 150).floor(True).albedo(0.2).draw()
        agl.Render.model(self.fbx_model).draw()
    
    def render_xray(self):
        super().render_xray()
        agl.Render.skeleton(self.fbx_model).draw()

if __name__ == "__main__":
    fbx_filename = os.path.join(agl.AGL_PATH, "data/fbx/model/ybot.fbx")
    agl.AppManager.start(MyApp(fbx_filename))