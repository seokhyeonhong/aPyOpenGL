import os

from aPyOpenGL import agl

class MyApp(agl.App):
    def __init__(self, filename):
        super().__init__()
        self.model = agl.FBX(filename).model()
    
    def render(self):
        super().render()
        agl.Render.model(self.model).draw()

if __name__ == "__main__":
    filename  = os.path.join(agl.AGL_PATH, "data/fbx/model/ybot.fbx")
    agl.AppManager.start(MyApp(filename))