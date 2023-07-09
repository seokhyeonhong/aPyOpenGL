import os

from pymovis.vis import App, AppManager, Render, FBX

class MyApp(App):
    def __init__(self, fbx_filename):
        super().__init__()
        self.fbx_filename = fbx_filename

    def start(self):
        super().start()
        self.plane = Render.plane(150, 150).floor(True).albedo(0.2)
        self.fbx_model = FBX(self.fbx_filename).model()

        self.model    = Render.model(self.fbx_model)
        self.skeleton = Render.skeleton(self.fbx_model)
    
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
    fbx_filename = os.path.join(os.path.dirname(__file__), "../data/fbx/model/ybot.fbx")
    AppManager.start(MyApp(fbx_filename))