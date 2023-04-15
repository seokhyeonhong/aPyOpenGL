from pymovis.motion.data.fbx import FBX
from pymovis.vis import App, AppManager, Render

class MyApp(App):
    def __init__(self, model):
        super().__init__()
        self.model = Render.model(model)
        self.plane = Render.plane().set_albedo([0.5, 0.5, 0.5]).set_scale(20)
    
    def render(self):
        self.model.draw()
        self.plane.draw()

if __name__ == "__main__":
    # app cycle manager
    app_manager = AppManager()

    # load data
    fbx = FBX("data/ybot.fbx")

    # create app
    app = MyApp(fbx.model())

    # run app
    app_manager.run(app)