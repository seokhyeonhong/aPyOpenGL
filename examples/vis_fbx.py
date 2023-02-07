from pymovis.motion.data.fbx import FBX
from pymovis.vis.app import App
from pymovis.vis.appmanager import AppManager
from pymovis.vis.render import Render

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
    fbx = FBX("data/character.fbx")

    # create app
    app = MyApp(fbx.model())

    # run app
    app_manager.run(app)