import os

from pymovis.vis import AppManager, App, Render, FBX

class MyApp(App):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def render(self):
        Render.model(self.model).draw()

if __name__ == "__main__":
    # app cycle manager
    app_manager = AppManager()

    # load data
    filepath = os.path.join(os.path.dirname(__file__), "../data/fbx/model/ybot.fbx")
    fbx = FBX(filepath)

    # create app
    app = MyApp(fbx.model())

    # run app
    app_manager.run(app)