import os

from pymovis.vis import App, AppManager, Render, FBX

class MyApp(App):
    def __init__(self, filename):
        super().__init__()
        self.model = Render.model(FBX(filename).model())
    
    def render(self):
        super().render()
        self.model.draw()

if __name__ == "__main__":
    filename  = os.path.join(os.path.dirname(__file__), "../data/fbx/model/ybot.fbx")
    AppManager.start(MyApp(filename))