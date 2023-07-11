import os

from pymovis import vis

class MyApp(vis.App):
    def __init__(self, filename):
        super().__init__()
        self.model = vis.FBX(filename).model()
    
    def render(self):
        super().render()
        vis.Render.model(self.model).draw()

if __name__ == "__main__":
    filename  = os.path.join(os.path.dirname(__file__), "../data/fbx/model/ybot.fbx")
    vis.AppManager.start(MyApp(filename))