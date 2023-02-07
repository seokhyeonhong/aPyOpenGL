from pymovis.vis.heightmap import Heightmap
from pymovis.vis.render import Render
from pymovis.vis.appmanager import AppManager
from pymovis.vis.app import App

class HeightApp(App):
    def __init__(self, heightmap):
        super().__init__()
        self.heightmap = heightmap
        self.mesh = Render.vao(heightmap.vao).set_texture("grid.png").set_uv_repeat(0.1)
    
    def render(self):
        super().render()
        self.mesh.draw()
        p = self.heightmap.sample(0, 0)

def main():
    # app cycle manager
    app_manager = AppManager()

    # load heightmap
    heightmap = Heightmap.load_from_file("data/heightmap.txt")

    # create app
    app = HeightApp(heightmap)

    # run app
    app_manager.run(app)

if __name__ == "__main__":
    main()