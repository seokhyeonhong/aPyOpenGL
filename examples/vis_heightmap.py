from pymovis.vis import Heightmap, Render, App, AppManager

class HeightApp(App):
    def __init__(self, heightmap):
        super().__init__()
        self.heightmap = heightmap
        self.mesh = Render.heightmap(heightmap).set_albedo(0.2).set_floor(True).set_background(0.0)
    
    def render(self):
        super().render()
        self.mesh.draw()

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