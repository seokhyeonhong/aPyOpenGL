import os

from pymovis import vis

class MyApp(vis.App):
    def __init__(self, filepath):
        super().__init__()
        self.heightmap = vis.Heightmap.load_from_file(filepath)
        self.heightmap = vis.Render.heightmap(self.heightmap).albedo(0.2).floor(True)

    def render(self):
        super().render()
        self.heightmap.draw()

if __name__ == "__main__":
    filepath = os.path.join(os.path.dirname(__file__), "../data/txt/heightmap.txt")
    vis.AppManager.start(MyApp(filepath))