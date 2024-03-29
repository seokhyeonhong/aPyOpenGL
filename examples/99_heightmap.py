import os

from aPyOpenGL import agl

class MyApp(agl.App):
    def __init__(self, filepath):
        super().__init__()
        hmap = agl.Heightmap.load_from_file(filepath)
        self.heightmap = agl.Render.heightmap(hmap).albedo(0.2).floor(True)

    def start(self):
        super().start()
        self.grid.visible(False)

    def render(self):
        super().render()
        self.heightmap.draw()

if __name__ == "__main__":
    filepath = os.path.join(agl.AGL_PATH, "data/txt/heightmap.txt")
    agl.AppManager.start(MyApp(filepath))