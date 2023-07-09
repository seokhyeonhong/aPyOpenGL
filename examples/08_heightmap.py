import os

from pymovis.vis import App, AppManager, Render, Heightmap

class MyApp(App):
    def __init__(self, filepath):
        super().__init__()
        self.heightmap = Heightmap.load_from_file(filepath)
        self.heightmap = Render.heightmap(self.heightmap).albedo(0.2).floor(True)

    def render(self):
        super().render()
        self.heightmap.draw()

if __name__ == "__main__":
    filepath = os.path.join(os.path.dirname(__file__), "../data/txt/heightmap.txt")
    AppManager.start(MyApp(filepath))