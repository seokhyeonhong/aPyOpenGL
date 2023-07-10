import glm

from pymovis import vis

class MyApp(vis.AnimApp):
    def __init__(self):
        super().__init__()
        self.total_frames = 100

    def start(self):
        super().start()
        self.sphere = vis.Render.sphere().albedo([0.2, 1, 0.2])
    
    def update(self):
        super().update()
        angle = self.curr_frame / self.total_frames * 2 * glm.pi()
        self.sphere.position([glm.cos(angle), 0.5, glm.sin(angle)])

    def render(self):
        super().render()
        self.sphere.draw()

if __name__ == "__main__":
    vis.AppManager.start(MyApp())