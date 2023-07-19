import glm

from aPyOpenGL import agl

class MyApp(agl.App):
    def __init__(self):
        super().__init__()

    def start(self):
        super().start()
        self.sphere = agl.Render.sphere().albedo([0.2, 1, 0.2])
    
    def update(self):
        super().update()
        angle = self.frame * 0.01 * glm.pi()
        self.sphere.position([glm.cos(angle), 0.5, glm.sin(angle)])

    def render(self):
        super().render()
        self.sphere.draw()
        self.plane.draw()
    
    def render_text(self):
        super().render_text()
        agl.Render.text_on_screen(self.frame).draw()

if __name__ == "__main__":
    agl.AppManager.start(MyApp())