import time
import glm

from pymovis.vis import App, AppManager, Render

class MyApp(App):
    def __init__(self):
        super().__init__()
        self.plane = Render.plane().set_texture("grid.png").set_scale(50).set_uv_repeat(5)
        self.plane2 = Render.cube().set_texture("brickwall.jpg").set_texture("brickwall_normal.jpg", "normal").set_position([0, 1, -2])
        # self.axis = Render.axis()
        self.cube = Render.cube().set_texture("brickwall.jpg").set_texture("brickwall_normal.jpg", "normal").set_position([1, 0.35, 2]).set_scale(0.7)
        # self.sphere = Render.sphere().set_albedo([0.2, 1, 0.2]).set_position([-2, 0, 2]).set_scale(1.8)
        # self.cone = Render.cone().set_albedo([0.2, 0.2, 1]).set_position([0, 0.5, 2])
        # self.pyramid = Render.pyramid().set_albedo([1, 1, 0]).set_position([0, 0.5, 2])
        self.text = Render.text("Hello, PyMoVis!")
        # self.text_on_screen = Render.text_on_screen("Hello, Screen!").set_scale(1)
    
    def render(self):
        super().render()
        self.plane.draw()
        self.plane2.set_orientation(glm.angleAxis(glm.pi() * (time.time() % 1.0), glm.vec3(1, 0, 0))).draw()
        # self.axis.draw()
        self.cube.draw()
        # self.sphere.set_scale(glm.sin((time.time() * 0.3 % 1.0) * glm.pi()) + 1.0).draw()
        # self.cone.set_position([glm.cos((time.time() * 0.5 % 1.0) * 2 * glm.pi()), 0.5, glm.sin((time.time() * 0.5 % 1.0) * 2 * glm.pi())]).draw()
        # self.pyramid.draw()
        # self.text.draw()
        # self.text_on_screen.draw()

if __name__ == "__main__":
    # app cycle manager
    app_manager = AppManager()
    
    # create app
    app = MyApp()

    # run app
    app_manager.run(app)