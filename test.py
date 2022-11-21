from pymovis.vis.app import App
from pymovis.vis.appmanager import AppManager
from pymovis.vis.render import Render
from OpenGL.GL import *
import glm

class MyApp(App):
    def render(self):
        # t = glm.vec3(glm.sin(glfw.get_time()))
        # Render.sphere().set_scale(.5).set_material(glm.vec3(1, 0,0)).draw()
        Render.plane().set_texture("example.png").set_scale(50).set_uv_repeat(5).draw()
        Render.sphere().set_texture("wood.jpg").draw()
        Render.cube().set_position(1, 0.5, 1).draw()

if __name__ == "__main__":
    app = MyApp()
    AppManager.run(app)