import glm

from pymovis.vis import App, AppManager, Render, TextureType

class MyApp(App):
    def start(self):
        super().start()
        self.plane   = Render.plane(150, 150).floor(True).albedo(0.2)
        self.plane2  = Render.plane().texture("brickwall.jpg", TextureType.eALBEDO).texture("brickwall_normal.jpg", TextureType.eNORMAL).texture("brickwall_disp.jpg", TextureType.eDISPLACEMENT).position([0, 2, 0])
        self.axis    = Render.axis()
        self.cube    = Render.cube().texture("brickwall.jpg", TextureType.eALBEDO).texture("brickwall_normal.jpg", TextureType.eNORMAL).position([1, 0.35, 2]).scale(0.7)
        self.sphere  = Render.sphere().albedo([0.2, 1, 0.2]).position([-2, 0, 2]).scale(1.8)
        self.sphere2 = Render.sphere().texture("pbr_albedo.png").texture("pbr_normal.png", TextureType.eNORMAL).texture("pbr_metallic.png", TextureType.eMETALIC).texture("pbr_roughness.png", TextureType.eROUGHNESS).position([2, 1, 2])
        self.cone    = Render.cone().albedo([0.2, 0.2, 1]).position([0, 0.5, 2])
        self.pyramid = Render.pyramid().albedo([1, 1, 0]).position([0, 0, -1])
        self.text    = Render.text("Hello, aPyOpenGL!")
        self.text_on_screen = Render.text_on_screen("Hello, Screen!").scale(1)
        # self.cubemap = Render.cubemap("skybox")
    
    def render(self):
        super().render()
        self.plane.draw()
        self.plane2.draw()
        self.axis.draw()
        self.cube.draw()
        self.sphere.draw()
        self.sphere2.draw()
        self.cone.draw()
        self.pyramid.draw()
        # self.cubemap.draw()

    def render_text(self):
        super().render_text()
        self.text.draw()
        self.text_on_screen.draw()

if __name__ == "__main__":
    AppManager.start(MyApp())