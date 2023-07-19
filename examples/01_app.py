from aPyOpenGL import agl

class MyApp(agl.App):
    def start(self):
        super().start()
        self.plane   = agl.Render.plane().texture("brickwall.jpg", agl.TextureType.eALBEDO).texture("brickwall_normal.jpg", agl.TextureType.eNORMAL).texture("brickwall_disp.jpg", agl.TextureType.eDISPLACEMENT).position([0, 2, 0])
        self.cube    = agl.Render.cube().texture("brickwall.jpg", agl.TextureType.eALBEDO).texture("brickwall_normal.jpg", agl.TextureType.eNORMAL).position([1, 0.35, 2]).scale(0.7)
        self.sphere  = agl.Render.sphere().albedo([0.2, 1, 0.2]).position([-2, 0, 2]).scale(1.8)
        self.sphere2 = agl.Render.sphere().texture("pbr_albedo.png").texture("pbr_normal.png", agl.TextureType.eNORMAL).texture("pbr_metallic.png", agl.TextureType.eMETALIC).texture("pbr_roughness.png", agl.TextureType.eROUGHNESS).position([2, 1, 2])
        self.cone    = agl.Render.cone().albedo([0.2, 0.2, 1]).position([0, 0.5, 2])
        self.pyramid = agl.Render.pyramid().albedo([1, 1, 0]).position([0, 0, -1])
        self.text    = agl.Render.text("Hello, aPyOpenGL!")
        self.text_on_screen = agl.Render.text_on_screen("Hello, Screen!").position([0.01, 0.05, 0]).scale(1)
        # self.cubemap = Render.cubemap("skybox")
    
    def render(self):
        super().render()
        self.plane.draw()
        self.cube.draw()
        self.sphere.draw()
        self.sphere2.draw()
        self.cone.draw()
        self.pyramid.draw()

    def render_text(self):
        super().render_text()
        self.text.draw()
        self.text_on_screen.draw()

if __name__ == "__main__":
    agl.AppManager.start(MyApp())