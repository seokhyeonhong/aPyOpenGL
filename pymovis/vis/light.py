import glm

class Light:
    def __init__(
        self,
        focus_position=glm.vec3(0),
        color=glm.vec3(1.0),
        intensity=1.0,
        L=5,
        z_near=1.0,
        z_far=40.0
    ):
        self.focus_position = focus_position
        self.color = color
        self.intensity = intensity
        self.L = L
        self.z_near = z_near
        self.z_far = z_far
    
    @property
    def vector(self):
        raise NotImplementedError

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.focus_position, glm.vec3(0, 1, 0))
    
    def get_projection_matrix(self):
        return glm.ortho(-self.L, self.L, -self.L, self.L, self.z_near, self.z_far)
    
    def get_view_projection_matrix(self):
        return self.get_projection_matrix() * self.get_view_matrix()

class DirectionalLight(Light):
    """
    Basic directional light.
    """
    def __init__(
        self,
        direction=glm.vec3(-1),
        focus_position=glm.vec3(0),
        color=glm.vec3(1.0),
        intensity=1.0,
        L=5,
        z_near=1.0,
        z_far=40.0
    ):
        self.direction = direction
        self.attenuation = glm.vec3(1.0, 0.0, 0.0) # Not used actually, but for completeness with other lights
        super().__init__(focus_position, color, intensity, L, z_near, z_far)

    @property
    def position(self):
        return self.focus_position - glm.normalize(self.direction) * 10

    @property
    def vector(self):
        return glm.vec4(self.direction, 0)
    

class PointLight(Light):
    """
    Basic directional light.
    """
    def __init__(
        self,
        position=glm.vec3(5),
        focus_position=glm.vec3(0),
        color=glm.vec3(1.0),
        intensity=1.0,
        attenuation=glm.vec3(0.1, 0.01, 0.0),
        L=5,
        z_near=1.0,
        z_far=40.0
    ):
        self.position    = position
        self.attenuation = attenuation
        super().__init__(focus_position, color, intensity, L, z_near, z_far)
    
    @property
    def vector(self):
        return glm.vec4(self.position, 1)