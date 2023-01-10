import glm

class Light:
    def __init__(
        self,
        focus_position: glm.vec3 = glm.vec3(0),
        color: glm.vec3 =glm.vec3(1.0),
        intensity: float = 1.0,
        L: float = 5.0,
        z_near: float = 1.0,
        z_far: float = 40.0
    ):
        self.__focus_position = focus_position
        self.__color = color
        self.__intensity = intensity
        self.__L = L
        self.__z_near = z_near
        self.__z_far = z_far
    
    @property
    def vector(self):
        raise NotImplementedError
    
    @property
    def position(self):
        return NotImplementedError
    
    @property
    def focus_position(self):
        return self.__focus_position
    
    @property
    def color(self):
        return self.__color

    @property
    def intensity(self):
        return self.__intensity

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.__focus_position, glm.vec3(0, 1, 0))
    
    def get_projection_matrix(self):
        return glm.ortho(-self.__L, self.__L, -self.__L, self.__L, self.__z_near, self.__z_far)
    
    def get_view_projection_matrix(self):
        return self.get_projection_matrix() * self.get_view_matrix()

class DirectionalLight(Light):
    def __init__(
        self,
        direction: glm.vec3 = glm.vec3(-1, -2, -1),
        focus_position: glm.vec3 = glm.vec3(0),
        color: glm.vec3 = glm.vec3(1.0),
        intensity: float = 1.0,
        L: float = 5.0,
        z_near: float = 1.0,
        z_far: float = 40.0
    ):
        self.__direction   = direction
        super().__init__(focus_position, color, intensity, L, z_near, z_far)

    @property
    def position(self):
        return self.focus_position - glm.normalize(self.__direction) * 10

    @property
    def attenuation(self):
        # Not used actually, but for completeness with other lights
        return glm.vec3(1.0, 0.0, 0.0)

    @property
    def vector(self):
        return glm.vec4(self.__direction, 0)
    

class PointLight(Light):
    def __init__(
        self,
        position: glm.vec3 = glm.vec3(5),
        focus_position: glm.vec3 = glm.vec3(0),
        color: glm.vec3 = glm.vec3(1.0),
        intensity: float = 1.0,
        attenuation: glm.vec3 = glm.vec3(0.1, 0.01, 0.0),
        L: float = 5.0,
        z_near: float = 1.0,
        z_far: float = 40.0
    ):
        self.__position    = position
        self.__attenuation = attenuation
        super().__init__(focus_position, color, intensity, L, z_near, z_far)
    
    @property
    def position(self):
        return self.__position

    @property
    def attenuation(self):
        return self.__attenuation

    @property
    def vector(self):
        return glm.vec4(self.position, 1)