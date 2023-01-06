import glm
from pymovis.vis import glconst

class Camera:
    def __init__(
        self,
        position      =glm.vec3(0, 5, 10),
        orientation   =glm.mat3(1.0),
        focus_position=glm.vec3(0, 0, 0),
        fov_y         =glm.radians(45),
        is_perspective=True,
        ortho_zoom    =100.0,
        z_near        =0.1,
        z_far         =1000.0,
        zoom_factor   =1.0,
    ):
        self.__position       = position
        self.__orientation    = orientation
        self.__focus_position = focus_position
        self.__up             = glm.vec3(0, 1, 0)
        self.__fov_y          = fov_y
        self.__is_perspective = is_perspective
        self.__ortho_zoom     = ortho_zoom
        self.__z_near         = z_near
        self.__z_far          = z_far
        self.__zoom_factor    = zoom_factor

        self.update()
    
    @property
    def position(self):
        return self.__position

    def update(self):
        z = glm.normalize(self.__focus_position - self.__position)
        x = glm.normalize(glm.cross(self.__up, z))
        self.__orientation = glm.mat3(x, self.__up, z)

    def get_view_matrix(self):
        return glm.lookAt(self.__position, self.__focus_position, self.__up)

    def get_projection_matrix(self, width, height):
        aspect = width / height
        if self.__is_perspective:
            return glm.perspective(self.__zoom_factor * self.__fov_y, aspect, self.__z_near, self.__z_far)
        else:
            scale = self.__ortho_zoom * 0.00001
            return glm.ortho(-width * scale, width * scale, -height * scale, height * scale, self.__z_near, self.__z_far)
    
    def dolly(self, yoffset):
        yoffset *= glconst.CAM_DOLLY_SENSITIVITY

        disp = self.__orientation[2] * yoffset
        self.__position += disp
        self.__focus_position += disp
        
        self.update()
    
    def zoom(self, yoffset):
        yoffset *= glconst.CAM_ZOOM_SENSITIVITY

        if self.__is_perspective:
            self.__zoom_factor -= yoffset
            self.__zoom_factor = max(0.1, min(self.__zoom_factor, 10))
        else:
            self.__ortho_zoom -= yoffset * 100
            self.__ortho_zoom  = max(0.1, min(self.__ortho_zoom, 1000))

        self.update()

    def tumble(self, dx, dy):
        dx *= glconst.CAM_TUMBLE_SENSITIVITY
        dy *= glconst.CAM_TUMBLE_SENSITIVITY

        disp = glm.vec4(self.__position - self.__focus_position, 1)
        alpha = 2.0
        Rx = glm.rotate(glm.mat4(1.0), alpha * dy, glm.vec3(glm.transpose(self.get_view_matrix())[0]))
        Ry = glm.rotate(glm.mat4(1.0), -alpha * dx, glm.vec3(0, 1, 0))
        R = Ry * Rx
        self.__position = self.__focus_position + glm.vec3(R * disp)
        self.__up = glm.mat3(R) * self.__up

        self.update()
    
    def track(self, dx, dy):
        dx *= glconst.CAM_TRACK_SENSITIVITY
        dy *= glconst.CAM_TRACK_SENSITIVITY

        VT = glm.transpose(self.get_view_matrix())
        self.__position += glm.vec3(-dx * VT[0] - dy * VT[1])
        self.__focus_position += glm.vec3(-dx * VT[0] - dy * VT[1])
        self.update()
    
    """ Camera manipulation functions """
    def set_focus_position(self, focus_position):
        self.__focus_position = glm.vec3(focus_position)
        self.update()
    
    def switch_projection(self):
        self.__is_perspective = not self.__is_perspective
        self.update()