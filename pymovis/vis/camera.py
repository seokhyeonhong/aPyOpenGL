import glm
from pymovis.vis import glconst

class Camera:
    def __init__(
        self,
        position      =glm.vec3(5),
        orientation   =glm.mat3(1.0),
        focus_position=glm.vec3(0, 0, 0),
        fov_y         =glm.radians(45),
        is_perspective=True,
        ortho_zoom    =100.0,
        z_near        =0.1,
        z_far         =1000.0,
        zoom_factor   =1.0,
    ):
        self.position       = position
        self.orientation    = orientation
        self.focus_position = focus_position
        self.up             = glm.vec3(0, 1, 0)
        self.fov_y          = fov_y
        self.is_perspective = is_perspective
        self.ortho_zoom     = ortho_zoom
        self.z_near         = z_near
        self.z_far          = z_far
        self.zoom_factor    = zoom_factor

        self.update()
    
    def update(self):
        z = glm.normalize(self.focus_position - self.position)
        x = glm.normalize(glm.cross(self.up, z))
        self.orientation = glm.mat3(x, self.up, z)

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.focus_position, self.up)

    def get_projection_matrix(self, width, height):
        aspect = width / height
        if self.is_perspective:
            return glm.perspective(self.zoom_factor * self.fov_y, aspect, self.z_near, self.z_far)
        else:
            scale = self.ortho_zoom * 0.00001
            return glm.ortho(-width * scale, width * scale, -height * scale, height * scale, self.z_near, self.z_far)
    
    def dolly(self, yoffset):
        yoffset *= glconst.CAM_DOLLY_SENSITIVITY

        disp = self.orientation[2] * yoffset
        self.position += disp
        self.focus_position += disp
        
        self.update()
    
    def zoom(self, yoffset):
        yoffset *= glconst.CAM_ZOOM_SENSITIVITY

        if self.is_perspective:
            self.zoom_factor -= yoffset
            self.zoom_factor = max(0.1, min(self.zoom_factor, 10))
        else:
            self.ortho_zoom -= yoffset * 100
            self.ortho_zoom  = max(0.1, min(self.ortho_zoom, 1000))

        self.update()

    def tumble(self, dx, dy):
        dx *= glconst.CAM_TUMBLE_SENSITIVITY
        dy *= glconst.CAM_TUMBLE_SENSITIVITY

        disp = glm.vec4(self.position - self.focus_position, 1)
        alpha = 2.0
        Rx = glm.rotate(glm.mat4(1.0), alpha * dy, glm.vec3(glm.transpose(self.get_view_matrix())[0]))
        Ry = glm.rotate(glm.mat4(1.0), -alpha * dx, glm.vec3(0, 1, 0))
        R = Ry * Rx
        self.position = self.focus_position + glm.vec3(R * disp)
        self.up = glm.mat3(R) * self.up

        self.update()
    
    def track(self, dx, dy):
        dx *= glconst.CAM_TRACK_SENSITIVITY
        dy *= glconst.CAM_TRACK_SENSITIVITY

        VT = glm.transpose(self.get_view_matrix())
        self.position += glm.vec3(-dx * VT[0] - dy * VT[1])
        self.focus_position += glm.vec3(-dx * VT[0] - dy * VT[1])
        self.update()