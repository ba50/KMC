import numpy as np
from vispy import gloo
from vispy.util.transforms import perspective, translate, rotate


class PlotModel:
    def __init__(self, app, positions, atoms_number=2, resolution=(800,600)):
        self.resolution = resolution
        self.positions = positions #- positions[:1].mean()
        self.atoms_number = atoms_number

        self.mouse_press_point = 0, 0
        self.mouse_press = False
        self.theta = 0
        self.phi = 0
        self.delta_theta = 0
        self.delta_phi = 0
        self.animation_step = 10

        self.translate = 50
        self.view = translate((0, 0, -self.translate), dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        gloo.set_viewport(0, 0, app.physical_size[0], app.physical_size[1])
        self.projection = perspective(45.0, app.size[0] /
                                      float(app.size[1]), 1.0, 1000.0)

        self.alpha = np.array((0, 0, 0, 0)).astype(np.float32).reshape(1, 4)
        self.a_color = [np.append(np.random.rand(3), 1).astype(np.float32).reshape(1,4) for i in range(self.atoms_number)]

        self.timer = app.Timer('auto', connect=self.on_timer)

    # ---------------------------------
    def on_timer(self, event):
        self.phi += np.clip(self.delta_phi, -1, 1)
        self.theta += np.clip(self.delta_theta, -1, 1)
        self.model = np.dot(rotate(self.theta, (0, 0, 1)),
                            rotate(self.phi, (0, 1, 0)))

    # ---------------------------------
    def on_resize(self, event):
        self.resolution = event.physical_size
        gloo.set_viewport(0, 0, self.resolution[0], self.resolution[1])
        self.projection = perspective(45.0, event.size[0] /
                                      float(event.size[1]), 1.0, 1000.0)

    # ---------------------------------
    def on_mouse_wheel(self, event):
        self.translate += event.delta[1]
        self.translate = max(2, self.translate)
        self.view = translate((0, 0, -self.translate))

    # ---------------------------------
    def on_draw(self, event):
        self.context.clear()

    # ---------------------------------
    def on_mouse_move(self, event):
        if self.mouse_press:
            speed = .001
            self.delta_phi += (event.pos[0] - self.mouse_press_point[0])*speed
            self.delta_theta += (event.pos[1] - self.mouse_press_point[1])*speed

    # ---------------------------------
    def on_mouse_press(self, event):
        self.mouse_press = True
        self.mouse_press_point = event.pos

    # ---------------------------------
    def on_mouse_release(self, event):
        self.mouse_press = False
        self.delta_phi = 0
        self.delta_theta = 0
