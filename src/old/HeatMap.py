import sys

import click
import numpy as np

from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective, translate, rotate

vert = """
#version 120

// Uniforms
// ------------------------------------
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_linewidth;
uniform float u_antialias;
uniform float u_size;

// Attributes
// ------------------------------------
attribute vec3  a_position;
attribute vec4  a_fg_color;
attribute vec4  a_bg_color;
attribute float a_size;

// Varyings
// ------------------------------------
varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_size;
varying float v_linewidth;
varying float v_antialias;

void main (void) {
    v_size = a_size * u_size;
    v_linewidth = u_linewidth;
    v_antialias = u_antialias;
    v_fg_color  = a_fg_color;
    v_bg_color  = a_bg_color;
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
    gl_PointSize = v_size + 2*(v_linewidth + 1.5*v_antialias);
}
"""

frag = """
#version 120

// Constants
// ------------------------------------


// Varyings
// ------------------------------------
varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_size;
varying float v_linewidth;
varying float v_antialias;

// Functions
// ------------------------------------

// ----------------
float disc(vec2 P, float size)
{
    float r = length((P.xy - vec2(0.5,0.5))*size);
    r -= v_size/2;
    return r;
}

// ----------------
float arrow_right(vec2 P, float size)
{
    float r1 = abs(P.x -.50)*size + abs(P.y -.5)*size - v_size/2;
    float r2 = abs(P.x -.25)*size + abs(P.y -.5)*size - v_size/2;
    float r = max(r1,-r2);
    return r;
}

// ----------------
float ring(vec2 P, float size)
{
    float r1 = length((gl_PointCoord.xy - vec2(0.5,0.5))*size) - v_size/2;
    float r2 = length((gl_PointCoord.xy - vec2(0.5,0.5))*size) - v_size/4;
    float r = max(r1,-r2);
    return r;
}

// ----------------
float clober(vec2 P, float size)
{
    const float PI = 3.14159265358979323846264;
    const float t1 = -PI/2;
    const vec2  c1 = 0.2*vec2(cos(t1),sin(t1));
    const float t2 = t1+2*PI/3;
    const vec2  c2 = 0.2*vec2(cos(t2),sin(t2));
    const float t3 = t2+2*PI/3;
    const vec2  c3 = 0.2*vec2(cos(t3),sin(t3));

    float r1 = length((gl_PointCoord.xy- vec2(0.5,0.5) - c1)*size);
    r1 -= v_size/3;
    float r2 = length((gl_PointCoord.xy- vec2(0.5,0.5) - c2)*size);
    r2 -= v_size/3;
    float r3 = length((gl_PointCoord.xy- vec2(0.5,0.5) - c3)*size);
    r3 -= v_size/3;
    float r = min(min(r1,r2),r3);
    return r;
}

// ----------------
float square(vec2 P, float size)
{
    float r = max(abs(gl_PointCoord.x -.5)*size,
                  abs(gl_PointCoord.y -.5)*size);
    r -= v_size/2;
    return r;
}

// ----------------
float diamond(vec2 P, float size)
{
    float r = abs(gl_PointCoord.x -.5)*size + abs(gl_PointCoord.y -.5)*size;
    r -= v_size/2;
    return r;
}

// ----------------
float vbar(vec2 P, float size)
{
    float r1 = max(abs(gl_PointCoord.x -.75)*size,
                   abs(gl_PointCoord.x -.25)*size);
    float r3 = max(abs(gl_PointCoord.x -.5)*size,
                   abs(gl_PointCoord.y -.5)*size);
    float r = max(r1,r3);
    r -= v_size/2;
    return r;
}

// ----------------
float hbar(vec2 P, float size)
{
    float r2 = max(abs(gl_PointCoord.y -.75)*size,
                   abs(gl_PointCoord.y -.25)*size);
    float r3 = max(abs(gl_PointCoord.x -.5)*size,
                   abs(gl_PointCoord.y -.5)*size);
    float r = max(r2,r3);
    r -= v_size/2;
    return r;
}

// ----------------
float cross(vec2 P, float size)
{
    float r1 = max(abs(gl_PointCoord.x -.75)*size,
                   abs(gl_PointCoord.x -.25)*size);
    float r2 = max(abs(gl_PointCoord.y -.75)*size,
                   abs(gl_PointCoord.y -.25)*size);
    float r3 = max(abs(gl_PointCoord.x -.5)*size,
                   abs(gl_PointCoord.y -.5)*size);
    float r = max(min(r1,r2),r3);
    r -= v_size/2;
    return r;
}


// Main
// ------------------------------------
void main()
{
    float size = v_size +2*(v_linewidth + 1.5*v_antialias);
    float t = v_linewidth/2.0-v_antialias;

    float r = disc(gl_PointCoord, size);
    // float r = square(gl_PointCoord, size);
    // float r = ring(gl_PointCoord, size);
    // float r = arrow_right(gl_PointCoord, size);
    // float r = diamond(gl_PointCoord, size);
    // float r = cross(gl_PointCoord, size);
    // float r = clober(gl_PointCoord, size);
    // float r = hbar(gl_PointCoord, size);
    // float r = vbar(gl_PointCoord, size);


    float d = abs(r) - t;
    if( r > (v_linewidth/2.0+v_antialias))
    {
        discard;
    }
    else if( d < 0.0 )
    {
       gl_FragColor = v_fg_color;
    }
    else
    {
        float alpha = d/v_antialias;
        alpha = exp(-alpha*alpha);
        if (r > 0)
            gl_FragColor = vec4(v_fg_color.rgb, alpha*v_fg_color.a);
        else
            gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
    }
}
"""


class HeatMap(app.Canvas):
    def __init__(self, data_in):
        app.Canvas.__init__(self, keys='interactive', size=(640, 480))
        ps = self.pixel_scale

        array = []
        with open(data_in) as heat_map_file:
            for line in heat_map_file:
                array.append([int(word) for word in line.split()])

        array = np.array(array) - 1

        dim = np.max(array[:, :3])+1
        heat_map = np.zeros((dim, dim, dim))
        for pos in array:
            heat_map[pos[0], pos[1], pos[2]] = pos[3]

        mean_map = heat_map.mean(axis=2).mean(axis=1)

        """
        array = self.clip(array, 3, 1, 6)
        array = self.clip(array, 0, 5, 9)
        """

        positions = []
        color = []
        for pos in array:
            positions.append(pos[:3])
            #color.append(self.getRGBfromI(pos[3]*(pow(2, 24)-1)/array.max()))
            color.append(self.getRGBfromI(mean_map[pos[2]]*(pow(2,
                                                                24)-1)/mean_map.max()))

        positions = np.array(positions)
        positions = positions-positions.max()/2
        color = np.array(color)

        # Create vertices
        n = len(array)
        data = np.zeros(n, [('a_position', np.float32, 3),
                            ('a_bg_color', np.float32, 4),
                            ('a_fg_color', np.float32, 4),
                            ('a_size', np.float32, 1)])

        data['a_position'] = positions
        data['a_bg_color'] = color
        data['a_fg_color'] = 0, 0, 0, 1
        data['a_size'] = 75*ps
        u_linewidth = 1.0
        u_antialias = 1.0

        self.translate = 75
        self.program = gloo.Program(vert, frag)
        self.view = translate((0, 0, -self.translate))
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.apply_zoom()

        self.program.bind(gloo.VertexBuffer(data))
        self.program['u_linewidth'] = u_linewidth
        self.program['u_antialias'] = u_antialias
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.program['u_size'] = 10 / self.translate

        self.theta = 0
        self.phi = 0

        gloo.set_state('translucent', clear_color='white')

        self.timer = app.Timer('auto', connect=self.on_timer, start=True)

        self.show()
        self.canvas = None

    @staticmethod
    def clip(array, index, min_, max_):
        temp_array = []
        for pos in array:
            if pos[index] >= min_ and pos[index] < max_:
                temp_array.append(pos)
        return np.array(temp_array)

    @staticmethod
    def getRGBfromI(RGBint):
        RGBint = int(RGBint)
        blue =  RGBint & 255
        green = (RGBint >> 8) & 255
        red =   (RGBint >> 16) & 255
        return red/255, green/255, blue/255, 1

    def on_key_press(self, event):
        if event.text == ' ':
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()

    def on_timer(self, event):
        self.theta += .1
        self.phi += .1
        self.model = np.dot(rotate(self.theta, (0, 0, 1)),
                            rotate(self.phi, (0, 1, 0)))
        self.program['u_model'] = self.model
        self.update()

    def on_resize(self, event):
        self.apply_zoom()

    def on_mouse_wheel(self, event):
        self.translate -= event.delta[1]
        self.translate = max(2, self.translate)
        self.view = translate((0, 0, -self.translate))

        self.program['u_view'] = self.view
        self.program['u_size'] = 10 / self.translate
        self.update()

    def on_draw(self, event):
        gloo.clear()
        self.program.draw('points')

    def apply_zoom(self):
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 1.0, 1000.0)
        self.program['u_projection'] = self.projection


if __name__ == '__main__' and sys.flags.interactive == 0:
    @click.command()
    @click.option('--data_in', prompt="Heat map data in: ", help="Heat map data in.") 
    def main(data_in):
        heat_map = HeatMap(data_in)
        app.run()

    main()

