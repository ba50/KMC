import sys

import click
import numpy as np

from vispy import scene
from vispy.visuals.transforms import STTransform

class HeatMap:
    def __init__(self, data_in):
        self.canvas = None

        array_temp = []
        with open(data_in) as heat_map_file:
            for line in heat_map_file:
                array_temp.append([int(word) for word in line.split()])

        self.array = np.zeros(max(array_temp)[:3])
        for pos in array_temp:
            self.array[pos[0]-1, pos[1]-1, pos[2]-1] = pos[3]

    @staticmethod
    def getRGBfromI(RGBint):
        RGBint = int(RGBint)
        blue =  RGBint & 255
        green = (RGBint >> 8) & 255
        red =   (RGBint >> 16) & 255
        return red/255, green/255, blue/255


    def plot(self):

        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='white',
                                   size=(640, 480), show=True)

        view = self.canvas.central_widget.add_view()
        view.camera = 'arcball'

        positions = []
        for pos in np.ndenumerate(self.array[:, 6:8, :]):
            positions.append(scene.visuals.Cube(size=0.2,
                color=self.getRGBfromI(pos[1]*(pow(2,24)-1)/self.array.max()),
                edge_color="black",
                parent=view.scene))

            positions[-1].transform = STTransform(translate=[pos[0][0]-self.array.shape[0]/2,
                pos[0][1]-self.array.shape[1]/2,
                pos[0][2]-self.array.shape[2]/2])

        view.camera.set_range(x=[-10, 10])


if __name__ == '__main__' and sys.flags.interactive == 0:
    @click.command()
    @click.option('--data_in', prompt="Heat map data in: ", help="Heat map data in: ") 
    def main(data_in):
        heat_map = HeatMap(data_in)
        heat_map.plot()
        heat_map.canvas.app.run()

    main()

