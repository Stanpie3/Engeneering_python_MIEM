import numpy as np
from vispy import app, scene
from vispy.scene.visuals import Text
from funcs_njit import *
import ffmpeg
from datetime import datetime
from math import floor, ceil
from numba import njit, prange
from numba.core import types
from numba.typed import Dict
#%%
np.seterr(divide='ignore', invalid='ignore')

app.use_app('pyglet')
asp = 16.0/9
h = 1080
w = int(h * asp)

canvas = scene.SceneCanvas(keys='interactive', show=True, size=(w, h))
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=(0, 0, asp, 1), aspect=1)

#%%

video = False
N = 1500
dt = 0.01
perception = 1.0 / 20
vrange = np.array([0.05, 0.1])


coeffs = np.array([1.0,  # alignment
                   0.2,  # cohesion
                   0.1,  # separation
                   0.0,  # noise
                   0.0001 # walls
                   ])

#%%

boids = np.zeros((N, 6), dtype=np.float64)
init_boids(boids, asp, vrange)

#%%

width = 36
num_cells = width**2 * asp
cell_to_boids = Dict.empty(key_type=types.int64, value_type=types.int64[:])
cell_size = w * h * 1.0 / width**2

#%%

fr = 0
arrows = scene.Arrow(arrows=directions(boids), arrow_color=(1, 1, 1, 1),
                     arrow_size=5, connect='segments', parent=view.scene)


scene.Line(pos=np.array([[0, 0], [asp, 0], [asp, 1], [0, 1], [0, 0]]), color=(1, 0, 0, 1),
                        connect='strip', method='gl', parent=view.scene)
if h == 1080:
    txt = Text(parent=canvas.scene, color='red', face='Consolas')
    txt.pos = canvas.size[0] // 16, canvas.size[1] // 35
    txt.font_size = 12
    txt_const = Text(parent=canvas.scene, color='red', face='Consolas')
    txt_const.pos = canvas.size[0] // 16, canvas.size[1] // 10
    txt_const.font_size = 10
    general_info = "boids: " + str(N) + "\n" +  "alignment: " + str(coeffs[0]) + "\n" + "cohesion: " + str(coeffs[1]) + "\n" + "separation: " + str(coeffs[2]) + "\n" + "noise: " + str(coeffs[3]) + "\n" + "walls: " + str(coeffs[4])
    txt_const.text = general_info
else:
    txt = Text(parent=canvas.scene, color='green', face='Consolas')
    txt.pos = canvas.size[0] // 12, canvas.size[1] // 30
    txt.font_size = 12
    txt_const = Text(parent=canvas.scene, color='green', face='Consolas')
    txt_const.pos = canvas.size[0] // 12, canvas.size[1] // 7
    txt_const.font_size = 10
    general_info = "boids: " + str(N) + "\n" + "alignment: " + str(coeffs[0]) + "\n" + "cohesion: " + str(coeffs[1]) + "\n" + "separation: " + str(coeffs[2]) + "\n" + "noise: " + str(coeffs[3]) + "\n" + "walls: " + str(coeffs[4])
    txt_const.text = general_info

#%%
if video:
    fname = f"boids_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
    print(fname)

    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f"{w}x{h}", r=60)
            .output(fname, pix_fmt='yuv420p', preset='slower', r=60)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
#%%


def update(event):
    global process, boids, fr, txt
    simulate_hashed(boids, perception, cell_to_boids, asp, coeffs, width)
    propagate(boids, dt, vrange)
    periodic_walls(boids, asp)
    cell_to_boids.clear()
    arrows.set_data(arrows=directions(boids))
    if fr % 30 == 0:
        txt.text = "fps:" + f"{canvas.fps:0.1f}"
    fr = fr + 1
    if video:
        if fr <= 3600:
            frame = canvas.render(alpha=False)
            process.stdin.write(frame.tobytes())
        else:
            app.quit()
    else:
        canvas.update(event)
    # print(f"{canvas.fps:0.1f}")


#%%
timer = app.Timer(interval=0, start=True, connect=update)

if __name__ == '__main__':
    canvas.measure_fps(window=5.0)
    app.run()
    if video:
        process.stdin.close()
        process.wait()
