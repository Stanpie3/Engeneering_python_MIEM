import numpy as np
import ffmpeg
from datetime import datetime
from vispy import app, scene
from funcs import *

#%%

app.use_app('pyglet')

asp = 16 / 9
h = 720
w = int(asp * h)

canvas = scene.SceneCanvas(keys='interactive',
                           show=True,
                           size=(w, h))

view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=(0, 0, asp, 1),
                                  aspect=1)


face = "Times New Roman"
#%%

video = False
N = 1000
dt = 0.05
perception = 1./20
vrange = np.array([0.05, 0.1])
arange = np.array([0., 1.0])

coeffs = np.array([1.0,  # alignment
                   1.0,  # cohesion
                   0.05,  # separation
                   0.001,  # walls
                   0.0   # noise
                   ])

# x, y, vx, vy, ax, ay
boids = np.zeros((N, 6), dtype=np.float64)
D = np.zeros((N, N), dtype=np.float64)
init_boids(boids, asp, vrange)
# nb0 = calc_neighbors(boids, perception)

arrows = \
scene.Arrow(arrows=directions(boids),
            arrow_color=(1, 1, 1, 1),
            arrow_size=5,
            connect='segments',
            parent=view.scene
            )

scene.Line(pos=np.array([[0, 0],
                         [asp, 0],
                         [asp, 1],
                         [0, 1],
                         [0, 0]
                         ]),
           color=(1, 0, 0, 1),
           connect='strip',
           method='gl',
           parent=view.scene
           )
#%%

if video:
    fname = f"boids_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
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
    global process, boids, D
    simulate(boids, D, perception, asp, coeffs)
    propagate(boids, dt, vrange)
    periodic_walls(boids, asp)
    arrows.set_data(arrows=directions(boids))
    if video:
        frame = canvas.render(alpha=False)
        process.stdin.write(frame.tobytes())
    else:
        canvas.update(event)
    print(f"{canvas.fps:0.1f}")

#%%

timer = app.Timer(interval=0, start=True, connect=update)

if __name__ == '__main__':
    canvas.measure_fps()
    app.run()
    if video:
        process.stdin.close()
        process.wait()