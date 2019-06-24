#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def plot_setup(result):
    fig, axes = plt.subplots(1,1,figsize=(8,8))
    return fig, axes

def plot_result(result, n, fig=None, axes=None):
    if fig is None or axes is None:
        fig, axes = plot_setup(result)
    axes.clear()
    axes.plot(result[0][n],result[1][n], 'k.', linewidth=2)
    axes.set_xlim(-6,6)
    axes.set_ylim(-6,6)
    return fig, axes

import base64
def plot_animation(plot_setup_func, plot_func, result, name="movie",fps=10,
                   writer="avconv", codec="libx264", verbose=False):
    """
    Create an animated plot of a Result object, as returned by one of
    the qutip evolution solvers.
    .. note :: experimental
    """

    fig, axes = plot_setup_func(result)

    def update(n):
        return plot_func(result, n, fig=fig, axes=axes)
    ims = []
    for i in range(len(result[0])):
        fig, axes = plot_func(result, i, fig, axes)
        ims.append([axes])
    #anim = animation.FuncAnimation(
    #    fig, update, frames=len(result[0]), blit=True)
    anim = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
    anim.save(name, fps=fps, writer=writer, codec=codec)

    plt.close(fig)

    #if verbose:
    #    print("Created %s" % name)

    video = open(name, "rb").read()
    video_encoded = base64.b64encode(video).decode("ascii")
    video_tag = '<video controls src="data:video/webm;base64,{0}">'.format(
       video_encoded, 'webm')
    #video_tag = '<img src="data:image/gif;base64,{0}" />'.format(video_encoded)
    return HTML(video_tag)
    
xc_list = expect((a + a.dag()), res.states)
yc_list = expect(-1j*(a - a.dag()), res.states)
result = (xc_list, yc_list)
#f, ax = plot_setup(result)
#plot_result(result, 2, f, ax)
plot_animation(plot_setup, plot_result, result, name='test.webm', fps=10, writer='ffmpeg',codec='libvpx-vp9')
#anim = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
#anim.save('hej.webm', fps=10, writer='ffmpeg', codec='libvpx-vp9')

