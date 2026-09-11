from OpenGL import GL
import pyglet

config = pyglet.gl.Config(double_buffer=True)
window = pyglet.window.Window(config=config)

print("OpenGL version:", GL.glGetString(GL.GL_VERSION))
window.close()
