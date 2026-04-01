import elements
import embodied
import numpy as np
from PIL import Image


class VizDoom(embodied.Env):

  def __init__(
      self, task, size=(64, 64), resize='pillow', frame_skip=4,
      obs_key='screen', **kwargs):
    assert resize in ('opencv', 'pillow'), resize
    from vizdoom import gymnasium_wrapper  # noqa: F401
    from . import from_gymnasium

    self.size = size
    self.resize = resize
    self._obs_key = obs_key

    env_id = f'Vizdoom{task}-v1'
    make_kw = dict(frame_skip=frame_skip, **kwargs)
    if 'render_mode' not in make_kw:
      make_kw['render_mode'] = 'rgb_array'
    self.env = from_gymnasium.FromGymnasium(
        env_id, obs_key=obs_key, **make_kw)

  @property
  def obs_space(self):
    spaces = self.env.obs_space.copy()
    if self._obs_key in spaces:
      spaces.pop(self._obs_key)
    spaces['image'] = elements.Space(np.uint8, (*self.size, 3))
    return spaces

  @property
  def act_space(self):
    return self.env.act_space

  def step(self, action):
    obs = self.env.step(action)
    if self._obs_key in obs:
      screen = obs.pop(self._obs_key)
      obs['image'] = self._resize(
          self._to_rgb(np.asarray(screen)), self.size, self.resize)
    return obs

  def render(self):
    return self.env.render()

  def close(self):
    self.env.close()

  def _to_rgb(self, screen):
    if screen.ndim == 2:
      screen = screen[..., None]
    if screen.shape[-1] == 1:
      screen = np.repeat(screen.astype(np.uint8), 3, axis=-1)
    return screen.astype(np.uint8, copy=False)

  def _resize(self, image, size, method):
    if method == 'opencv':
      import cv2
      image = cv2.resize(
          image, (size[1], size[0]), interpolation=cv2.INTER_AREA)
      return image
    elif method == 'pillow':
      image = Image.fromarray(image)
      image = image.resize((size[1], size[0]), Image.BILINEAR)
      return np.array(image)
    else:
      raise NotImplementedError(method)