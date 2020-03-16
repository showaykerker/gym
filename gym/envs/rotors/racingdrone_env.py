# set PYTHON_PATH
import sys, os, pathlib
dir_path = str(pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent)
if dir_path not in sys.path: 
	print('Append %s to sys.path' % dir_path)
	sys.path.append(dir_path)

from RDPG_Drone.env import simple_validation_env as sve
from RDPG_Drone.agent.model import VAE, ViAE
from RDPG_Drone.agent import torch_agent
import cv2
import torch
import gym
import numpy as np
import ros_numpy
from sensor_msgs.msg import Image


def to_tensor(ndarray):
	return torch.from_numpy(ndarray).type(torch.FloatTensor)

def to_numpy(tensor):
	return tensor.cpu().data.numpy()

class EnvWrapper(sve):

	ob_mode_enum = ['latent_mono', 'latent_bino', 'img_mono', 'img_bino', 'info_arr', 'info_rtpp', 'info_dict', None]

	def __init__(self, *args, **kwargs):
		observation_mode = kwargs.get('observation_mode')
		assert observation_mode in self.ob_mode_enum, 'observation_mode must be one of following: %s' % self.ob_mode_enum
		super(EnvWrapper, self).__init__(*args, **kwargs)

	def step(self, action):
		try: self.render()
		except: pass
		ob_, r, done, info = super().step(action)
		# if done: print(info)
		return ob_, r, done, info


	def render(self, *args, **kwargs):
		return
		#if not self.to_render: return	
		if self.to_render and hasattr(self, 'last_latents'):
			# print(self.last_latent)
			# print(self.last_latent.min(), self.last_latent.max())


			img = np.zeros((120*self.n_history, 192*self.input_channel)).astype(np.uint8)

			imgs = self.loader.model.decode(self.last_latents)

			if self.mode == 'latent_mono':  # imgs.shape = (self.n_history, 1, 120, 192)
				for i, im in enumerate(imgs):
					img[120*i: 120*(i+1), :] = to_numpy(im*255).astype(np.uint8)
			elif self.mode == 'latent_bino':  # imgs.shape = (self.n_history, 2, 120, 192)
				for i, im in enumerate(imgs):
					# print('im.shape:', im.shape)  # (2, 120, 192)
					img[120*i: 120*(i+1), :192] = to_numpy(im[0]*255).astype(np.uint8)
					img[120*i: 120*(i+1), 192:] = to_numpy(im[1]*255).astype(np.uint8)

			if img.shape[0] > 720:
				ratio = 720 / img.shape[0]
				height = int(img.shape[0] * ratio)
				width = int(img.shape[1] * ratio)
				img = cv2.resize(img, (width, height))

			cv2.imshow('decoded%s'%self.env_suffix, img)
			
			k = cv2.waitKey(1)
			if k == ord('q'):
				self.loader.model.rm_decoder_part()
				self.to_render = False
				cv2.destroyAllWindows()

