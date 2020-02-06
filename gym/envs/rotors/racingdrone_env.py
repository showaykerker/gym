# set PYTHON_PATH
import sys, os, pathlib
dir_path = str(pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent)
if dir_path not in sys.path: 
    print('Append %s to sys.path' % dir_path)
    sys.path.append(dir_path)

from RDPG_Drone.env import simple_validation_env as sve
from RDPG_Drone.agent.model import VAE, ViAE
import cv2
import torch
import gym
import numpy as np
import ros_numpy

def to_tensor(ndarray):
	return torch.from_numpy(ndarray).type(torch.FloatTensor)

def to_numpy(tensor):
	return tensor.cpu().data.numpy()

class EnvWrapper(sve):

	ob_mode_enum = ['latent_mono', 'latent_bino', 'img_mono', 'img_bino']
	weight_path = { 'VAE': '/media/storage1/rdpg_drone_data/encoder_model/encoder7_0203_mono_ep600_64.pt',
					'ViAE': ''
				  }

	def __init__(self, model='VAE', observation_mode='latent_mono', latent_dim=64, input_channel=1, *args, **kwargs):

		assert observation_mode in self.ob_mode_enum, 'observation_mode must be one of following: %s' % self.ob_mode_enum

		super(EnvWrapper, self).__init__(*args, **kwargs)
		self.n_gates_once = 1

		if model in ['VAE', 'ViAE']:
			self.device = torch.device("cuda:1")
			self.weight_path = self.weight_path[model]
			if  model == 'ViAE': self.model = ViAE(extra_latent_dim=kwargs['latent_dim']-6, device=self.device)
			elif model == 'VAE': self.model = VAE(latent_dim=latent_dim, input_channel=input_channel, device=self.device)

		
		self.observation_mode = observation_mode
		if observation_mode in ['latent_mono', 'latent_bino']:
			assert model is not None
			self.model.load_state_dict(torch.load(self.weight_path))
			self.model.to(self.device)
			self.model.eval()
			self.observation_space = gym.spaces.Box(-5, 5, (latent_dim, ), dtype=float)
		elif observation_mode == 'img_mono':
			self.observation_space = gym.spaces.Box(0, 1, (120, 192, 1), dtype=float)
		elif observation_mode == 'img_bino':
			self.observation_space = gym.spaces.Box(0, 1, (120, 192, 2), dtype=float)

		self.to_render = True

	def step(self, action):
		ob_, r, done, info = super().step(action)
		if done: print(info)
		return ob_, r, done, info


	def _make_observation(self):

		image = np.zeros((2, 120, 192)).astype(np.float32)
		
		image[0] = ros_numpy.numpify(self.image_msg_queue['left'][-1]).copy().mean(axis=2)/255
		image[1] = ros_numpy.numpify(self.image_msg_queue['right'][-1]).copy().mean(axis=2)/255

		self.render()

		if self.observation_mode == 'latent_mono':
			with torch.no_grad():
				tensor = to_tensor(image[1].reshape(1, 1, 120, 192)).to(self.device)
				latent = self.model.encode_to_latent(tensor)
				self.last_latent = latent.clone()
				#print(to_numpy(latent).reshape(-1, ))
				return to_numpy(latent).reshape(-1, )
		elif self.observation_mode == 'latent_bino':
			with torch.no_grad():
				tensor = to_tensor(image.reshape(1, 2, 120, 192)).to(self.device)
				latent = self.model.encode_to_latent()
				self.last_latent = latent.clone()
				return to_numpy(latent).reshape(-1, )
		elif self.observation_mode == 'img_mono':
			return image[1].transpose(1, 2, 0)
		elif self.observation_mode == 'img_bino':
			return image.transpose(1, 2, 0)

	def render(self):
		if self.to_render and hasattr(self, 'last_latent'):
			# print(self.last_latent)
			# print(self.last_latent.min(), self.last_latent.max())
			img = self.model.decode(self.last_latent.to(self.device))
			cv2.imshow('decoded', (to_numpy(img.view(120, 192)*255)).astype(np.uint8))
			k = cv2.waitKey(1)
			if k == ord('q'):
				self.model.rm_decoder_part()
				self.to_render = False
				cv2.destroyAllWindows()