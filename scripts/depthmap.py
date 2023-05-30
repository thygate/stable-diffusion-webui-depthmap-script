# Author: thygate
# https://github.com/thygate/stable-diffusion-webui-depthmap-script

import modules.scripts as scripts
import gradio as gr

from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call
from modules.ui import plaintext_to_html
from modules import processing, images, shared, sd_samplers, devices
from modules.processing import create_infotext, process_images, Processed
from modules.shared import opts, cmd_opts, state, Options
from modules import script_callbacks
from modules.images import get_next_sequence_number
from numba import njit, prange
from torchvision.transforms import Compose, transforms
from PIL import Image
from pathlib import Path
from operator import getitem
from tqdm import trange
from functools import reduce
from skimage.transform import resize
from trimesh import transformations

import sys
import torch, gc
import torch.nn as nn
import cv2
import os.path
import contextlib
import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import copy
import platform
import vispy
import trimesh
import math
import subprocess

sys.path.append('extensions/stable-diffusion-webui-depthmap-script/scripts')

from stereoimage_generation import create_stereoimages

# midas imports
from dmidas.dpt_depth import DPTDepthModel
from dmidas.midas_net import MidasNet
from dmidas.midas_net_custom import MidasNet_small
from dmidas.transforms import Resize, NormalizeImage, PrepareForNet

# AdelaiDepth/LeReS imports
from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import strip_prefix_if_present

# pix2pix/merge net imports
from pix2pix.options.test_options import TestOptions
from pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel

# 3d-photo-inpainting imports
from inpaint.mesh import write_mesh, read_mesh, output_3d_photo
from inpaint.networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net
from inpaint.utils import path_planning
from inpaint.bilateral_filtering import sparse_bilateral_filtering

# zoedepth
from dzoedepth.models.builder import build_model
from dzoedepth.utils.config import get_config
from dzoedepth.utils.misc import colorize
from dzoedepth.utils.geometry import depth_to_points, create_triangles

# background removal
from rembg import new_session, remove

whole_size_threshold = 1600  # R_max from the paper
pix2pixsize = 1024
scriptname = "DepthMap"
scriptversion = "v0.3.11"

global video_mesh_data, video_mesh_fn
video_mesh_data = None
video_mesh_fn = None

global depthmap_model_depth, depthmap_model_pix2pix, depthmap_model_type, depthmap_deviceidx
depthmap_model_depth = None
depthmap_model_pix2pix = None
depthmap_model_type = None
depthmap_deviceidx = None

def get_commit_hash():
	try:
		hash = subprocess.check_output([os.environ.get('GIT', "git"), "rev-parse", "HEAD"], shell=False, encoding='utf8').strip()
		hash = hash[0:8]
		return hash
	except Exception:
		return "<none>"
commit_hash = get_commit_hash()

def main_ui_panel(is_depth_tab):
	with gr.Blocks():
		with gr.Row():
			compute_device = gr.Radio(label="Compute on", choices=['GPU', 'CPU'], value='GPU', type="index")
			model_type = gr.Dropdown(label="Model", choices=['res101', 'dpt_beit_large_512 (midas 3.1)',
															 'dpt_beit_large_384 (midas 3.1)',
															 'dpt_large_384 (midas 3.0)', 'dpt_hybrid_384 (midas 3.0)',
															 'midas_v21', 'midas_v21_small', 
															 'zoedepth_n (indoor)', 'zoedepth_k (outdoor)', 'zoedepth_nk'], value='res101',
									 type="index", elem_id="tabmodel_type")
		with gr.Group():
			with gr.Row():
				boost = gr.Checkbox(label="BOOST (multi-resolution merging)", value=True)
				invert_depth = gr.Checkbox(label="Invert DepthMap (black=near, white=far)", value=False)
			with gr.Group(visible=False) as options_depend_on_boost:
				match_size = gr.Checkbox(label="Match input size", value=False)
				with gr.Row() as options_depend_on_match_size:
					net_width = gr.Slider(minimum=64, maximum=2048, step=64, label='Net width', value=512)
					net_height = gr.Slider(minimum=64, maximum=2048, step=64, label='Net height', value=512)

		with gr.Group():
			with gr.Row():
				clipdepth = gr.Checkbox(label="Clip and renormalize", value=False)
			with gr.Row(visible=False) as clip_options_row_1:
				clipthreshold_far = gr.Slider(minimum=0, maximum=1, step=0.001, label='Far clip', value=0)
				clipthreshold_near = gr.Slider(minimum=0, maximum=1, step=0.001, label='Near clip', value=1)

		with gr.Group():
			with gr.Row():
				combine_output = gr.Checkbox(label="Combine into one image", value=False)
				combine_output_axis = gr.Radio(label="Combine axis", choices=['Vertical', 'Horizontal'],
											   value='Horizontal', type="index")
			with gr.Row():
				save_depth = gr.Checkbox(label="Save DepthMap", value=True)
				show_depth = gr.Checkbox(label="Show DepthMap", value=True)
				show_heat = gr.Checkbox(label="Show HeatMap", value=False)

		with gr.Group():
			with gr.Row():
				gen_stereo = gr.Checkbox(label="Generate stereoscopic image(s)", value=False)
				with gr.Group(visible=False) as stereo_options_row_0:
					with gr.Row():
						stereo_modes = gr.CheckboxGroup(["left-right", "right-left", "top-bottom", "bottom-top", "red-cyan-anaglyph"], label="Output", value=["left-right","red-cyan-anaglyph"])

			with gr.Row(visible=False) as stereo_options_row_1:
				stereo_divergence = gr.Slider(minimum=0.05, maximum=10.005, step=0.01, label='Divergence (3D effect)',
											  value=2.5)
			with gr.Row(visible=False) as stereo_options_row_2:
				stereo_fill = gr.Dropdown(label="Gap fill technique",
										  choices=['none', 'naive', 'naive_interpolating', 'polylines_soft',
												   'polylines_sharp'], value='polylines_sharp', type="value",
										  elem_id="stereo_fill_type")
				stereo_balance = gr.Slider(minimum=-1.0, maximum=1.0, step=0.05, label='Balance between eyes',
										   value=0.0)

		with gr.Group():
			with gr.Row():
				gen_mesh = gr.Checkbox(label="Generate simple 3D mesh. (Fast, accurate only with ZoeDepth models and no boost, no custom maps)", value=False, visible=True)
			with gr.Row(visible=False) as mesh_options_row_0:
				mesh_occlude = gr.Checkbox(label="Remove occluded edges", value=True, visible=True)
				mesh_spherical = gr.Checkbox(label="Equirectangular projection", value=False, visible=True)

		with gr.Group(visible=is_depth_tab):
			with gr.Row():
				inpaint = gr.Checkbox(label="Generate 3D inpainted mesh. (Sloooow, required for generating videos)", value=False, visible=is_depth_tab)
			with gr.Row(visible=False) as inpaint_options_row_0:
				inpaint_vids = gr.Checkbox(label="Generate 4 demo videos with 3D inpainted mesh.", value=False, visible=is_depth_tab)

		with gr.Group():
			with gr.Row():
				background_removal = gr.Checkbox(label="Remove background", value=False)
			with gr.Row(visible=False) as bgrem_options_row_1:
				save_background_removal_masks = gr.Checkbox(label="Save the foreground masks", value=False)
				pre_depth_background_removal = gr.Checkbox(label="Pre-depth background removal", value=False)
			with gr.Row(visible=False) as bgrem_options_row_2:
				background_removal_model = gr.Dropdown(label="Rembg Model",
													   choices=['u2net', 'u2netp', 'u2net_human_seg', 'silueta'],
													   value='u2net', type="value", elem_id="backgroundmodel_type")

		with gr.Box():
			gr.HTML("Information, comment and share @ <a "
					"href='https://github.com/thygate/stable-diffusion-webui-depthmap-script'>"
					"https://github.com/thygate/stable-diffusion-webui-depthmap-script</a>")

		gen_normal = gr.Checkbox(label="Generate Normalmap (hidden! api only)", value=False, visible=False)


		clipthreshold_far.change(
			fn=lambda a, b: a if b < a else b,
			inputs=[clipthreshold_far, clipthreshold_near],
			outputs=[clipthreshold_near]
		)

		clipthreshold_near.change(
			fn=lambda a, b: a if b > a else b,
			inputs=[clipthreshold_near, clipthreshold_far],
			outputs=[clipthreshold_far]
		)

		boost.change(
			fn=lambda a: options_depend_on_boost.update(visible = not a),
			inputs=[boost],
			outputs=[options_depend_on_boost]
		)

		match_size.change(
			fn=lambda a: options_depend_on_match_size.update(visible = not a),
			inputs=[match_size],
			outputs=[options_depend_on_match_size]
		)

		def clipdepth_options_visibility(v):
			return clip_options_row_1.update(visible=v)
		clipdepth.change(
			fn=clipdepth_options_visibility,
			inputs=[clipdepth],
			outputs=[clip_options_row_1]
		)

		def stereo_options_visibility(v):
			return stereo_options_row_0.update(visible=v),\
				   stereo_options_row_1.update(visible=v),\
				   stereo_options_row_2.update(visible=v)
		gen_stereo.change(
			fn=stereo_options_visibility,
			inputs=[gen_stereo],
			outputs=[stereo_options_row_0, stereo_options_row_1, stereo_options_row_2]
		)

		def mesh_options_visibility(v):
			return mesh_options_row_0.update(visible=v)
		gen_mesh.change(
			fn=mesh_options_visibility,
			inputs=[gen_mesh],
			outputs=[mesh_options_row_0]
		)

		def inpaint_options_visibility(v):
			return inpaint_options_row_0.update(visible=v)
		inpaint.change(
			fn=inpaint_options_visibility,
			inputs=[inpaint],
			outputs=[inpaint_options_row_0]
		)

		def background_removal_options_visibility(v):
			return bgrem_options_row_1.update(visible=v), \
				   bgrem_options_row_2.update(visible=v)
		background_removal.change(
			fn=background_removal_options_visibility,
			inputs=[background_removal],
			outputs=[bgrem_options_row_1, bgrem_options_row_2]
		)

	return [compute_device, model_type, net_width, net_height, match_size, boost, invert_depth, clipdepth, clipthreshold_far, clipthreshold_near, combine_output, combine_output_axis, save_depth, show_depth, show_heat, gen_stereo, stereo_modes, stereo_divergence, stereo_fill, stereo_balance, inpaint, inpaint_vids, background_removal, save_background_removal_masks, gen_normal, pre_depth_background_removal, background_removal_model, gen_mesh, mesh_occlude, mesh_spherical]


class Script(scripts.Script):
	def title(self):
		return scriptname

	def show(self, is_img2img):
		return True

	def ui(self, is_img2img):
		with gr.Column(variant='panel'):
			ret = main_ui_panel(False)
		return ret

	# run from script in txt2img or img2img
	def run(self, p,
			compute_device, model_type, net_width, net_height, match_size, boost, invert_depth, clipdepth, clipthreshold_far, clipthreshold_near, combine_output, combine_output_axis, save_depth, show_depth, show_heat, gen_stereo, stereo_modes, stereo_divergence, stereo_fill, stereo_balance, inpaint, inpaint_vids, background_removal, save_background_removal_masks, gen_normal, pre_depth_background_removal, background_removal_model, gen_mesh, mesh_occlude, mesh_spherical
			):

		# sd process 
		processed = processing.process_images(p)

		processed.sampler = p.sampler # for create_infotext

		inputimages = []
		for count in range(0, len(processed.images)):
			# skip first grid image
			if count == 0 and len(processed.images) > 1 and opts.return_grid:
				continue
			inputimages.append(processed.images[count])
		
		#remove on base image before depth calculation
		background_removed_images = []
		if background_removal:
			if pre_depth_background_removal:
				inputimages = batched_background_removal(inputimages, background_removal_model)
				background_removed_images = inputimages
			else:
				background_removed_images = batched_background_removal(inputimages, background_removal_model)			

		newmaps, mesh_fi, meshsimple_fi = run_depthmap(processed, p.outpath_samples, inputimages, None,
                                        compute_device, model_type,
                                        net_width, net_height, match_size, boost, invert_depth, clipdepth, clipthreshold_far, clipthreshold_near, combine_output, combine_output_axis, save_depth, show_depth, show_heat, gen_stereo, stereo_modes, stereo_divergence, stereo_fill, stereo_balance, inpaint, inpaint_vids, background_removal, save_background_removal_masks, gen_normal,
                                        background_removed_images, "mp4", 0, False, None, False, gen_mesh, mesh_occlude, mesh_spherical )
		
		for img in newmaps:
			processed.images.append(img)

		return processed

def run_depthmap(processed, outpath, inputimages, inputnames,
                 compute_device, model_type, net_width, net_height, match_size, boost, invert_depth, clipdepth, clipthreshold_far, clipthreshold_near, combine_output, combine_output_axis, save_depth, show_depth, show_heat, gen_stereo, stereo_modes, stereo_divergence, stereo_fill, stereo_balance, inpaint, inpaint_vids, background_removal, save_background_removal_masks, gen_normal,
                 background_removed_images, fnExt, vid_ssaa, custom_depthmap, custom_depthmap_img, depthmap_batch_reuse, gen_mesh, mesh_occlude, mesh_spherical):

	if len(inputimages) == 0 or inputimages[0] == None:
		return [], []
	
	print(f"\n{scriptname} {scriptversion} ({commit_hash})")

	# unload sd model
	shared.sd_model.cond_stage_model.to(devices.cpu)
	shared.sd_model.first_stage_model.to(devices.cpu)

	meshsimple_fi = None
	mesh_fi = None

	resize_mode = "minimal"
	normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

	# init torch device
	global device
	if compute_device == 0:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	else:
		device = torch.device("cpu")
	print("device: %s" % device)

	# model path and name
	model_dir = "./models/midas"
	if model_type == 0:
		model_dir = "./models/leres"
	# create paths to model if not present
	os.makedirs(model_dir, exist_ok=True)
	os.makedirs('./models/pix2pix', exist_ok=True)

	global depthmap_model_depth, depthmap_model_pix2pix, depthmap_model_type, depthmap_device_idx
	loadmodels = True
	if hasattr(opts, 'depthmap_script_keepmodels') and opts.depthmap_script_keepmodels:
		loadmodels = False
		if depthmap_model_type != model_type or depthmap_model_depth == None or depthmap_device_idx != compute_device:
			del depthmap_model_depth
			depthmap_model_depth = None
			loadmodels = True

	outimages = []
	try:
		if loadmodels and not (custom_depthmap and custom_depthmap_img != None):
			print("Loading model weights from ", end=" ")

			#"res101"
			if model_type == 0: 
				model_path = f"{model_dir}/res101.pth"
				print(model_path)
				if not os.path.exists(model_path):
					download_file(model_path,"https://cloudstor.aarnet.edu.au/plus/s/lTIJF4vrvHCAI31/download")
				if compute_device == 0:
					checkpoint = torch.load(model_path)
				else:
					checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
				model = RelDepthModel(backbone='resnext101')
				model.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."), strict=True)
				del checkpoint
				devices.torch_gc()

			#"dpt_beit_large_512" midas 3.1
			if model_type == 1: 
				model_path = f"{model_dir}/dpt_beit_large_512.pt"
				print(model_path)
				if not os.path.exists(model_path):
					download_file(model_path,"https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt")
				model = DPTDepthModel(
					path=model_path,
					backbone="beitl16_512",
					non_negative=True,
				)
				net_w, net_h = 512, 512
				resize_mode = "minimal"
				normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

			#"dpt_beit_large_384" midas 3.1
			if model_type == 2: 
				model_path = f"{model_dir}/dpt_beit_large_384.pt"
				print(model_path)
				if not os.path.exists(model_path):
					download_file(model_path,"https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_384.pt")
				model = DPTDepthModel(
					path=model_path,
					backbone="beitl16_384",
					non_negative=True,
				)
				net_w, net_h = 384, 384
				resize_mode = "minimal"
				normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

			#"dpt_large_384" midas 3.0
			if model_type == 3: 
				model_path = f"{model_dir}/dpt_large-midas-2f21e586.pt"
				print(model_path)
				if not os.path.exists(model_path):
					download_file(model_path,"https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt")
				model = DPTDepthModel(
					path=model_path,
					backbone="vitl16_384",
					non_negative=True,
				)
				net_w, net_h = 384, 384
				resize_mode = "minimal"
				normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

			#"dpt_hybrid_384" midas 3.0
			elif model_type == 4: 
				model_path = f"{model_dir}/dpt_hybrid-midas-501f0c75.pt"
				print(model_path)
				if not os.path.exists(model_path):
					download_file(model_path,"https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt")
				model = DPTDepthModel(
					path=model_path,
					backbone="vitb_rn50_384",
					non_negative=True,
				)
				net_w, net_h = 384, 384
				resize_mode="minimal"
				normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

			#"midas_v21"
			elif model_type == 5: 
				model_path = f"{model_dir}/midas_v21-f6b98070.pt"
				print(model_path)
				if not os.path.exists(model_path):
					download_file(model_path,"https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt")
				model = MidasNet(model_path, non_negative=True)
				net_w, net_h = 384, 384
				resize_mode="upper_bound"
				normalization = NormalizeImage(
					mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
				)

			#"midas_v21_small"
			elif model_type == 6: 
				model_path = f"{model_dir}/midas_v21_small-70d6b9c8.pt"
				print(model_path)
				if not os.path.exists(model_path):
					download_file(model_path,"https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt")
				model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
				net_w, net_h = 256, 256
				resize_mode="upper_bound"
				normalization = NormalizeImage(
					mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
				)

			# zoedepth_n
			elif model_type == 7:
				print("zoedepth_n\n")
				conf = get_config("zoedepth", "infer")
				conf.img_size = [net_width, net_height]
				model = build_model(conf)

			# zoedepth_k
			elif model_type == 8:
				print("zoedepth_k\n")
				conf = get_config("zoedepth", "infer", config_version="kitti")
				conf.img_size = [net_width, net_height]
				model = build_model(conf)

			# zoedepth_nk
			elif model_type == 9:
				print("zoedepth_nk\n")
				conf = get_config("zoedepth_nk", "infer")
				conf.img_size = [net_width, net_height]
				model = build_model(conf)

			pix2pixmodel = None
			# load merge network if boost enabled or keepmodels enabled
			if boost or (hasattr(opts, 'depthmap_script_keepmodels') and opts.depthmap_script_keepmodels):
				pix2pixmodel_path = './models/pix2pix/latest_net_G.pth'
				if not os.path.exists(pix2pixmodel_path):
					download_file(pix2pixmodel_path,"https://sfu.ca/~yagiz/CVPR21/latest_net_G.pth")
				opt = TestOptions().parse()
				if compute_device == 1:
					opt.gpu_ids = [] # cpu mode
				pix2pixmodel = Pix2Pix4DepthModel(opt)
				pix2pixmodel.save_dir = './models/pix2pix'
				pix2pixmodel.load_networks('latest')
				pix2pixmodel.eval()

			devices.torch_gc()

			# prepare for evaluation
			model.eval()
		
			# optimize
			if device == torch.device("cuda") and model_type < 7:
				model = model.to(memory_format=torch.channels_last)  
				if not cmd_opts.no_half and model_type != 0 and not boost:
					model = model.half()

			model.to(device)

			depthmap_model_depth = model
			depthmap_model_pix2pix = pix2pixmodel
			depthmap_model_type = model_type
			depthmap_device_idx = compute_device

		if not loadmodels:
			model = depthmap_model_depth
			pix2pixmodel = depthmap_model_pix2pix
			if device == torch.device("cuda"):
				model = model.to(device)


		print("Computing depthmap(s) ..")
		inpaint_imgs = []
		inpaint_depths = []
		# iterate over input (generated) images
		numimages = len(inputimages)
		for count in trange(0, numimages):

			print('\n')

			# filename
			basename = 'depthmap'

			# figuring out the name of custom DepthMap
			custom_depthmap_fn = None  # None means that DepthMap should be computed
			# find filename if in the single image mode
			if custom_depthmap and custom_depthmap_img is not None:
				custom_depthmap_fn = custom_depthmap_img.name
			# find filename if in batch mode
			if inputnames is not None and depthmap_batch_reuse:
				save_depth = True
				if inputnames[count] is not None:
					p = Path(inputnames[count])
					basename = p.stem
					if outpath != opts.outdir_extras_samples:
						custom_depthmap_fn = os.path.join(outpath, basename + '-0000.' + opts.samples_format)
						if not os.path.isfile(custom_depthmap_fn):
							custom_depthmap_fn = None

			# override net size
			if (match_size):
				net_width, net_height = inputimages[count].width, inputimages[count].height

			# Convert single channel input (PIL) images to rgb
			if inputimages[count].mode == 'I':
				inputimages[count].point(lambda p: p*0.0039063096, mode='RGB')
				inputimages[count] = inputimages[count].convert('RGB')

			# input image
			img = cv2.cvtColor(np.asarray(inputimages[count]), cv2.COLOR_BGR2RGB) / 255.0

			skipInvertAndSave = False
			if custom_depthmap_fn is not None:
				# use custom depthmap
				dimg = Image.open(os.path.abspath(custom_depthmap_fn))
				# resize if not same size as input
				if dimg.width != inputimages[count].width or dimg.height != inputimages[count].height:
					dimg = dimg.resize((inputimages[count].width, inputimages[count].height), Image.Resampling.LANCZOS)
				if dimg.mode == 'I' or dimg.mode == 'P' or dimg.mode == 'L':
					prediction = np.asarray(dimg, dtype="float")
				else:
					prediction = np.asarray(dimg, dtype="float")[:,:,0]
				skipInvertAndSave = True #skip invert for leres model (0)
			else:
				# compute depthmap
				if not boost:
					if model_type == 0:
						prediction = estimateleres(img, model, net_width, net_height)
					elif model_type >= 7:
						prediction = estimatezoedepth(inputimages[count], model, net_width, net_height)
					else:
						prediction = estimatemidas(img, model, net_width, net_height, resize_mode, normalization)
				else:
					prediction = estimateboost(img, model, model_type, pix2pixmodel)

			# output
			depth = prediction
			numbytes=2
			depth_min = depth.min()
			depth_max = depth.max()
			max_val = (2**(8*numbytes))-1

			# check output before normalizing and mapping to 16 bit
			if depth_max - depth_min > np.finfo("float").eps:
				out = max_val * (depth - depth_min) / (depth_max - depth_min)
			else:
				out = np.zeros(depth.shape)
			
			# single channel, 16 bit image
			img_output = out.astype("uint16")

			# invert depth map
			if invert_depth ^ (((model_type == 0) or (model_type >= 7)) and not skipInvertAndSave):
				img_output = cv2.bitwise_not(img_output)

			# apply depth clip and renormalize if enabled
			if clipdepth:
				img_output = clipdepthmap(img_output, clipthreshold_far, clipthreshold_near)
				#img_output = cv2.blur(img_output, (3, 3))

			# three channel, 8 bits per channel image
			img_output2 = np.zeros_like(inputimages[count])
			img_output2[:,:,0] = img_output / 256.0
			img_output2[:,:,1] = img_output / 256.0
			img_output2[:,:,2] = img_output / 256.0

			# if 3dinpainting, store maps for processing in second pass
			if inpaint:
				inpaint_imgs.append(inputimages[count])
				inpaint_depths.append(img_output)

			# get generation parameters
			if processed is not None and hasattr(processed, 'all_prompts') and opts.enable_pnginfo:
				info = create_infotext(processed, processed.all_prompts, processed.all_seeds, processed.all_subseeds, "", 0, count)
			else:
				info = None

			rgb_image = inputimages[count]

			#applying background masks after depth
			if background_removal:
				print('applying background masks')
				background_removed_image = background_removed_images[count]
				#maybe a threshold cut would be better on the line below.
				background_removed_array = np.array(background_removed_image)
				bg_mask = (background_removed_array[:,:,0]==0)&(background_removed_array[:,:,1]==0)&(background_removed_array[:,:,2]==0)&(background_removed_array[:,:,3]<=0.2)
				far_value = 255 if invert_depth else 0

				img_output[bg_mask] = far_value * far_value #255*255 or 0*0
				
				#should this be optional
				if (processed is not None):
					images.save_image(background_removed_image, outpath, "", processed.all_seeds[count], processed.all_prompts[count], opts.samples_format, info=info, p=processed, suffix="_background_removed")
				else:
					images.save_image(background_removed_image, path=outpath, basename=basename, seed=None, prompt=None, extension=opts.samples_format, info=info, short_filename=True,no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=None, forced_filename=None, suffix="_background_removed")
				outimages.append(background_removed_image )
				if save_background_removal_masks:
					bg_array = (1 - bg_mask.astype('int8'))*255
					mask_array = np.stack( (bg_array, bg_array, bg_array, bg_array), axis=2)
					mask_image = Image.fromarray( mask_array.astype(np.uint8))
					if (processed is not None):
						images.save_image(mask_image, outpath, "", processed.all_seeds[count], processed.all_prompts[count], opts.samples_format, info=info, p=processed, suffix="_foreground_mask")
					else:
						images.save_image(mask_image, path=outpath, basename=basename, seed=None, prompt=None, extension=opts.samples_format, info=info, short_filename=True,no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=None, forced_filename=None, suffix="_foreground_mask")
					outimages.append(mask_image)

			img_concat = np.concatenate((rgb_image, img_output2), axis=combine_output_axis)
			if show_depth:
				if not combine_output:
					outimages.append(Image.fromarray(img_output))
				else:
					outimages.append(Image.fromarray(img_concat))
					
			if not skipInvertAndSave:
				if not combine_output:
					if save_depth and processed is not None:
						# only save 16 bit single channel image when PNG format is selected
						if opts.samples_format == "png":
							images.save_image(Image.fromarray(img_output), outpath, "", processed.all_seeds[count], processed.all_prompts[count], opts.samples_format, info=info, p=processed, suffix="_depth")
						else:
							images.save_image(Image.fromarray(img_output2), outpath, "", processed.all_seeds[count], processed.all_prompts[count], opts.samples_format, info=info, p=processed, suffix="_depth")
					elif save_depth:
						# from depth tab
						# only save 16 bit single channel image when PNG format is selected
						if opts.samples_format == "png":
							images.save_image(Image.fromarray(img_output), path=outpath, basename=basename, seed=None, prompt=None, extension=opts.samples_format, info=info, short_filename=True,no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=None, forced_filename=None)
						else:
							images.save_image(Image.fromarray(img_output2), path=outpath, basename=basename, seed=None, prompt=None, extension=opts.samples_format, info=info, short_filename=True,no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=None, forced_filename=None)
				else:
					if save_depth and processed is not None:
						images.save_image(Image.fromarray(img_concat), outpath, "", processed.all_seeds[count], processed.all_prompts[count], opts.samples_format, info=info, p=processed, suffix="_depth")
					elif save_depth:
						# from tab
						images.save_image(Image.fromarray(img_concat), path=outpath, basename=basename, seed=None, prompt=None, extension=opts.samples_format, info=info, short_filename=True,no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=None, forced_filename=None)
			if show_heat:
				heatmap = colorize(img_output, cmap='inferno')
				outimages.append(heatmap)

			if gen_stereo:
				print("Generating stereoscopic images..")

				stereomodes = stereo_modes
				stereoimages = create_stereoimages(inputimages[count], img_output, stereo_divergence, stereomodes, stereo_balance, stereo_fill)

				for c in range(0, len(stereoimages)):
					outimages.append(stereoimages[c])
					if processed is not None:
						images.save_image(stereoimages[c], outpath, "", processed.all_seeds[count],
										processed.all_prompts[count], opts.samples_format, info=info, p=processed,
										suffix=f"_{stereomodes[c]}")
					else:
						# from tab
						images.save_image(stereoimages[c], path=outpath, basename=basename, seed=None,
										prompt=None, extension=opts.samples_format, info=info, short_filename=True,
										no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=None,
										forced_filename=None, suffix=f"_{stereomodes[c]}")

			if gen_normal:
				# taken from @graemeniedermayer, hidden, for api use only, will remove in future.
				# take gradients 
				zx = cv2.Sobel(np.float64(img_output), cv2.CV_64F, 1, 0, ksize=3)     
				zy = cv2.Sobel(np.float64(img_output), cv2.CV_64F, 0, 1, ksize=3) 

				# combine and normalize gradients.
				normal = np.dstack((zx, -zy, np.ones_like(img_output)))
				n = np.linalg.norm(normal, axis=2)
				normal[:, :, 0] /= n
				normal[:, :, 1] /= n
				normal[:, :, 2] /= n

				# offset and rescale values to be in 0-255
				normal += 1
				normal /= 2
				normal *= 255	
				normal = normal.astype(np.uint8)
				
				outimages.append(Image.fromarray(normal))

			# gen mesh
			if gen_mesh:
				print(f"\nGenerating (occluded) mesh ..")

				meshsimple_fi = get_uniquefn(outpath, basename, 'obj')
				meshsimple_fi = os.path.join(outpath, meshsimple_fi + '_simple.obj')

				depthi = prediction
				# try to map output to sensible values for non zoedepth models, boost, or custom maps
				if model_type < 7 or boost or (custom_depthmap and custom_depthmap_img != None):
					# invert if midas
					if model_type > 0 or ((custom_depthmap and custom_depthmap_img != None) and not invert_depth):
						depthi = depth_max - depthi + depth_min
						depth_max = depthi.max()
						depth_min = depthi.min()
					# make positive
					if depth_min < 0:
						depthi = depthi - depth_min
						depth_max = depthi.max()
						depth_min = depthi.min()
					# scale down 
					if depthi.max() > 10:
						depthi = 4 * (depthi - depth_min) / (depth_max - depth_min)
					# offset
					depthi = depthi + 1

				mesh = create_mesh(inputimages[count], depthi, keep_edges=not mesh_occlude, spherical=mesh_spherical)
				save_mesh_obj(meshsimple_fi, mesh)

		print("Done.")

	except RuntimeError as e:
		if 'out of memory' in str(e):
			print("ERROR: out of memory, could not generate depthmap !")
		else:
			print(e)

	finally:
		if not (hasattr(opts, 'depthmap_script_keepmodels') and opts.depthmap_script_keepmodels):
			if 'model' in locals():
				del model
			if boost and 'pix2pixmodel' in locals():
				del pix2pixmodel
		else:
			if 'model' in locals():
				model.to(devices.cpu)

		gc.collect()
		devices.torch_gc()
		# reload sd model
		shared.sd_model.cond_stage_model.to(devices.device)
		shared.sd_model.first_stage_model.to(devices.device)

	
	try:
		if inpaint:
			# unload sd model
			shared.sd_model.cond_stage_model.to(devices.cpu)
			shared.sd_model.first_stage_model.to(devices.cpu)

			mesh_fi = run_3dphoto(device, inpaint_imgs, inpaint_depths, inputnames, outpath, fnExt, vid_ssaa, inpaint_vids)
	
	finally:
		# reload sd model
		shared.sd_model.cond_stage_model.to(devices.device)
		shared.sd_model.first_stage_model.to(devices.device)
		print("All done.")

	return outimages, mesh_fi, meshsimple_fi

@njit(parallel=True)
def clipdepthmap(img, clipthreshold_far, clipthreshold_near):
	clipped_img = img
	w, h = img.shape
	min = img.min()
	max = img.max()
	drange = max - min
	clipthreshold_far = min + (clipthreshold_far * drange)
	clipthreshold_near = min + (clipthreshold_near * drange)

	for x in prange(w):
		for y in range(h):
			if clipped_img[x,y] < clipthreshold_far:
				clipped_img[x,y] = 0
			elif clipped_img[x,y] > clipthreshold_near:
				clipped_img[x,y] = 65535
			else:
				clipped_img[x,y] = ((clipped_img[x,y] + min) / drange * 65535)

	return clipped_img

def get_uniquefn(outpath, basename, ext):
	# unique filename
	basecount = get_next_sequence_number(outpath, basename)
	if basecount > 0: basecount = basecount - 1
	fullfn = None
	for i in range(500):
		fn = f"{basecount + i:05}" if basename == '' else f"{basename}-{basecount + i:04}"
		fullfn = os.path.join(outpath, f"{fn}.{ext}")
		if not os.path.exists(fullfn):
			break
	basename = Path(fullfn).stem
	
	return basename

def run_3dphoto(device, img_rgb, img_depth, inputnames, outpath, fnExt, vid_ssaa, inpaint_vids):
	mesh_fi = ''
	try:
		print("Running 3D Photo Inpainting .. ")
		edgemodel_path = './models/3dphoto/edge_model.pth'
		depthmodel_path = './models/3dphoto/depth_model.pth'
		colormodel_path = './models/3dphoto/color_model.pth'
		# create paths to model if not present
		os.makedirs('./models/3dphoto/', exist_ok=True)

		if not os.path.exists(edgemodel_path):
			download_file(edgemodel_path,"https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/edge-model.pth")
		if not os.path.exists(depthmodel_path):
			download_file(depthmodel_path,"https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/depth-model.pth")
		if not os.path.exists(colormodel_path):
			download_file(colormodel_path,"https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/color-model.pth")
		
		print("Loading edge model ..")
		depth_edge_model = Inpaint_Edge_Net(init_weights=True)
		depth_edge_weight = torch.load(edgemodel_path, map_location=torch.device(device))
		depth_edge_model.load_state_dict(depth_edge_weight)
		depth_edge_model = depth_edge_model.to(device)
		depth_edge_model.eval()
		print("Loading depth model ..")
		depth_feat_model = Inpaint_Depth_Net()
		depth_feat_weight = torch.load(depthmodel_path, map_location=torch.device(device))
		depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
		depth_feat_model = depth_feat_model.to(device)
		depth_feat_model.eval()
		depth_feat_model = depth_feat_model.to(device)
		print("Loading rgb model ..")
		rgb_model = Inpaint_Color_Net()
		rgb_feat_weight = torch.load(colormodel_path, map_location=torch.device(device))
		rgb_model.load_state_dict(rgb_feat_weight)
		rgb_model.eval()
		rgb_model = rgb_model.to(device)

		config = {}
		config["gpu_ids"] = 0
		config['extrapolation_thickness'] = 60
		config['extrapolate_border'] = True
		config['depth_threshold'] = 0.04
		config['redundant_number'] = 12
		config['ext_edge_threshold'] = 0.002
		config['background_thickness'] = 70
		config['context_thickness'] = 140
		config['background_thickness_2'] = 70
		config['context_thickness_2'] = 70
		config['log_depth'] = True
		config['depth_edge_dilate'] = 10
		config['depth_edge_dilate_2'] = 5
		config['largest_size'] = 512
		config['repeat_inpaint_edge'] = True
		config['ply_fmt'] = "bin"

		config['save_ply'] = False
		if hasattr(opts, 'depthmap_script_save_ply') and opts.depthmap_script_save_ply:
			config['save_ply'] = True

		config['save_obj'] = True
		

		if device == torch.device("cpu"):
			config["gpu_ids"] = -1

		# process all inputs
		numimages = len(img_rgb)
		for count in trange(0, numimages):

			basename = 'depthmap'
			if inputnames is not None:
				if inputnames[count] is not None:
					p = Path(inputnames[count])
					basename = p.stem

			basename = get_uniquefn(outpath, basename, 'obj')
			mesh_fi = os.path.join(outpath, basename + '.obj')

			print(f"\nGenerating inpainted mesh .. (go make some coffee) ..")

			# from inpaint.utils.get_MiDaS_samples
			W = img_rgb[count].width
			H = img_rgb[count].height
			int_mtx = np.array([[max(H, W), 0, W//2], [0, max(H, W), H//2], [0, 0, 1]]).astype(np.float32)
			if int_mtx.max() > 1:
				int_mtx[0, :] = int_mtx[0, :] / float(W)
				int_mtx[1, :] = int_mtx[1, :] / float(H)

			# how inpaint.utils.read_MiDaS_depth() imports depthmap
			disp = img_depth[count].astype(np.float32)
			disp = disp - disp.min()
			disp = cv2.blur(disp / disp.max(), ksize=(3, 3)) * disp.max()
			disp = (disp / disp.max()) * 3.0
			depth = 1. / np.maximum(disp, 0.05)

			# rgb input
			img = np.asarray(img_rgb[count])

			# run sparse bilateral filter
			config['sparse_iter'] = 5
			config['filter_size'] = [7, 7, 5, 5, 5]
			config['sigma_s'] = 4.0
			config['sigma_r'] = 0.5
			vis_photos, vis_depths = sparse_bilateral_filtering(depth.copy(), img.copy(), config, num_iter=config['sparse_iter'], spdb=False)
			depth = vis_depths[-1]

			#bilat_fn = os.path.join(outpath, basename +'_bilatdepth.png')
			#cv2.imwrite(bilat_fn, depth)

			rt_info = write_mesh(img,
								depth,
								int_mtx,
								mesh_fi,
								config,
								rgb_model,
								depth_edge_model,
								depth_edge_model,
								depth_feat_model)

			if rt_info is not False and inpaint_vids:
				run_3dphoto_videos(mesh_fi, basename, outpath, 300, 40, 
					[0.03, 0.03, 0.05, 0.03], 
					['double-straight-line', 'double-straight-line', 'circle', 'circle'], 
					[0.00, 0.00, -0.015, -0.015], 
					[0.00, 0.00, -0.015, -0.00], 
					[-0.05, -0.05, -0.05, -0.05], 
					['dolly-zoom-in', 'zoom-in', 'circle', 'swing'], False, fnExt, vid_ssaa)
				
			devices.torch_gc()

	finally:
		del rgb_model
		rgb_model = None
		del depth_edge_model
		depth_edge_model = None
		del depth_feat_model
		depth_feat_model = None
		devices.torch_gc()

	return mesh_fi

def run_3dphoto_videos(mesh_fi, basename, outpath, num_frames, fps, crop_border, traj_types, x_shift_range, y_shift_range, z_shift_range, video_postfix, vid_dolly, fnExt, vid_ssaa):

	if platform.system() == 'Windows':
		vispy.use(app='PyQt5')
	elif platform.system() == 'Darwin':
		vispy.use('PyQt6')
	else:
		vispy.use(app='egl')

	# read ply
	global video_mesh_data, video_mesh_fn
	if video_mesh_fn == None or video_mesh_fn != mesh_fi:
		del video_mesh_data
		video_mesh_fn = mesh_fi
		video_mesh_data = read_mesh(mesh_fi)
		
	verts, colors, faces, Height, Width, hFov, vFov, mean_loc_depth = video_mesh_data

	original_w = output_w = W = Width
	original_h = output_h = H = Height
	int_mtx = np.array([[max(H, W), 0, W//2], [0, max(H, W), H//2], [0, 0, 1]]).astype(np.float32)
	if int_mtx.max() > 1:
		int_mtx[0, :] = int_mtx[0, :] / float(W)
		int_mtx[1, :] = int_mtx[1, :] / float(H)

	config = {}
	config['video_folder'] = outpath
	config['num_frames'] = num_frames
	config['fps'] = fps
	config['crop_border'] = crop_border
	config['traj_types'] = traj_types
	config['x_shift_range'] = x_shift_range
	config['y_shift_range'] = y_shift_range
	config['z_shift_range'] = z_shift_range
	config['video_postfix'] = video_postfix
	config['ssaa'] = vid_ssaa

	# from inpaint.utils.get_MiDaS_samples
	generic_pose = np.eye(4)
	assert len(config['traj_types']) == len(config['x_shift_range']) ==\
		len(config['y_shift_range']) == len(config['z_shift_range']) == len(config['video_postfix']), \
		"The number of elements in 'traj_types', 'x_shift_range', 'y_shift_range', 'z_shift_range' and \
			'video_postfix' should be equal."
	tgt_pose = [[generic_pose * 1]]
	tgts_poses = []
	for traj_idx in range(len(config['traj_types'])):
		tgt_poses = []
		sx, sy, sz = path_planning(config['num_frames'], config['x_shift_range'][traj_idx], config['y_shift_range'][traj_idx],
								config['z_shift_range'][traj_idx], path_type=config['traj_types'][traj_idx])
		for xx, yy, zz in zip(sx, sy, sz):
			tgt_poses.append(generic_pose * 1.)
			tgt_poses[-1][:3, -1] = np.array([xx, yy, zz])
		tgts_poses += [tgt_poses]    
	tgt_pose = generic_pose * 1

	# seems we only need the depthmap to calc mean_loc_depth, which is only used when doing 'dolly'
	# width and height are already in the ply file in the comments ..
	# might try to add the mean_loc_depth to it too 
	# did just that
	#mean_loc_depth = img_depth[img_depth.shape[0]//2, img_depth.shape[1]//2]

	print("Generating videos ..")

	normal_canvas, all_canvas = None, None
	videos_poses, video_basename = copy.deepcopy(tgts_poses), basename
	top = (original_h // 2 - int_mtx[1, 2] * output_h)
	left = (original_w // 2 - int_mtx[0, 2] * output_w)
	down, right = top + output_h, left + output_w
	border = [int(xx) for xx in [top, down, left, right]]
	normal_canvas, all_canvas, fn_saved = output_3d_photo(verts.copy(), colors.copy(), faces.copy(), copy.deepcopy(Height), copy.deepcopy(Width), copy.deepcopy(hFov), copy.deepcopy(vFov),
						copy.deepcopy(tgt_pose), config['video_postfix'], copy.deepcopy(generic_pose), copy.deepcopy(config['video_folder']),
						None, copy.deepcopy(int_mtx), config, None,
						videos_poses, video_basename, original_h, original_w, border=border, depth=None, normal_canvas=normal_canvas, all_canvas=all_canvas,
						mean_loc_depth=mean_loc_depth, dolly=vid_dolly, fnExt=fnExt)
	return fn_saved

# called from gen vid tab button
def run_makevideo(fn_mesh, vid_numframes, vid_fps, vid_traj, vid_shift, vid_border, dolly, vid_format, vid_ssaa):
	if len(fn_mesh) == 0 or not os.path.exists(fn_mesh):
		raise Exception("Could not open mesh.")

	# file type
	fnExt = "mp4" if vid_format == 0 else "webm"

	vid_ssaa = vid_ssaa + 1
	
	# traj type
	if vid_traj == 0:
		vid_traj = ['straight-line']
	elif vid_traj == 1:
		vid_traj = ['double-straight-line']
	elif vid_traj == 2:
		vid_traj = ['circle']

	num_fps = int(vid_fps)
	num_frames = int(vid_numframes)
	shifts = vid_shift.split(',')
	if len(shifts) != 3:
		raise Exception("Translate requires 3 elements.")
	x_shift_range = [ float(shifts[0]) ]
	y_shift_range = [ float(shifts[1]) ]
	z_shift_range = [ float(shifts[2]) ]
	
	borders = vid_border.split(',')
	if len(borders) != 4:
		raise Exception("Crop Border requires 4 elements.")
	crop_border = [float(borders[0]), float(borders[1]), float(borders[2]), float(borders[3])]

	# output path and filename mess ..
	basename = Path(fn_mesh).stem
	outpath = opts.outdir_samples or opts.outdir_extras_samples
	# unique filename
	basecount = get_next_sequence_number(outpath, basename)
	if basecount > 0: basecount = basecount - 1
	fullfn = None
	for i in range(500):
		fn = f"{basecount + i:05}" if basename == '' else f"{basename}-{basecount + i:04}"
		fullfn = os.path.join(outpath, f"{fn}_." + fnExt)
		if not os.path.exists(fullfn):
			break
	basename = Path(fullfn).stem
	basename = basename[:-1]
	
	print("Loading mesh ..")

	fn_saved = run_3dphoto_videos(fn_mesh, basename, outpath, num_frames, num_fps, crop_border, vid_traj, x_shift_range, y_shift_range, z_shift_range, [''], dolly, fnExt, vid_ssaa)

	return fn_saved[-1], fn_saved[-1], ''


# called from depth tab
def run_generate(depthmap_mode, 
                depthmap_image,
                image_batch,
                depthmap_batch_input_dir,
                depthmap_batch_output_dir,
                compute_device,
                model_type,
                net_width,
                net_height,
                match_size,
                boost,
                invert_depth,
                clipdepth,
                clipthreshold_far,
                clipthreshold_near,
                combine_output,
                combine_output_axis,
                save_depth,
                show_depth,
                show_heat,
                gen_stereo,
                stereo_modes,
                stereo_divergence,
                stereo_fill,
                stereo_balance,
                inpaint,
                inpaint_vids,
                background_removal,
                save_background_removal_masks,
                gen_normal,

                background_removal_model,
                pre_depth_background_removal,
                vid_format,
                vid_ssaa,
                custom_depthmap, 
                custom_depthmap_img,
                depthmap_batch_reuse,
                gen_mesh, mesh_occlude, mesh_spherical
                ):

				
	# file type
	fnExt = "mp4" if vid_format == 0 else "webm"

	vid_ssaa = vid_ssaa + 1

	imageArr = []
	# Also keep track of original file names
	imageNameArr = []
	outputs = []

	if depthmap_mode == 1:
		#convert file to pillow image
		for img in image_batch:
			image = Image.open(os.path.abspath(img.name))
			imageArr.append(image)
			imageNameArr.append(os.path.splitext(img.orig_name)[0])
	elif depthmap_mode == 2:
		assert not shared.cmd_opts.hide_ui_dir_config, '--hide-ui-dir-config option must be disabled'

		if depthmap_batch_input_dir == '':
			return outputs, "Please select an input directory.", ''
		image_list = shared.listfiles(depthmap_batch_input_dir)
		for img in image_list:
			try:
				image = Image.open(img)
			except Exception:
				continue
			imageArr.append(image)
			imageNameArr.append(img)
	else:
		imageArr.append(depthmap_image)
		imageNameArr.append(None)

	if depthmap_mode == 2 and depthmap_batch_output_dir != '':
		outpath = depthmap_batch_output_dir
	else:
		outpath = opts.outdir_samples or opts.outdir_extras_samples

	background_removed_images = []
	if background_removal:
		if pre_depth_background_removal:
			imageArr = batched_background_removal(imageArr, background_removal_model)
			background_removed_images = imageArr
		else:
			background_removed_images = batched_background_removal(imageArr, background_removal_model)

	outputs, mesh_fi, meshsimple_fi = run_depthmap(
        None, outpath, imageArr, imageNameArr,
        compute_device, model_type, net_width, net_height, match_size, boost, invert_depth, clipdepth, clipthreshold_far, clipthreshold_near, combine_output, combine_output_axis, save_depth, show_depth, show_heat, gen_stereo, stereo_modes, stereo_divergence, stereo_fill, stereo_balance, inpaint, inpaint_vids, background_removal, save_background_removal_masks, gen_normal,
        background_removed_images, fnExt, vid_ssaa, custom_depthmap, custom_depthmap_img, depthmap_batch_reuse, gen_mesh, mesh_occlude, mesh_spherical)

	# use inpainted 3d mesh to show in 3d model output when enabled in settings
	if hasattr(opts, 'depthmap_script_show_3d_inpaint') and opts.depthmap_script_show_3d_inpaint and mesh_fi != None and len(mesh_fi) > 0:
			meshsimple_fi = mesh_fi

	# don't show 3dmodel when disabled in settings
	if hasattr(opts, 'depthmap_script_show_3d') and not opts.depthmap_script_show_3d:
			meshsimple_fi = None

	return outputs, mesh_fi, meshsimple_fi, plaintext_to_html('info'), ''

def unload_models():
	global depthmap_model_depth, depthmap_model_pix2pix, depthmap_model_type
	depthmap_model_type = -1
	del depthmap_model_depth
	del depthmap_model_pix2pix
	depthmap_model_depth = None
	depthmap_model_pix2pix = None
	gc.collect()
	devices.torch_gc()

def clear_mesh():
	return None

def on_ui_settings():
    section = ('depthmap-script', "Depthmap extension")
    shared.opts.add_option("depthmap_script_keepmodels", shared.OptionInfo(False, "Keep depth models loaded.", section=section))
    shared.opts.add_option("depthmap_script_boost_rmax", shared.OptionInfo(1600, "Maximum wholesize for boost (Rmax)", section=section))
    shared.opts.add_option("depthmap_script_save_ply", shared.OptionInfo(False, "Save additional PLY file with 3D inpainted mesh.", section=section))
    shared.opts.add_option("depthmap_script_show_3d", shared.OptionInfo(True, "Enable showing 3D Meshes in output tab. (Experimental)", section=section))
    shared.opts.add_option("depthmap_script_show_3d_inpaint", shared.OptionInfo(True, "Also show 3D Inpainted Mesh in 3D Mesh output tab. (Experimental)", section=section))
    shared.opts.add_option("depthmap_script_mesh_maxsize", shared.OptionInfo(2048, "Max size for generating simple mesh.", section=section))

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as depthmap_interface:
        dummy_component = gr.Label(visible=False)
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="mode_depthmap"):
                    with gr.TabItem('Single Image'):
                        with gr.Row():
                            depthmap_image = gr.Image(label="Source", source="upload", interactive=True, type="pil", elem_id="depthmap_input_image")
                            with gr.Group(visible=False) as custom_depthmap_row_0:
                                custom_depthmap_img = gr.File(label="Custom DepthMap", file_count="single", interactive=True, type="file")
                        custom_depthmap = gr.Checkbox(label="Use custom DepthMap",value=False)

                    with gr.TabItem('Batch Process'):
                        image_batch = gr.File(label="Batch Process", file_count="multiple", interactive=True, type="file")

                    with gr.TabItem('Batch from Directory'):
                        depthmap_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, placeholder="A directory on the same machine where the server is running.")
                        depthmap_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, placeholder="Leave blank to save images to the default path.")
                        depthmap_batch_reuse = gr.Checkbox(label="Skip generation and use (edited/custom) depthmaps in output directory when a file exists.",value=True)

                submit = gr.Button('Generate', elem_id="depthmap_generate", variant='primary')

				# insert main panel
                compute_device, model_type, net_width, net_height, match_size, boost, invert_depth, clipdepth, clipthreshold_far, clipthreshold_near, combine_output, combine_output_axis, save_depth, show_depth, show_heat, gen_stereo, stereo_modes, stereo_divergence, stereo_fill, stereo_balance, inpaint, inpaint_vids, background_removal, save_background_removal_masks, gen_normal, pre_depth_background_removal, background_removal_model, gen_mesh, mesh_occlude, mesh_spherical = main_ui_panel(True)

                unloadmodels = gr.Button('Unload models', elem_id="depthmap_unloadmodels")

            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="mode_depthmap_output"):
                    with gr.TabItem('Depth Output'):

                        with gr.Group():
                            result_images = gr.Gallery(label='Output', show_label=False, elem_id=f"depthmap_gallery").style(grid=4)
                        with gr.Column():
                            html_info_x = gr.HTML()
                            html_info = gr.HTML()

                    with gr.TabItem('3D Mesh'):
                        with gr.Group():
                            result_depthmesh = gr.Model3D(label="3d Mesh", clear_color=[1.0, 1.0, 1.0, 1.0])
                            with gr.Row():
                                #loadmesh = gr.Button('Load')
                                clearmesh = gr.Button('Clear')
    
                    with gr.TabItem('Generate video'):
                        # generate video
                        with gr.Group():
                            with gr.Row():
                                gr.Markdown("Generate video from inpainted(!) mesh.")
                            with gr.Row():
                                depth_vid = gr.Video(interactive=False)
                            with gr.Column():
                                vid_html_info_x = gr.HTML()
                                vid_html_info = gr.HTML()
                                fn_mesh = gr.Textbox(label="Input Mesh (.ply | .obj)", **shared.hide_dirs, placeholder="A file on the same machine where the server is running.")
                            with gr.Row():
                                vid_numframes = gr.Textbox(label="Number of frames", value="300")
                                vid_fps = gr.Textbox(label="Framerate", value="40")
                                vid_format = gr.Dropdown(label="Format", choices=['mp4', 'webm'], value='mp4', type="index", elem_id="video_format")
                                vid_ssaa = gr.Dropdown(label="SSAA", choices=['1', '2', '3', '4'], value='3', type="index", elem_id="video_ssaa")
                            with gr.Row():
                                vid_traj = gr.Dropdown(label="Trajectory", choices=['straight-line', 'double-straight-line', 'circle'], value='double-straight-line', type="index", elem_id="video_trajectory")
                                vid_shift = gr.Textbox(label="Translate: x, y, z", value="-0.015, 0.0, -0.05")
                                vid_border = gr.Textbox(label="Crop: top, left, bottom, right", value="0.03, 0.03, 0.05, 0.03")
                                vid_dolly = gr.Checkbox(label="Dolly", value=False, elem_classes="smalltxt")
                            with gr.Row():
                                submit_vid = gr.Button('Generate Video', elem_id="depthmap_generatevideo", variant='primary')

        def custom_depthmap_visibility(v):
            return custom_depthmap_row_0.update(visible=v)
        custom_depthmap.change(
            fn=custom_depthmap_visibility,
            inputs=[custom_depthmap],
            outputs=[custom_depthmap_row_0]
        )

        unloadmodels.click(
            fn=unload_models,
            inputs=[],
            outputs=[]
        )
	
        clearmesh.click(
            fn=clear_mesh,
            inputs=[],
            outputs=[result_depthmesh]
        )

        submit.click(
            fn=wrap_gradio_gpu_call(run_generate),
            _js="get_depthmap_tab_index",
            inputs=[
                dummy_component,
                depthmap_image,
                image_batch,
                depthmap_batch_input_dir,
                depthmap_batch_output_dir,

				compute_device,
				model_type,
				net_width, 
				net_height, 
				match_size,
                boost,
                invert_depth,
                clipdepth,
                clipthreshold_far,
                clipthreshold_near,
                combine_output,
                combine_output_axis,
                save_depth,
				show_depth, 
				show_heat,
				gen_stereo,
				stereo_modes,
				stereo_divergence,
				stereo_fill,
				stereo_balance,
				inpaint,
				inpaint_vids,
                background_removal,
                save_background_removal_masks,
                gen_normal,

                background_removal_model,
				pre_depth_background_removal,
				vid_format,
				vid_ssaa,
				custom_depthmap, 
				custom_depthmap_img,
				depthmap_batch_reuse,
				gen_mesh, mesh_occlude, mesh_spherical
            ],
            outputs=[
                result_images,
				fn_mesh,
				result_depthmesh,
                html_info_x,
                html_info
            ]
        )

        submit_vid.click(
            fn=wrap_gradio_gpu_call(run_makevideo),
            inputs=[
				fn_mesh,
				vid_numframes,
				vid_fps,
				vid_traj,
				vid_shift,
				vid_border,
				vid_dolly,
				vid_format,
				vid_ssaa
            ],
            outputs=[
                depth_vid,
                vid_html_info_x,
                vid_html_info
            ]
        )

    return (depthmap_interface , "Depth", "depthmap_interface"),

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)


def batched_background_removal(inimages, model_name):
	print('creating background masks')
	outimages = []

	# model path and name
	bg_model_dir = Path.joinpath(Path().resolve(), "models/rem_bg")
	os.makedirs(bg_model_dir, exist_ok=True)
	os.environ["U2NET_HOME"] = str(bg_model_dir)
	
	#starting a session
	background_removal_session = new_session(model_name)
	for count in range(0, len(inimages)):
		bg_remove_img = np.array(remove(inimages[count], session=background_removal_session))
		outimages.append(Image.fromarray(bg_remove_img))
	#The line below might be redundant
	del background_removal_session
	return outimages

def download_file(filename, url):
	print("Downloading", url, "to", filename)
	torch.hub.download_url_to_file(url, filename)
	# check if file exists
	if not os.path.exists(filename):
		raise RuntimeError('Download failed. Try again later or manually download the file to that location.')

def estimatezoedepth(img, model, w, h):
	#x = transforms.ToTensor()(img).unsqueeze(0)
	#x = x.type(torch.float32)
	#x.to(device)
	#prediction = model.infer(x)
	model.core.prep.resizer._Resize__width = w
	model.core.prep.resizer._Resize__height = h
	prediction = model.infer_pil(img)

	return prediction

def scale_torch(img):
	"""
	Scale the image and output it in torch.tensor.
	:param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
	:param scale: the scale factor. float
	:return: img. [C, H, W]
	"""
	if len(img.shape) == 2:
		img = img[np.newaxis, :, :]
	if img.shape[2] == 3:
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
		img = transform(img.astype(np.float32))
	else:
		img = img.astype(np.float32)
		img = torch.from_numpy(img)
	return img
	
def estimateleres(img, model, w, h):
	# leres transform input
	rgb_c = img[:, :, ::-1].copy()
	A_resize = cv2.resize(rgb_c, (w, h))
	img_torch = scale_torch(A_resize)[None, :, :, :] 
	
	# compute
	with torch.no_grad():
		if device == torch.device("cuda"):
			img_torch = img_torch.cuda()
		prediction = model.depth_model(img_torch)

	prediction = prediction.squeeze().cpu().numpy()
	prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

	return prediction

def estimatemidas(img, model, w, h, resize_mode, normalization):
	# init transform
	transform = Compose(
		[
			Resize(
				w,
				h,
				resize_target=None,
				keep_aspect_ratio=True,
				ensure_multiple_of=32,
				resize_method=resize_mode,
				image_interpolation_method=cv2.INTER_CUBIC,
			),
			normalization,
			PrepareForNet(),
		]
	)

	# transform input
	img_input = transform({"image": img})["image"]

	# compute
	precision_scope = torch.autocast if shared.cmd_opts.precision == "autocast" and device == torch.device("cuda") else contextlib.nullcontext
	with torch.no_grad(), precision_scope("cuda"):
		sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
		if device == torch.device("cuda"):
			sample = sample.to(memory_format=torch.channels_last) 
			if not cmd_opts.no_half:
				sample = sample.half()
		prediction = model.forward(sample)
		prediction = (
			torch.nn.functional.interpolate(
				prediction.unsqueeze(1),
				size=img.shape[:2],
				mode="bicubic",
				align_corners=False,
			)
			.squeeze()
			.cpu()
			.numpy()
		)

	return prediction

def estimatemidasBoost(img, model, w, h):
	# init transform
    transform = Compose(
        [
            Resize(
                w,
                h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

	# transform input
    img_input = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        if device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last) 
        prediction = model.forward(sample)

    prediction = prediction.squeeze().cpu().numpy()    
    prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # normalization
    depth_min = prediction.min()
    depth_max = prediction.max()

    if depth_max - depth_min > np.finfo("float").eps:
        prediction = (prediction - depth_min) / (depth_max - depth_min)
    else:
        prediction = 0

    return prediction

def generatemask(size):
    # Generates a Guassian mask
    mask = np.zeros(size, dtype=np.float32)
    sigma = int(size[0]/16)
    k_size = int(2 * np.ceil(2 * int(size[0]/16)) + 1)
    mask[int(0.15*size[0]):size[0] - int(0.15*size[0]), int(0.15*size[1]): size[1] - int(0.15*size[1])] = 1
    mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.astype(np.float32)
    return mask

def resizewithpool(img, size):
    i_size = img.shape[0]
    n = int(np.floor(i_size/size))

    out = skimage.measure.block_reduce(img, (n, n), np.max)
    return out

def rgb2gray(rgb):
    # Converts rgb to gray
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def calculateprocessingres(img, basesize, confidence=0.1, scale_threshold=3, whole_size_threshold=3000):
    # Returns the R_x resolution described in section 5 of the main paper.

    # Parameters:
    #    img :input rgb image
    #    basesize : size the dilation kernel which is equal to receptive field of the network.
    #    confidence: value of x in R_x; allowed percentage of pixels that are not getting any contextual cue.
    #    scale_threshold: maximum allowed upscaling on the input image ; it has been set to 3.
    #    whole_size_threshold: maximum allowed resolution. (R_max from section 6 of the main paper)

    # Returns:
    #    outputsize_scale*speed_scale :The computed R_x resolution
    #    patch_scale: K parameter from section 6 of the paper

    # speed scale parameter is to process every image in a smaller size to accelerate the R_x resolution search
    speed_scale = 32
    image_dim = int(min(img.shape[0:2]))

    gray = rgb2gray(img)
    grad = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)) + np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    grad = cv2.resize(grad, (image_dim, image_dim), cv2.INTER_AREA)

    # thresholding the gradient map to generate the edge-map as a proxy of the contextual cues
    m = grad.min()
    M = grad.max()
    middle = m + (0.4 * (M - m))
    grad[grad < middle] = 0
    grad[grad >= middle] = 1

    # dilation kernel with size of the receptive field
    kernel = np.ones((int(basesize/speed_scale), int(basesize/speed_scale)), float)
    # dilation kernel with size of the a quarter of receptive field used to compute k
    # as described in section 6 of main paper
    kernel2 = np.ones((int(basesize / (4*speed_scale)), int(basesize / (4*speed_scale))), float)

    # Output resolution limit set by the whole_size_threshold and scale_threshold.
    threshold = min(whole_size_threshold, scale_threshold * max(img.shape[:2]))

    outputsize_scale = basesize / speed_scale
    for p_size in range(int(basesize/speed_scale), int(threshold/speed_scale), int(basesize / (2*speed_scale))):
        grad_resized = resizewithpool(grad, p_size)
        grad_resized = cv2.resize(grad_resized, (p_size, p_size), cv2.INTER_NEAREST)
        grad_resized[grad_resized >= 0.5] = 1
        grad_resized[grad_resized < 0.5] = 0

        dilated = cv2.dilate(grad_resized, kernel, iterations=1)
        meanvalue = (1-dilated).mean()
        if meanvalue > confidence:
            break
        else:
            outputsize_scale = p_size

    grad_region = cv2.dilate(grad_resized, kernel2, iterations=1)
    patch_scale = grad_region.mean()

    return int(outputsize_scale*speed_scale), patch_scale

# Generate a double-input depth estimation
def doubleestimate(img, size1, size2, pix2pixsize, model, net_type, pix2pixmodel):
    # Generate the low resolution estimation
    estimate1 = singleestimate(img, size1, model, net_type)
    # Resize to the inference size of merge network.
    estimate1 = cv2.resize(estimate1, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    # Generate the high resolution estimation
    estimate2 = singleestimate(img, size2, model, net_type)
    # Resize to the inference size of merge network.
    estimate2 = cv2.resize(estimate2, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    # Inference on the merge model
    pix2pixmodel.set_input(estimate1, estimate2)
    pix2pixmodel.test()
    visuals = pix2pixmodel.get_current_visuals()
    prediction_mapped = visuals['fake_B']
    prediction_mapped = (prediction_mapped+1)/2
    prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (
                torch.max(prediction_mapped) - torch.min(prediction_mapped))
    prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

    return prediction_mapped

# Generate a single-input depth estimation
def singleestimate(img, msize, model, net_type):
	if net_type == 0:
		return estimateleres(img, model, msize, msize)
	elif net_type >= 7:
		# np to PIL
		return estimatezoedepth(Image.fromarray(np.uint8(img * 255)).convert('RGB'), model, msize, msize)
	else:
		return estimatemidasBoost(img, model, msize, msize)

def applyGridpatch(blsize, stride, img, box):
    # Extract a simple grid patch.
    counter1 = 0
    patch_bound_list = {}
    for k in range(blsize, img.shape[1] - blsize, stride):
        for j in range(blsize, img.shape[0] - blsize, stride):
            patch_bound_list[str(counter1)] = {}
            patchbounds = [j - blsize, k - blsize, j - blsize + 2 * blsize, k - blsize + 2 * blsize]
            patch_bound = [box[0] + patchbounds[1], box[1] + patchbounds[0], patchbounds[3] - patchbounds[1],
                           patchbounds[2] - patchbounds[0]]
            patch_bound_list[str(counter1)]['rect'] = patch_bound
            patch_bound_list[str(counter1)]['size'] = patch_bound[2]
            counter1 = counter1 + 1
    return patch_bound_list

# Generating local patches to perform the local refinement described in section 6 of the main paper.
def generatepatchs(img, base_size):
    
    # Compute the gradients as a proxy of the contextual cues.
    img_gray = rgb2gray(img)
    whole_grad = np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)) +\
        np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3))

    threshold = whole_grad[whole_grad > 0].mean()
    whole_grad[whole_grad < threshold] = 0

    # We use the integral image to speed-up the evaluation of the amount of gradients for each patch.
    gf = whole_grad.sum()/len(whole_grad.reshape(-1))
    grad_integral_image = cv2.integral(whole_grad)

    # Variables are selected such that the initial patch size would be the receptive field size
    # and the stride is set to 1/3 of the receptive field size.
    blsize = int(round(base_size/2))
    stride = int(round(blsize*0.75))

    # Get initial Grid
    patch_bound_list = applyGridpatch(blsize, stride, img, [0, 0, 0, 0])

    # Refine initial Grid of patches by discarding the flat (in terms of gradients of the rgb image) ones. Refine
    # each patch size to ensure that there will be enough depth cues for the network to generate a consistent depth map.
    print("Selecting patches ...")
    patch_bound_list = adaptiveselection(grad_integral_image, patch_bound_list, gf)

    # Sort the patch list to make sure the merging operation will be done with the correct order: starting from biggest
    # patch
    patchset = sorted(patch_bound_list.items(), key=lambda x: getitem(x[1], 'size'), reverse=True)
    return patchset

def getGF_fromintegral(integralimage, rect):
    # Computes the gradient density of a given patch from the gradient integral image.
    x1 = rect[1]
    x2 = rect[1]+rect[3]
    y1 = rect[0]
    y2 = rect[0]+rect[2]
    value = integralimage[x2, y2]-integralimage[x1, y2]-integralimage[x2, y1]+integralimage[x1, y1]
    return value

# Adaptively select patches
def adaptiveselection(integral_grad, patch_bound_list, gf):
    patchlist = {}
    count = 0
    height, width = integral_grad.shape

    search_step = int(32/factor)

    # Go through all patches
    for c in range(len(patch_bound_list)):
        # Get patch
        bbox = patch_bound_list[str(c)]['rect']

        # Compute the amount of gradients present in the patch from the integral image.
        cgf = getGF_fromintegral(integral_grad, bbox)/(bbox[2]*bbox[3])

        # Check if patching is beneficial by comparing the gradient density of the patch to
        # the gradient density of the whole image
        if cgf >= gf:
            bbox_test = bbox.copy()
            patchlist[str(count)] = {}

            # Enlarge each patch until the gradient density of the patch is equal
            # to the whole image gradient density
            while True:

                bbox_test[0] = bbox_test[0] - int(search_step/2)
                bbox_test[1] = bbox_test[1] - int(search_step/2)

                bbox_test[2] = bbox_test[2] + search_step
                bbox_test[3] = bbox_test[3] + search_step

                # Check if we are still within the image
                if bbox_test[0] < 0 or bbox_test[1] < 0 or bbox_test[1] + bbox_test[3] >= height \
                        or bbox_test[0] + bbox_test[2] >= width:
                    break

                # Compare gradient density
                cgf = getGF_fromintegral(integral_grad, bbox_test)/(bbox_test[2]*bbox_test[3])
                if cgf < gf:
                    break
                bbox = bbox_test.copy()

            # Add patch to selected patches
            patchlist[str(count)]['rect'] = bbox
            patchlist[str(count)]['size'] = bbox[2]
            count = count + 1
    
    # Return selected patches
    return patchlist

def impatch(image, rect):
    # Extract the given patch pixels from a given image.
    w1 = rect[0]
    h1 = rect[1]
    w2 = w1 + rect[2]
    h2 = h1 + rect[3]
    image_patch = image[h1:h2, w1:w2]
    return image_patch

class ImageandPatchs:
    def __init__(self, root_dir, name, patchsinfo, rgb_image, scale=1):
        self.root_dir = root_dir
        self.patchsinfo = patchsinfo
        self.name = name
        self.patchs = patchsinfo
        self.scale = scale

        self.rgb_image = cv2.resize(rgb_image, (round(rgb_image.shape[1]*scale), round(rgb_image.shape[0]*scale)),
                                    interpolation=cv2.INTER_CUBIC)

        self.do_have_estimate = False
        self.estimation_updated_image = None
        self.estimation_base_image = None

    def __len__(self):
        return len(self.patchs)

    def set_base_estimate(self, est):
        self.estimation_base_image = est
        if self.estimation_updated_image is not None:
            self.do_have_estimate = True

    def set_updated_estimate(self, est):
        self.estimation_updated_image = est
        if self.estimation_base_image is not None:
            self.do_have_estimate = True

    def __getitem__(self, index):
        patch_id = int(self.patchs[index][0])
        rect = np.array(self.patchs[index][1]['rect'])
        msize = self.patchs[index][1]['size']

        ## applying scale to rect:
        rect = np.round(rect * self.scale)
        rect = rect.astype('int')
        msize = round(msize * self.scale)

        patch_rgb = impatch(self.rgb_image, rect)
        if self.do_have_estimate:
            patch_whole_estimate_base = impatch(self.estimation_base_image, rect)
            patch_whole_estimate_updated = impatch(self.estimation_updated_image, rect)
            return {'patch_rgb': patch_rgb, 'patch_whole_estimate_base': patch_whole_estimate_base,
                    'patch_whole_estimate_updated': patch_whole_estimate_updated, 'rect': rect,
                    'size': msize, 'id': patch_id}
        else:
            return {'patch_rgb': patch_rgb, 'rect': rect, 'size': msize, 'id': patch_id}

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        """
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        """

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        #self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        #if len(opt.gpu_ids) > 0:
        #    torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt


def estimateboost(img, model, model_type, pix2pixmodel):
	# get settings
	if hasattr(opts, 'depthmap_script_boost_rmax'):
		whole_size_threshold = opts.depthmap_script_boost_rmax
		
	if model_type == 0: #leres
		net_receptive_field_size = 448
		patch_netsize = 2 * net_receptive_field_size
	elif model_type == 1: #dpt_beit_large_512
		net_receptive_field_size = 512
		patch_netsize = 2 * net_receptive_field_size
	else: #other midas
		net_receptive_field_size = 384
		patch_netsize = 2 * net_receptive_field_size

	gc.collect()
	devices.torch_gc()

	# Generate mask used to smoothly blend the local pathc estimations to the base estimate.
	# It is arbitrarily large to avoid artifacts during rescaling for each crop.
	mask_org = generatemask((3000, 3000))
	mask = mask_org.copy()

	# Value x of R_x defined in the section 5 of the main paper.
	r_threshold_value = 0.2
	#if R0:
	#	r_threshold_value = 0

	input_resolution = img.shape
	scale_threshold = 3  # Allows up-scaling with a scale up to 3

	# Find the best input resolution R-x. The resolution search described in section 5-double estimation of the main paper and section B of the
	# supplementary material.
	whole_image_optimal_size, patch_scale = calculateprocessingres(img, net_receptive_field_size, r_threshold_value, scale_threshold, whole_size_threshold)

	print('wholeImage being processed in :', whole_image_optimal_size)

	# Generate the base estimate using the double estimation.
	whole_estimate = doubleestimate(img, net_receptive_field_size, whole_image_optimal_size, pix2pixsize, model, model_type, pix2pixmodel)

	# Compute the multiplier described in section 6 of the main paper to make sure our initial patch can select
	# small high-density regions of the image.
	global factor
	factor = max(min(1, 4 * patch_scale * whole_image_optimal_size / whole_size_threshold), 0.2)
	print('Adjust factor is:', 1/factor)

	# Compute the default target resolution.
	if img.shape[0] > img.shape[1]:
		a = 2 * whole_image_optimal_size
		b = round(2 * whole_image_optimal_size * img.shape[1] / img.shape[0])
	else:
		a = round(2 * whole_image_optimal_size * img.shape[0] / img.shape[1])
		b = 2 * whole_image_optimal_size
	b = int(round(b / factor))
	a = int(round(a / factor))

	"""
	# recompute a, b and saturate to max res.
	if max(a,b) > max_res:
		print('Default Res is higher than max-res: Reducing final resolution')
		if img.shape[0] > img.shape[1]:
			a = max_res
			b = round(option.max_res * img.shape[1] / img.shape[0])
		else:
			a = round(option.max_res * img.shape[0] / img.shape[1])
			b = max_res
		b = int(b)
		a = int(a)
	"""

	img = cv2.resize(img, (b, a), interpolation=cv2.INTER_CUBIC)

	# Extract selected patches for local refinement
	base_size = net_receptive_field_size * 2
	patchset = generatepatchs(img, base_size)

	print('Target resolution: ', img.shape)

	# Computing a scale in case user prompted to generate the results as the same resolution of the input.
	# Notice that our method output resolution is independent of the input resolution and this parameter will only
	# enable a scaling operation during the local patch merge implementation to generate results with the same resolution
	# as the input.
	"""
	if output_resolution == 1:
		mergein_scale = input_resolution[0] / img.shape[0]
		print('Dynamicly change merged-in resolution; scale:', mergein_scale)
	else:
		mergein_scale = 1
	"""
	# always rescale to input res for now
	mergein_scale = input_resolution[0] / img.shape[0]

	imageandpatchs = ImageandPatchs('', '', patchset, img, mergein_scale)
	whole_estimate_resized = cv2.resize(whole_estimate, (round(img.shape[1]*mergein_scale),
										round(img.shape[0]*mergein_scale)), interpolation=cv2.INTER_CUBIC)
	imageandpatchs.set_base_estimate(whole_estimate_resized.copy())
	imageandpatchs.set_updated_estimate(whole_estimate_resized.copy())

	print('Resulting depthmap resolution will be :', whole_estimate_resized.shape[:2])
	print('patches to process: '+str(len(imageandpatchs)))

	# Enumerate through all patches, generate their estimations and refining the base estimate.
	for patch_ind in range(len(imageandpatchs)):
		
		# Get patch information
		patch = imageandpatchs[patch_ind] # patch object
		patch_rgb = patch['patch_rgb'] # rgb patch
		patch_whole_estimate_base = patch['patch_whole_estimate_base'] # corresponding patch from base
		rect = patch['rect'] # patch size and location
		patch_id = patch['id'] # patch ID
		org_size = patch_whole_estimate_base.shape # the original size from the unscaled input
		print('\t processing patch', patch_ind, '/', len(imageandpatchs)-1, '|', rect)

		# We apply double estimation for patches. The high resolution value is fixed to twice the receptive
		# field size of the network for patches to accelerate the process.
		patch_estimation = doubleestimate(patch_rgb, net_receptive_field_size, patch_netsize, pix2pixsize, model, model_type, pix2pixmodel)
		patch_estimation = cv2.resize(patch_estimation, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)
		patch_whole_estimate_base = cv2.resize(patch_whole_estimate_base, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

		# Merging the patch estimation into the base estimate using our merge network:
		# We feed the patch estimation and the same region from the updated base estimate to the merge network
		# to generate the target estimate for the corresponding region.
		pix2pixmodel.set_input(patch_whole_estimate_base, patch_estimation)

		# Run merging network
		pix2pixmodel.test()
		visuals = pix2pixmodel.get_current_visuals()

		prediction_mapped = visuals['fake_B']
		prediction_mapped = (prediction_mapped+1)/2
		prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

		mapped = prediction_mapped

		# We use a simple linear polynomial to make sure the result of the merge network would match the values of
		# base estimate
		p_coef = np.polyfit(mapped.reshape(-1), patch_whole_estimate_base.reshape(-1), deg=1)
		merged = np.polyval(p_coef, mapped.reshape(-1)).reshape(mapped.shape)

		merged = cv2.resize(merged, (org_size[1],org_size[0]), interpolation=cv2.INTER_CUBIC)

		# Get patch size and location
		w1 = rect[0]
		h1 = rect[1]
		w2 = w1 + rect[2]
		h2 = h1 + rect[3]

		# To speed up the implementation, we only generate the Gaussian mask once with a sufficiently large size
		# and resize it to our needed size while merging the patches.
		if mask.shape != org_size:
			mask = cv2.resize(mask_org, (org_size[1],org_size[0]), interpolation=cv2.INTER_LINEAR)

		tobemergedto = imageandpatchs.estimation_updated_image

		# Update the whole estimation:
		# We use a simple Gaussian mask to blend the merged patch region with the base estimate to ensure seamless
		# blending at the boundaries of the patch region.
		tobemergedto[h1:h2, w1:w2] = np.multiply(tobemergedto[h1:h2, w1:w2], 1 - mask) + np.multiply(merged, mask)
		imageandpatchs.set_updated_estimate(tobemergedto)

	# output
	return cv2.resize(imageandpatchs.estimation_updated_image, (input_resolution[1], input_resolution[0]), interpolation=cv2.INTER_CUBIC)

def pano_depth_to_world_points(depth):
    """
    360 depth to world points
    given 2D depth is an equirectangular projection of a spherical image
    Treat depth as radius
    longitude : -pi to pi
    latitude : -pi/2 to pi/2
    """

    # Convert depth to radius
    radius = depth.flatten()

    lon = np.linspace(-np.pi, np.pi, depth.shape[1])
    lat = np.linspace(-np.pi/2, np.pi/2, depth.shape[0])

    lon, lat = np.meshgrid(lon, lat)
    lon = lon.flatten()
    lat = lat.flatten()

    # Convert to cartesian coordinates
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)

    pts3d = np.stack([x, y, z], axis=1)

    return pts3d

def depth_edges_mask(depth):
    """Returns a mask of edges in the depth map.
    Args:
    depth: 2D numpy array of shape (H, W) with dtype float32.
    Returns:
    mask: 2D numpy array of shape (H, W) with dtype bool.
    """
    # Compute the x and y gradients of the depth map.
    depth_dx, depth_dy = np.gradient(depth)
    # Compute the gradient magnitude.
    depth_grad = np.sqrt(depth_dx ** 2 + depth_dy ** 2)
    # Compute the edge mask.
    mask = depth_grad > 0.05
    return mask

def create_mesh(image, depth, keep_edges=False, spherical=False):
	maxsize = 1024
	if hasattr(opts, 'depthmap_script_mesh_maxsize'):
		maxsize = opts.depthmap_script_mesh_maxsize

	# limit the size of the input image
	image.thumbnail((maxsize, maxsize))  

	if not spherical:
		pts3d = depth_to_points(depth[None])
	else:
		pts3d = pano_depth_to_world_points(depth)

	pts3d = pts3d.reshape(-1, 3)

	verts = pts3d.reshape(-1, 3)
	image = np.array(image)
	if keep_edges:
		triangles = create_triangles(image.shape[0], image.shape[1])
	else:
		triangles = create_triangles(image.shape[0], image.shape[1], mask=~depth_edges_mask(depth))
	colors = image.reshape(-1, 3)

	mesh = trimesh.Trimesh(vertices=verts, faces=triangles, vertex_colors=colors)

	# rotate 90deg over X when spherical
	if spherical:
		angle = math.pi / 2
		direction = [1, 0, 0]
		center = [0, 0, 0]
		rot_matrix = transformations.rotation_matrix(angle, direction, center)
		mesh.apply_transform(rot_matrix)

	return mesh

def save_mesh_obj(fn, mesh):
		mesh.export(fn)
