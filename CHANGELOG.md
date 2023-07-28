## Changelog
### 0.4.3 video processing tab
 * Added an option to process videos directly from a video file. This leads to better results than batch-processing individual frames of a video. Allows generating depthmap videos, that can be used in further generations as custom depthmap videos.
 * UI improvements.
 * Extra stereoimage generation modes - enable in extension settings if you want to use them.
 * New stereoimage generation parameter - offset exponent. Setting it to 1 may produce more realistic outputs.
### 0.4.2
 * Added UI options for 2 additional rembg models.
 * Heatmap generation UI option is hidden - if you want to use it, please activate it in the extension settings.
 * Bugfixes.
### 0.4.1 standalone mode
 * Added ability to run DepthMap without WebUI. (Use main.py. Make sure all the dependencies are installed. The support is not feature-complete.)
 * NormalMap generation
### 0.4.0 large code refactor
 * UI improvements
 * Improved Batch from Directory, Clip and renormalize DepthMap
 * Slightly changed the behaviour of various options
 * Extension may partially work even if some of the dependencies are unmet

### 0.3.12
 * Fixed stereo image generation
 * Other bugfixes
### 0.3.11
 * 3D model viewer (Experimental!)
 * simple and fast (occluded) 3D mesh generation, support for equirectangular projection
      (accurate results with ZoeDepth models only, no boost, no custom maps)
 * default output format is now obj for inpainted mesh and simple mesh
### 0.3.10 
 * ZoeDepth support (with boost), 3 new models, best results so far
 * better heatmap
### 0.3.9
 * use existing/custom depthmaps in output dir for batch mode
 * custom depthmap support for single file
 * wavefront obj output support for inpainted mesh (enabled in settings)
 * option to generate all stereo formats at once
 * bugfix: convert single channel input image to rgb
 * renamed midas imports to fix conflict with deforum
 * ui cleanup
### 0.3.8 bugfix
 * bugfix in remove background path
### 0.3.7 new features
 * [rembg](https://github.com/danielgatis/rembg) Remove Background [PR](https://github.com/thygate/stable-diffusion-webui-depthmap-script/pull/78) by [@graemeniedermayer](https://github.com/graemeniedermayer) merged
 * setting to flip Left/Right SBS images
 * added missing parameter for 3d inpainting (repeat_inpaint_edge)
 * option to generate demo videos with mesh
### 0.3.6 new feature
 * implemented binary ply file format for the inpainted 3D mesh, big reduction in filesize and save/load times. 
 * added progress indicators to the inpainting process 
### 0.3.5 bugfix
 * create path to 3dphoto models before download (see [issue](https://github.com/thygate/stable-diffusion-webui-depthmap-script/issues/76))
### 0.3.4 new featues
 * depth clipping option (original idea by [@Extraltodeus](https://github.com/Extraltodeus))
 * by popular demand, 3D-Photo-Inpainting is now implemented
 * generate inpainted 3D mesh (PLY) and videos of said mesh 
### 0.3.3 bugfix and new midas models
 * updated to midas 3.1, bringing 2 new depth models (the 512 one eats VRAM for breakfast!)
 * fix Next-ViT dependency issue for new installs
 * extension no longer clones repositories, all dependencies are now contained in the extension
### 0.3.2 new feature and bugfixes
 * several bug fixes for apple silicon and other machines without cuda
 * NEW Stereo Image Generation techniques for gap filling by [@semjon00](https://github.com/semjon00) using polylines. (See [here](https://github.com/thygate/stable-diffusion-webui-depthmap-script/pull/56)) Significant improvement in quality.
### 0.3.1 bugfix
 * small speed increase for anaglyph creation
 * clone midas repo before midas 3.1 to fix issue (see [here](https://github.com/thygate/stable-diffusion-webui-depthmap-script/issues/55#issue-1510266008))
### 0.3.0 improved stereo image generation
 * New improved technique for generating stereo images and balancing distortion between eyes by [@semjon00](https://github.com/semjon00) (See [here](https://github.com/thygate/stable-diffusion-webui-depthmap-script/pull/51))
 * Substantial speedup of stereo image generation code using numba JIT
### 0.2.9 new feature 
 * 3D Stereo (side-by-side) and red/cyan anaglyph image generation.   
    (Thanks to [@sina-masoud-ansari](https://github.com/sina-masoud-ansari) for the tip! Discussion [here](https://github.com/thygate/stable-diffusion-webui-depthmap-script/discussions/45))
### 0.2.8 bugfix
 * boost (pix2pix) now also able to compute on cpu
 * res101 able to compute on cpu
### 0.2.7 separate tab
 * Depth Tab now available for easier stand-alone (batch) processing
### 0.2.6 ui layout and settings
 * added link to repo so more people find their way to the instructions.
 * boost rmax setting
### 0.2.5 bugfix
 * error checking on model download (now with progressbar)
### 0.2.4 high resolution depthmaps
 * multi-resolution merging is now implemented, significantly improving results!
 * res101 can now also compute on CPU
### 0.2.3 bugfix
 * path error on linux fixed
### 0.2.2 new features
 * added (experimental) support for AdelaiDepth/LeReS (GPU Only!)
 * new option to view depthmap as heatmap
 * optimised ui layout
### 0.2.1 bugfix
 * Correct seed is now used in filename and pnginfo when running batches. (see [issue](https://github.com/thygate/stable-diffusion-webui-depthmap-script/issues/35))
### 0.2.0 upgrade
 * the script is now an extension, enabling auto installation.
### 0.1.9 bugfixes
 * sd model moved to system memory while computing depthmap
 * memory leak/fragmentation issue fixed
 * recover from out of memory error
### 0.1.8 new options
 * net size can now be set as width and height, option to match input size, sliders now have the same range as generation parameters. (see usage below)
 * better error handling
### 0.1.7 bugfixes
 * batch img2img now works (see [issue](https://github.com/thygate/stable-diffusion-webui-depthmap-script/issues/21#issuecomment-1306445056))
 * generation parameters now only saved when enabled in settings
 * model memory freed explicitly at end of script
### 0.1.6 new option
 * option to invert depthmap (black=near, white=far), as required by some viewers.
### 0.1.5 bugfix
 * saving as any format other than PNG now always produces an 8 bit, 3 channel RGB image. A single channel 16 bit image is only supported when saving as PNG. (see [issue](https://github.com/thygate/stable-diffusion-webui-depthmap-script/issues/15#issuecomment-1304909019))
### 0.1.4 update
 * added support for `--no-half`. Now also works with cards that don't support half precision like GTX 16xx. ([verified](https://github.com/thygate/stable-diffusion-webui-depthmap-script/issues/12#issuecomment-1304656398))
### 0.1.3 bugfix
 * bugfix where some controls where not visible (see [issue](https://github.com/thygate/stable-diffusion-webui-depthmap-script/issues/11#issuecomment-1304400537))
### 0.1.2 new option
 * network size slider. higher resolution depth maps (see usage below)
### 0.1.1 bugfixes
 * overflow issue (see [here](https://github.com/thygate/stable-diffusion-webui-depthmap-script/issues/10) for details and examples of artifacts)
 * when not combining, depthmap is now saved as single channel 16 bit
### 0.1.0
 * initial version: script mode, supports generating depthmaps with 4 different midas models