# An Upgraded Version of Image-Matching-Toolbox 
This repo is developed from the [original toolbox](https://github.com/GrumpyZhou/image-matching-toolbox) of Zhou et. al.
I've made several updates as follows
- [x] Add evaluation of my image-matching method, [TopicFM+](https://github.com/TruongKhang/TopicFM)
- [x] Update homography estimation on HPatches. I added several options for the `cv2.findHomography` function
- [x] Add quantization step for the evaluation of Aachen Day-Night, based on the maximum confidence of keypoints.

We only provide the evaluation of TopicFM in this code. 

For more detailed description and other functions of the toolbox, please visit the original version.
## Installation
Firstly, setup the experimental environment:

	conda create -n immatch python=3.8
	conda activate immatch
	conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=10.2 -c pytorch
	pip install jupyter, matplotlib, opencv-python==4.7.0.72
	pip install pycolmap

To evaluate on Aachen Day-Night and Inloc, install `COLMAP` from the [original website](https://colmap.github.io/). Make sure that you install successfully by testing `colmap` in your terminal.

Secondly, install the dependencies of toolbox:

	cd image-matching-toolbox/
	git submodule update --init
	# ignore this step if you don't want evaluate other methods
	cd pretrained && bash download.sh && cd ..

Clone the code of TopicFM and put it into the `third_party` folder

	cd third_party && git clone https://github.com/TruongKhang/TopicFM 
	# change the folder name TopicFM --> topicfmv2
	mv TopicFM topicfmv2 && cd ..

Next, download the pretrained models of [TopicFM+](https://github.com/TruongKhang/TopicFM) and put them into `third_party/topicfmv2/pretrained/`. This toolbox can support evaluations of two models `third_party/topicfmv2/pretrained/topicfm_fast.ckpt` and `third_party/topicfmv2/pretrained/topicfm_plus.ckpt`.


Finally, install the toolbox as follows:

	python setup.py develop

**Notes**: when running the program, use `pip install <package-name>` if there are any uninstalled packages.

## Evaluation of TopicFM+

All settings of the model and datasets are specified in `configs/topicfmv2.yml`

### HPatches

	python -m immatch.eval_hpatches --gpu 0 --config 'topicfmv2' --task 'both' --h_solver 'cv' --ransac_thres 6 --root_dir . --odir 'outputs/hpatches'

### AAchen Day-Night v1.1

	python -m immatch.eval_aachen --gpu 0 --config 'topicfmv2' --colmap colmap --benchmark_name 'aachen_v1.1'

### InLoc

	python -m immatch.eval_inloc --gpu 0 --config 'topicfmv2'