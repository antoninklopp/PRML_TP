WIDER Database for Viola Jones faces detector

The WIDER/ folder contains :
	- train_info.dat
		file containing informations about training images such as :
			<path_to_image> <number_of_faces> (for each face) <left> <top> <width> <height>

	- val_info.dat
		file containing informations about validation images, same structured as train_info.dat

	- false_img/
		folder containing false images, i.e image which DO NOT contain faces (not a single one !!!)

	- [0, ..., 51]--<name folder>/
		folders containing pre-marked images, annotations being stored in .dat files

 For each event class, we randomly select 40%/10%/50% data as training, validation and testing sets.


Sources : http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/
 @inproceedings{yang2016wider,
 	Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
 	Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 	Title = {WIDER FACE: A Face Detection Benchmark},
 	Year = {2016}}
