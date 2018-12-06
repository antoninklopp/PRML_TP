WIDER Database for Viola Jones faces detector

The WIDER/ folder contains :
	- train_info.dat
		file containing informations about training images such as :
			<path_to_image>
			<number_of_faces>
			for each face :
				<left> <top> <width> <height>

	- val_info.dat
		file containing informations about validation images, same structured as train_info.dat

	- false_img/
		folder containing false images, i.e image which DO NOT contain faces (not a single one !!!)

	- [0, ..., 9]--<name folder>/
		folders containing pre-marked images, annotations being stored in .dat files
