# Surveillance Tracking
Video analysis software to identify and track people in footage recorded from stationary surveillance cameras

## Installation

Following are installation instructions:
* Install anaconda (Or use you package manager if conda is available, but seems not in most distros)
	* See official installation instructions: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
	* Miniconda downloads from: https://docs.conda.io/en/latest/miniconda.html
	* Linux Install : Run the following commands from the command line
		* wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
		* bash Miniconda3-latest-Linux-x86_64.sh
			* <enter> to review license
			* <space> multiple times to scroll through license page at a time
			* "yes" to accept license
			* <enter> to accept default install location
			* "yes" to run conda init
		* source ~/.bashrc # Or you can relogin
		* conda config --set auto_activate_base false
	* Windows Install: 
		* Download Miniconda for Python 3.8 x64 : https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
		* Open Anaconda Prompt (Miniconda 3) from the start menu
* git clone https://github.com/bjcosta/surveillance_tracking.git
* cd surveillance_tracking
* conda env update --file environment.yml
* conda activate surveillance_tracking


## Running

Assuming you followed the above instructions to install, you can now try and run it on the small sample sonar log file included in this repository using the command line below. Following sections will describe and show examples of the files produced.

* TODO


## Known Issues
* TODO


## Future work
There are a number of features I intend to get to over time though this is done in my spare time so we will see what I get to:
* TODO

