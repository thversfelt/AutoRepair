# AutoRepair

## Development

To setup the development evironment for AutoRepair, follow the following steps.

## Tools:

For both **Windows** and **McOS**, you will need the following tools:

- [Visual Studio Code](https://code.visualstudio.com/download) is the recommended editor for developing, running and debugging AutoRepair. So, make sure to have an up-to-date version of Visual Studio Code.

- [Git](https://git-scm.com/) is used for version control and should be installed.

- [Python ](https://www.python.org/downloads/) is the used programming language to run AutoRepair. 

- [pip](https://pypi.org/project/pip/) is the package manager used to download and install dependencies, as well as maintaining a virtual Python environment. The latest Python installation should come with a pre-installed version of pip.

For **MacOS**, you will also need:

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html) 4.11.0 for Python 3.7.x. Conda is required to install PyTorch 1.10 and related dependencies (https://pytorch.org/get-started/locally/)

## Installation:

### For Windows:

1. Download or clone this repository to any folder: ```git clone https://github.com/thversfelt/AutoRepair.git```

2. Navigate to the AutoRepair folder and create a virtual Python environment by running the following command: ```python -m venv env```

3. Activate the newly created virtual environment by running the following command: ```.\env\Scripts\activate```

4. Install the depencencies by running: ```pip install -r requirements.txt```

### For MacOS:

1. Download or clone this repository to any folder: ```git clone https://github.com/thversfelt/AutoRepair.git```

2. Navigate to the AutoRepair folder and create a virtual Python environment by running the following command: ```python3 -m venv env```

3. Activate the newly created virtual environment by running the following command: ```source env/bin/activate```

4. Install the depencencies by running: ```conda install pytorch torchvision torchaudio -c pytorch```

5. Install the ```l5kit``` module by running: ```pip3 install l5kit```

## Debugging

The Visual Studio Code launch configuration (```.vscode\launch.json```) allows you to run the example script (```benchmark\example.py```). Press ```F5``` to debug any scene of the benchmark, where the selected scenes can be changed in the example script.

## Description

As of now, AutoRepair is divided into 3 folders, where each folder corresponds to a major part of the thesis. 

- ```benchmark```: the benchmark that is based on woven planet's level-5 autonomous-driving dataset.
    - ```l5kit_data```: contains a sample (subset) dataset with 100 scenes for development purposes.
    - ```custom_map_api```: an api that contains important functions that are able to relate data to the provided road network.
    - ```ego_model```: ...
- ```pyariel```: the python implementation of ARIEL.
- ```autorepair```: the (to-be-implemented) variation of ARIEL.
