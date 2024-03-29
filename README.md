### <div align='center'> Planning & Decision Making - RO47005 <br/> 16-01-2023 </div>

# <div align='center'> 2022_Q2_PDM-Project </div>
#### <div align='center'><i>Project about global and local path planning through an environment with obstacles. </i></div>

### <div align ='center'> Group 027:</div>
#### <div align='center'>Annelouk van Mierlo: 4693746 </br> Caspar Meijer: 4719298 </br> Jurjen Scharringa: 4708652 </br> Marijn de Rijk: 4888871 </div>

This project is made for the group assignment of the course RO47005 at the TU Delft. 
The aim for this project is to implement motion planning algorithms for a mobile manipulator, allowing it to navigate through a simulated environment and perform an action with the robot arm.
The mobile manipulator that is used is the <i>Albert Robot</i>, provided in [Max Spahn's package](https://github.com/maxspahn/gym_envs_urdf).


This file will contain the following contents:
- Content description
- Installation guide
  - Windows
  - Linux
- Run guide

## Content description
The repository consists of the following files with corresponding descriptions:
- ```main.py```: main file to run the simulation
- ```urdf_env_helpers.py```: helper file to add obstacles and goals to the urfd environment
- ```global_path_planning.py```: to compute the path for <i>Albert</i> using RRT*Smart
- ```local_path_planning.py```: to follow the path found
- ```arm_kinematics.py```: forward and inverse kinematics equations needed to control the robot arm
- ``transformations.py``: contains functions to transform between global and robot coordinate frames
  
 
## Installation guide
The following pre-requisites are required:
- Python >3.6, <3.10
- git

Then, the repository needs to be downloaded into a directory of choice:
```
git clone https://github.com/Capsar/2022_Q2_PDM-Project.git
```
Depending on the system that you are using follow the next steps. 

### Windows
- Install [Microsoft Build Tools](https://visualstudio.microsoft.com/downloads/?q=build+tools) for version 14.0
- navigate to the folder in which you installed the repository: ``cd PATH\TO\REPOSITORY``
- Create new conda environment: ```conda env create -f environment_win.yml```
- Activate the new conda environment ```conda activate PDM_group27```
- Install newer version of ```networkx```: ```conda install networkx==2.8.8```

### Linux
- navigate to the folder in which you installed the repository: ``cd PATH/TO/REPOSITORY``
- Create new conda environment: ```conda env create --name PDM_group27 -f environment_linux.yaml```
- Activate the new conda environment ```conda activate PDM_group27```
- Install newer version of ```networkx```: ```conda install networkx==2.8.8```

The manual installment of ```networkx``` is required because ```urdfpy``` has a restriction on the ```networkx``` version. We are not aware of any complications using the newer version of ```networkx``` in regards with ```urdfpy```.
If no errors have occurred, then the repository is correctly installed. 

## Run guide
The simulation can by running ``python3 main.py``.  Make sure you are within the "group27" folder, and that the ```PDM_group27``` conda environment is activated.

### Selecting environment
Multiple simulation environments for the robot to navigate are provided. Options are:
- ```1```: standard
- ```2```: second pre-made environment
- ```3```: third pre-made environment

Environmnets can be chosen by setting the ```--environment``` argument.  To run the simulation in environment 3, for example, run ``python3 main.py --environment=3``.

### Arm only
There is also an option to skip the navigation of the mobile base and to look only at the robot arm path-following.  To do this, set the arm_only parameter as follows:

``python3 main.py --arm_only``

### Environment & Path pictures

[<img src="env_pictures/env1.png" alt="Environment 1" width="300"/>](https://github.com/Capsar/2022_Q2_PDM-Project/blob/main/env_pictures/env1.png)
[<img src="env_pictures/env2.png" alt="Environment 2" width="315"/>](https://github.com/Capsar/2022_Q2_PDM-Project/blob/main/env_pictures/env2.png)
[<img src="env_pictures/env3.png" alt="Environment 3" width="300"/>](https://github.com/Capsar/2022_Q2_PDM-Project/blob/main/env_pictures/env3.png)
[<img src="env_pictures/Arm_path_result.png" alt="3D Arm Path" width="330"/>](https://github.com/Capsar/2022_Q2_PDM-Project/blob/main/env_pictures/Arm_path_result.png)