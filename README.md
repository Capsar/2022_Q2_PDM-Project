### <div align='center'> Planning & Decision Making - RO47005 <br/> 16-01-2023 </div>

# <div align='center'> 2022_Q2_PDM-Project </div>
### <div align='center'>Project about global and local path planning through an environment with obstacles. </div>

### <div align ='center'> Group 027:</div>
#### <div align='center'>Annelouk van Mierlo: 4693746 </br> Caspar Meijer: 4719298 </br> Jurjen Scharringa: 4708652 </br> Marijn de Rijk: 4888871 </div>

<div> This project is made for the group assignment of the course RO47005 at the TU Delft. 
The aim for this project is to implement motion planning algorithms for a mobile manipulator, allowing it to navigate through a simulated environment and perform an action with the robot arm.
The mobile manipulator that is used is the <i>Albert Robot</i> created by the <i>Urdf-Environment</i> [source].
</div>

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
- ```arm_kinematics.py```:
- ```arm_test.py```:
  
 
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
- Create new conda environment: ```conda env create -f environment.yml```

### Linux
- Create new conda environment: ```conda env create -f environment_linux.yml```

If no errors have occurred, then the repository is correctly installed. 

## Run guide
To run the programm, simply run ```main.py```. This will run the simulation with the standard environment. 
To change the standard environment into other pre-made environments change ```obstacle_setup=``` in ```main.py: line23 ``` into either:
- ```1```: standard
- ```2```: second pre-made environment
- ```3```: third pre-made environment
- ```"random"```: random obstacles added to environment

