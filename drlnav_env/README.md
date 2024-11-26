# drlnav_env

Install ubuntu20.04 and ros noetic on your computer.

### 🌄 Quick Start

#### Cmake:

```
catkin_make --only-pkg-with-deps drl_nav
```

Add Dependencies

```
echo "source `pwd`/devel/setup.bash" >> ~/.bashrc
```

#### Configure yaml

Enter envs/cfg to configure the yaml to be run.

#### Generate a launch file

You can write it manually or use the `create_launch.py` script. When you have many ROS nodes, it is recommended to use the script. The usage is as follows:

`python create_launch.py test envs/cfg/test.yaml`

The first parameter is the task name (also the generated launch file name), and the subsequent parameters are yaml files. If you want to run multiple yaml environments,

`python create_launch.py test A.yaml B.yaml C.yaml`

The generated launch file is located under `src/drl_nav/img_env/tmp`

#### run ROS

Open a new terminal

`roslaunch img_env test.launch`

### 💁 Explanation

#### Wrapper

To modify reward, action space and other operations, please follow the gym specification and perform wrapper encapsulation.

**!!!** NOTE: The order of filling in the wrapper in yaml is very important. The first one is encapsulated in the innermost layer and executed first, and the last one is encapsulated in the outermost layer and executed last. If you don’t understand here, imagine the pre-order and post-order of tree traversal.

#### Environment parameters

Currently, all parameter settings are in yaml, which is read by python and sent to C++.



### 💻 How to close the open while the GUI is running

Parameter Server

```
#Close
rosparam set test/show_gui false
#Open
rosparam set test/show_gui true
#Note that test here is changed to the node name you want to open, specifically the env_name+env_id set in yaml
#You can view all nodes through rosnode list
For example, if you want to close the GUI of the environment named image_ped_circle0, you can use the following command:

`rosparam set image_ped_circle0/show_gui true`
```



#### Pedestrian model

Pedestrian control methods are divided into pedsim, rvo, ervo, the codes are all open source, downloaded and placed on 3rd_party

rvo is orca control, ervo is emotional rvo, some modifications have been made based on rvo, see ervo code for details

Modify the pedestrian model in the yaml file, under `ped_sim:type`, you can choose `pedscene`, `rvoscene`, `ervoscene`

#### Supplement
For more details on using the simulator, please refer to [DRL-Navigation](https://github.com/DRL-Navigation/img_env)