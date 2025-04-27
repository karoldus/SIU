# SIU
# How to run it?
```bash
git clone https://github.com/karoldus/SIU.git

docker run --name siu \
  -p 6080:80 \
  -e RESOLUTION=1920x1080 \
  -v <path_to_repo>/SIU:/root/siu_ws/src/SIU \
  dudekw/siu-20.04
```

# How to change map?
In docker container:
```bash
cp /root/siu_ws/src/SIU/roads.png /root/siu_ws/src/ros_tutorials/turtlesim/images/roads.png
```

Check the result:
```bash
source /root/siu_ws/devel/setup.bash
roslaunch turtlesim siu.launch
```