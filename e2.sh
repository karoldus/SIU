!#/usr/bin/bash
cd /root/SIU_E2

roslaunch turtlesim siu.launch &
PID1=$!

(
sleep 2
python symulator.py
) &
PID2=$!

wait $PID1 $PID2
