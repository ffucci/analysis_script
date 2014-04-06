#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "Please put the list of the drivers to test in a file e pass as an argument"
    exit 0
fi

EXEC_TIME=10
#EXEC_TIME=$($EXEC_TIME*60)
FILE=$1 #name of the file
old_IFS=$IFS
IFS=$'\n'
drivers_list=($(cat $FILE)) # array
IFS=$old_IFS
BASE_PATH="$HOME/s2e-websvc/s2e-websvc/smoke-tests/winxp"
CUR_PATH=$(pwd)
echo ${drivers_list[@]}
pid_list=()

for driver in "${drivers_list[@]}"
do
	echo "Executing testsuite for driver:" ${driver}
	cd $BASE_PATH/$driver
	timeout $EXEC_TIME ./launch.sh > /dev/null 2>&1 &
	pid_list+=($!)
done 

echo "Waiting for processes: " ${pid_list[@]}
wait ${pid_list[@]}
cd $CUR_PATH

if [ ! -d "stats" ]; then
  # Control will enter here if $DIRECTORY doesn't exist. 
   echo "Directory stats doesn't exists creating it..."
   mkdir -p "stats"
fi

#Copying stuff
for driver in "${drivers_list[@]}"
do
	echo "Copying stats for driver:" ${driver}
	cp "$BASE_PATH/$driver/s2e-last/run.stats" "stats/$driver""_stat"
done 

echo "=========Displaying graphs======"
python ./display_stats.py
echo "Work done..."

