#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "Please put the drivers list to test in a file and pass as an argument"
    exit 0
fi

EXEC_TIME=$2
EXEC_TIME=$((EXEC_TIME*60))
FILE=$1 #name of the file
old_IFS=$IFS
IFS=$'\n'
drivers_list=($(cat $FILE)) # array
IFS=$old_IFS
BASE_PATH="$HOME/s2e-websvc/s2e-websvc/smoke-tests/winxp"
VANILLA_PATH="$HOME/current_websvc/s2e-websvc/smoke-tests/winxp"
CUR_PATH=$(pwd)
echo ${drivers_list[@]}
pid_list=()

echo "Preparing the execution..."
if [ ! -d "stats" ]; then
  # Control will enter here if $DIRECTORY doesn't exist. 
  echo "Directory stats doesn't exists creating it..."
  mkdir -p "stats"
fi

#rm -f stats/*
echo "Removing old results directory..."
rm -rf results

for driver in "${drivers_list[@]}"
do
	echo "Executing testsuite for driver:" ${driver}
	echo "===============S2E TCI Interpreter=================="
	echo
	cd $BASE_PATH/$driver
	timeout $EXEC_TIME ./launch.sh > /dev/null 2>&1 & 
	pid_list+=($!)
	echo "=============== VANILLA S2E ========================"
	echo
	cd $VANILLA_PATH/$driver
        timeout $EXEC_TIME ./launch.sh > /dev/null 2>&1 & 
        pid_list+=($!)
done 

echo "Waiting for processes: " ${pid_list[@]}
wait ${pid_list[@]}
cd $CUR_PATH

#Copying stuff
for driver in "${drivers_list[@]}"
do
	echo "Copying stats for driver:" ${driver}
	echo "Copying stats for TCI interpreter"
	cp "$BASE_PATH/$driver/s2e-last/run.stats" "stats/$driver""_tci_stat"
	echo "Copying stats for KLEE interpreter"
	cp "$VANILLA_PATH/$driver/s2e-last/run.stats" "stats/$driver""_klee_stat"
done 

echo "=========Displaying graphs======"
python ./display_stats.py
echo "Work done..."

