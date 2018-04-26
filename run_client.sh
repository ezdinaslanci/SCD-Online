#!/bin/bash

build=false
hostName="localhost"
portNo=20000
numberOfTokens=0
while getopts "h:p:n:b" opt; do
	case $opt in
		h)
			hostName=${OPTARG}
		;;
		p)
			portNo=${OPTARG}
		;;
		n)
			numberOfTokens=${OPTARG}
		;;
		b)
			build=true
		;;
		*)
			# echo "Invalid option: -$OPTARG"
			exit 1
		;;
	esac
done

clear
if [ "$build" = true ] ; then
	echo "Building Auto Client..."
	g++ source/auto_client.cpp -o bin/auto_client
fi

clear
cd bin && ./auto_client $hostName $portNo $numberOfTokens
