#!/bin/bash

build=false
portNo=20000
while getopts "p:b" opt; do
	case $opt in
		p)
			portNo=${OPTARG}
		;;
		b)
			build=true
		;;
		\?)
			echo "Invalid option: -$OPTARG"
			exit 1
		;;
	esac
done

clear
if [ "$build" = true ] ; then
    echo -e "Building SCD... \n"
	g++ -std=c++11 source/scd.cpp -o bin/scd -lpthread -larmadillo -lboost_system -lboost_filesystem
fi

cd bin && ./scd $portNo
