# README #

This README documents whatever steps are necessary to get the application up and running.

### What is this repository for? ###

This repository includes an online implementation of SCD (SLWE Change Detection).
TODO: add paper link here

### How do I get set up? ###

Dependencies: 

- Armadillo (http://arma.sourceforge.net/)

- Boost Filesystem (http://www.boost.org/doc/libs/1_65_1/libs/filesystem/doc/index.htm)

to run SCD:

	./runSCD.sh

	arguments:
	-b: builds before running
	-p [PORTNO]: port number to listen, if not specified 20000 is used

	notes:
	if specified or default port number cannot be bound, incremental trial is applied.

-----

to run auto client: (for testing)

	./runClient.sh

	arguments:
	-b:	builds before running
	-h [HOST]: host to connect, if not specified "localhost" is used
	-p [PORT]: port number to connect, if not specified 20000 is used

	notes:
	currently the input file is hard-coded. check source/autoClient.cpp & look for inputFileName

-----

The result of dynamism amplification process is subjected to discounting in order to raise the influence of the most recent token. This is applied directly on the dynamism scores. The discount function can be modified by changing the "dynamismDiscountType" and "dynamismDiscount" parameters in settings.json.

dynamismDiscountType = {"none", "linear", "exponential"}

dynamismDiscount = {DONT_CARE, endPoint, factor}
