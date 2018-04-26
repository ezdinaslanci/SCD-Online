#include <armadillo>
#include <string.h>
#include <cstring>
#include <unistd.h>
#include <stdio.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <strings.h>
#include <stdlib.h>
#include <string>
#include <vector>

int main (int argc, char* argv[]) {

	int listen_result, port_no, number_of_tokens;
	struct sockaddr_in server_address;
	struct hostent *server;
	char token;
	std::string input_file_name;
	arma::ivec sequence;

	std::cout << "Loading data sequence..." << std::endl;
	// input_file_name = "17-09-17-13-44-24-280-a4-m5-v0.25-n0.dat";
	input_file_name = "input_sequence.txt";
	// input_file_name = std::string("cropped_raw_data_1000_1002_k15") + std::string(".dat");
	sequence.load("../input_sequences/" + input_file_name);

	if (argc < 4) {
		std::cerr << "Syntax : ./client <host name> <port>" << std::endl;
		return 0;
	}

	port_no = atoi(argv[2]);

	if ((port_no > 65535) || (port_no < 2000)) {
		std::cerr << "Please enter port number between 2000 - 65535" << std::endl;
		return 0;
	}

	number_of_tokens = atoi(argv[3]);
	if (number_of_tokens < 0) {
		std::cerr << "Invalid number of tokens!" << std::endl;
		return 0;
	}
	if (number_of_tokens == 0) {
		number_of_tokens = sequence.size();
	}

    // create client skt
	listen_result = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);

	if (listen_result < 0) {
		std::cerr << "Cannot open socket" << std::endl;
		return 0;
	}

	server = gethostbyname(argv[1]);

	if (server == NULL) {
		std::cerr << "Host does not exist" << std::endl;
		return 0;
	}

	bzero((char *) &server_address, sizeof(server_address));
	server_address.sin_family = AF_INET;

	bcopy((char *) server -> h_addr, (char *) &server_address.sin_addr.s_addr, server -> h_length);

	server_address.sin_port = htons(port_no);

	int checker = connect(listen_result,(struct sockaddr *) &server_address, sizeof(server_address));

	if (checker < 0) {
		std::cerr << "Cannot connect!" << std::endl;
		return 0;
	}
	else {
		std::cout << "Connection successful." << std::endl;
	}

	std::cout << "Bursting data..." << std::endl;

	char token_array[1024], token_char, acknowledgement;
	int read_result;

    // send stuff to server
	for (size_t sequence_walker = 0; sequence_walker < number_of_tokens; sequence_walker++) {
		token = sequence[sequence_walker % sequence.size()] + '0';
		if (token > '9') {
			token += 7;
		}
		write(listen_result, &token, sizeof(token));
		read_result = read(listen_result, &token_char, 1);
		if (read_result < 0) {
			std::cerr << "ERROR reading from socket" << std::endl;
		}
		std::cout << "token sent: " << token << " | " << sequence_walker + 1 << " / " << sequence.size() << " | ack: " << token_char << std::endl;
	}

	token = 'X';
	write(listen_result, &token, 1);
	std::cout << "token sent: " << token << std::endl;

	std::cout << "\n" << "All tokens in file " << input_file_name << " are sent." << std::endl;

}
