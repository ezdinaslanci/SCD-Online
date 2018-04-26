#include <stdio.h>
#include <string.h>
#include <netinet/in.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <chrono>
#include <armadillo>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <boost/filesystem.hpp>
#include "../include/rapidjson/document.h"
#include "../include/rapidjson/filereadstream.h"

// alphabet cardinality
#define alpha 4

// socket buffer size
#define BUFFER_SIZE 4096

// struct for storing joint probabilities
struct joint_probability {
	arma::Row <double> zero = arma::Row <double> (alpha, arma::fill::ones);
	arma::Mat <double> first = arma::Mat <double> (alpha, alpha, arma::fill::ones);
	arma::Cube <double> second = arma::Cube <double> (alpha, alpha, alpha, arma::fill::ones);
};

// struct for storing markov probabilities
struct markov_probability {
	arma::Row <double> zero = arma::Row <double> (alpha, arma::fill::ones);
	arma::Mat <double> first = arma::Mat <double> (alpha, alpha, arma::fill::ones);
	arma::Mat <double> second = arma::Mat <double> (alpha, alpha, arma::fill::ones);
};

// struct for storing dynamism scores
struct dynamism_score {
	arma::Row <double> zero = arma::Row <double> (alpha, arma::fill::zeros);
	arma::Mat <double> first = arma::Mat <double> (alpha, alpha, arma::fill::zeros);
	arma::Mat <double> second = arma::Mat <double> (alpha, alpha, arma::fill::zeros);
};

// struct for storing entropy values
struct entropy_value {
	double zero;
	arma::Row <double> first = arma::Row <double> (alpha, arma::fill::zeros);
	arma::Row <double> second = arma::Row <double> (alpha, arma::fill::zeros);
};

// markov chain parameters that will be stored for each regime
struct markov_chain_parameter_set {
	int model_id;
	joint_probability jp;
	arma::Row <double> parameter_set_vector = arma::Row <double> (alpha + std::pow(alpha, 2) + std::pow(alpha, 2), arma::fill::zeros);
};

// variables needed by socket programming
static int connection_result;
int port_no_start;

// variables needed by SCD
int first_token, second_token, queue_size, distance_from_last_peak, number_of_tokens_processed = 0;
bool terminate_SCD = false;
std::mutex sequence_mutex;
std::condition_variable sequence_ready;

// parameters from JSON
auto parameter_set = "";
int start_point, depth, look_buffer, number_of_dependencies_considered, moving_window_size, min_peak_distance;
double lambda_max, lambda_min, lambda_discount_rate, lambda_current, dynamism_discount, min_peak_height, entropy_threshold, model_error_threshold;
char markov_dependency_switch[3];
std::string dynamism_discount_type, acknowledgement_type, model_parameter_type;
bool log_input_sequence, log_joint_probabilities, log_markov_probabilities, log_dynamism_scores, log_AOLN, log_change_scores, log_change_points, log_entropy_values, log_regime_markov_models, log_regime_errors, log_debug;
bool send_acknowledgements;

// queues
std::deque <int> sequence = {};
std::deque <joint_probability> jp_queue = {};
std::deque <markov_probability> mp_queue = {};
std::deque <dynamism_score> dyn_queue = {};
std::deque <double> AOLN_queue = {};
std::deque <double> change_score_queue = {};
std::deque <int> change_point_queue = {};
std::deque <entropy_value> entropy_value_queue = {};
std::deque <markov_chain_parameter_set> markov_chain_parameter_set_queue = {};

// function prototypes
void read_settings ();
void display_settings ();
void run_SCD_server ();
double mean_absolute_error (arma::Row <double>, arma::Row <double>);
void SCD ();

// this function reads settings from a JSON file (placed in "include" folder)
void read_settings () {

	// reading & parsing JSON parameter file
	FILE* settings_file = fopen("../include/settings.json", "rb");
	char buffer[65536];
	rapidjson::FileReadStream is(settings_file, buffer, sizeof(buffer));
	rapidjson::Document settings_JSON;
	settings_JSON.ParseStream <0, rapidjson::UTF8 <>, rapidjson::FileReadStream> (is);

	// getting the parameter set to use
	assert(settings_JSON.HasMember("parameter_set_to_use"));
	assert(settings_JSON["parameter_set_to_use"].IsString());
	parameter_set = settings_JSON["parameter_set_to_use"].GetString();

	// checking if the JSON file has any missing parameters
	assert(settings_JSON.HasMember("parameters"));
	assert(settings_JSON["parameters"].HasMember(parameter_set));
	assert(settings_JSON["parameters"][parameter_set].HasMember("start_point"));
	assert(settings_JSON["parameters"][parameter_set].HasMember("lambda_max"));
	assert(settings_JSON["parameters"][parameter_set].HasMember("lambda_min"));
	assert(settings_JSON["parameters"][parameter_set].HasMember("lambda_discount_rate"));
	assert(settings_JSON["parameters"][parameter_set].HasMember("depth"));
	assert(settings_JSON["parameters"][parameter_set].HasMember("look_buffer"));
	assert(settings_JSON["parameters"][parameter_set].HasMember("dynamism_discount_type"));
	assert(settings_JSON["parameters"][parameter_set].HasMember("dynamism_discount"));
	assert(settings_JSON["parameters"][parameter_set].HasMember("markov_dependency_switch"));
	assert(settings_JSON["parameters"][parameter_set].HasMember("number_of_dependencies_considered"));
	assert(settings_JSON["parameters"][parameter_set].HasMember("moving_window_size"));
	assert(settings_JSON["parameters"][parameter_set].HasMember("min_peak_distance"));
	assert(settings_JSON["parameters"][parameter_set].HasMember("min_peak_height"));
	assert(settings_JSON["parameters"][parameter_set].HasMember("entropy_threshold"));
	assert(settings_JSON["parameters"][parameter_set].HasMember("model_error_threshold"));
	assert(settings_JSON["parameters"][parameter_set].HasMember("model_parameter_type"));
	assert(settings_JSON.HasMember("send_acknowledgements"));
	assert(settings_JSON.HasMember("acknowledgement_type"));
	assert(settings_JSON.HasMember("what_to_log"));
	assert(settings_JSON["what_to_log"].HasMember("input_sequence"));
	assert(settings_JSON["what_to_log"].HasMember("joint_probabilities"));
	assert(settings_JSON["what_to_log"].HasMember("markov_probabilities"));
	assert(settings_JSON["what_to_log"].HasMember("dynamism_scores"));
	assert(settings_JSON["what_to_log"].HasMember("AOLN"));
	assert(settings_JSON["what_to_log"].HasMember("change_scores"));
	assert(settings_JSON["what_to_log"].HasMember("change_points"));
	assert(settings_JSON["what_to_log"].HasMember("entropy_values"));
	assert(settings_JSON["what_to_log"].HasMember("regime_markov_models"));
	assert(settings_JSON["what_to_log"].HasMember("regime_errors"));
	assert(settings_JSON["what_to_log"].HasMember("debug_log"));

	// checking if the JSON file has any invalid types
	assert(settings_JSON["parameters"][parameter_set]["start_point"].IsInt());
	assert(settings_JSON["parameters"][parameter_set]["lambda_max"].IsDouble());
	assert(settings_JSON["parameters"][parameter_set]["lambda_min"].IsDouble());
	assert(settings_JSON["parameters"][parameter_set]["lambda_discount_rate"].IsDouble());
	assert(settings_JSON["parameters"][parameter_set]["depth"].IsInt());
	assert(settings_JSON["parameters"][parameter_set]["look_buffer"].IsInt());
	assert(settings_JSON["parameters"][parameter_set]["dynamism_discount_type"].IsString());
	assert(settings_JSON["parameters"][parameter_set]["dynamism_discount"].IsDouble());
	assert(settings_JSON["parameters"][parameter_set]["markov_dependency_switch"].IsString());
	assert(settings_JSON["parameters"][parameter_set]["number_of_dependencies_considered"].IsInt());
	assert(settings_JSON["parameters"][parameter_set]["moving_window_size"].IsInt());
	assert(settings_JSON["parameters"][parameter_set]["min_peak_distance"].IsInt());
	assert(settings_JSON["parameters"][parameter_set]["min_peak_height"].IsDouble());
	assert(settings_JSON["parameters"][parameter_set]["entropy_threshold"].IsDouble());
	assert(settings_JSON["parameters"][parameter_set]["model_error_threshold"].IsDouble());
	assert(settings_JSON["parameters"][parameter_set]["model_parameter_type"].IsString());
	assert(settings_JSON["send_acknowledgements"].IsBool());
	assert(settings_JSON["acknowledgement_type"].IsString());
	assert(settings_JSON["what_to_log"]["input_sequence"].IsBool());
	assert(settings_JSON["what_to_log"]["joint_probabilities"].IsBool());
	assert(settings_JSON["what_to_log"]["markov_probabilities"].IsBool());
	assert(settings_JSON["what_to_log"]["dynamism_scores"].IsBool());
	assert(settings_JSON["what_to_log"]["AOLN"].IsBool());
	assert(settings_JSON["what_to_log"]["change_scores"].IsBool());
	assert(settings_JSON["what_to_log"]["change_points"].IsBool());
	assert(settings_JSON["what_to_log"]["entropy_values"].IsBool());
	assert(settings_JSON["what_to_log"]["regime_markov_models"].IsBool());
	assert(settings_JSON["what_to_log"]["regime_errors"].IsBool());
	assert(settings_JSON["what_to_log"]["debug_log"].IsBool());

	// assigning values
	start_point = settings_JSON["parameters"][parameter_set]["start_point"].GetInt();
	lambda_max = settings_JSON["parameters"][parameter_set]["lambda_max"].GetDouble();
	lambda_min = settings_JSON["parameters"][parameter_set]["lambda_min"].GetDouble();
	lambda_discount_rate = settings_JSON["parameters"][parameter_set]["lambda_discount_rate"].GetDouble();
	depth = settings_JSON["parameters"][parameter_set]["depth"].GetInt();
	look_buffer = settings_JSON["parameters"][parameter_set]["look_buffer"].GetInt();
	dynamism_discount_type = settings_JSON["parameters"][parameter_set]["dynamism_discount_type"].GetString();
	dynamism_discount = settings_JSON["parameters"][parameter_set]["dynamism_discount"].GetDouble();
	stpcpy(markov_dependency_switch, settings_JSON["parameters"][parameter_set]["markov_dependency_switch"].GetString());
	number_of_dependencies_considered = settings_JSON["parameters"][parameter_set]["number_of_dependencies_considered"].GetInt();
	moving_window_size = settings_JSON["parameters"][parameter_set]["moving_window_size"].GetInt();
	min_peak_distance = settings_JSON["parameters"][parameter_set]["min_peak_distance"].GetInt();
	min_peak_height = settings_JSON["parameters"][parameter_set]["min_peak_height"].GetDouble();
	entropy_threshold = settings_JSON["parameters"][parameter_set]["entropy_threshold"].GetDouble();
	model_error_threshold = settings_JSON["parameters"][parameter_set]["model_error_threshold"].GetDouble();
	model_parameter_type = settings_JSON["parameters"][parameter_set]["model_parameter_type"].GetString();
	send_acknowledgements = settings_JSON["send_acknowledgements"].GetBool();
	acknowledgement_type = settings_JSON["acknowledgement_type"].GetString();
	log_input_sequence = settings_JSON["what_to_log"]["input_sequence"].GetBool();
	log_joint_probabilities = settings_JSON["what_to_log"]["joint_probabilities"].GetBool();
	log_markov_probabilities = settings_JSON["what_to_log"]["markov_probabilities"].GetBool();
	log_dynamism_scores = settings_JSON["what_to_log"]["dynamism_scores"].GetBool();
	log_AOLN = settings_JSON["what_to_log"]["AOLN"].GetBool();
	log_change_scores = settings_JSON["what_to_log"]["change_scores"].GetBool();
	log_change_points = settings_JSON["what_to_log"]["change_points"].GetBool();
	log_entropy_values = settings_JSON["what_to_log"]["entropy_values"].GetBool();
	log_regime_markov_models = settings_JSON["what_to_log"]["regime_markov_models"].GetBool();
	log_regime_errors = settings_JSON["what_to_log"]["regime_errors"].GetBool();
	log_debug = settings_JSON["what_to_log"]["debug_log"].GetBool();

	// check validity
	if (start_point < 0) {
		std::cerr << "Error: Parameter \"start_point\" must be non-negative." << std::endl;
		exit(1);
	}
	if (lambda_max < 0) {
		std::cerr << "Error: Parameter \"lambda_max\" must be non-negative." << std::endl;
		exit(1);
	}
	if (lambda_min < 0) {
		std::cerr << "Error: Parameter \"lambda_min\" must be non-negative." << std::endl;
		exit(1);
	}
	if (lambda_discount_rate < 0) {
		std::cerr << "Error: Parameter \"lambda_discount_rate\" must be non-negative." << std::endl;
		exit(1);
	}
	if (depth <= 0) {
		std::cerr << "Error: Parameter \"depth\" must be positive." << std::endl;
		exit(1);
	}
	if (look_buffer < 0) {
		std::cerr << "Error: Parameter \"look_buffer\" must be non-negative." << std::endl;
		exit(1);
	}
	if (dynamism_discount_type != "none" && dynamism_discount_type != "linear" && dynamism_discount_type != "exponential") {
		std::cerr << "Error: Parameter \"dynamism_discount_type\" must be either \"none\", \"linear\" or \"exponential\"." << std::endl;
		exit(1);
	}
	if (dynamism_discount_type == "linear" && dynamism_discount > std::max({depth, moving_window_size, look_buffer})) {
		std::cerr << "Error: Dynamism linear discount end point cannot be greater than the queue size." << std::endl;
		exit(1);
	}
	if ((markov_dependency_switch[0] != '0' && markov_dependency_switch[0] != '1') || (markov_dependency_switch[1] != '0' && markov_dependency_switch[1] != '1') || (markov_dependency_switch[2] != '0' && markov_dependency_switch[2] != '1')) {
		std::cerr << "Error: Invalid Markov dependency switch." << std::endl;
		exit(1);
	}
	if (number_of_dependencies_considered > (2 * alpha * alpha) + alpha) {
		std::cerr << "Error: Invalid number of dependecies to consider.." << std::endl;
		exit(1);
	}
	if (moving_window_size < 0) {
		std::cerr << "Error: Parameter \"moving_window_size\" must be non-negative." << std::endl;
		exit(1);
	}
	if (min_peak_distance < 0) {
		std::cerr << "Error: Parameter \"min_peak_distance\" must be non-negative." << std::endl;
		exit(1);
	}
	if (min_peak_height < 0) {
		std::cerr << "Error: Parameter \"min_peak_height\" must be non-negative." << std::endl;
		exit(1);
	}
	if (model_parameter_type != "mp" && model_parameter_type != "jp*mp") {
		std::cerr << "Error: Parameter \"model_parameter_type\" must be either \"mp\" or \"jp*mp\"." << std::endl;
		exit(1);
	}
	if (acknowledgement_type != "onyly_ack" && acknowledgement_type != "changes" && acknowledgement_type != "model_id") {
		std::cerr << "Error: Parameter \"acknowledgement_type\" must be either \"onyly_ack\", \"changes\" or \"model_id\"." << std::endl;
		exit(1);
	}
}

// this function displays settings read from a JSON file
void display_settings () {

	std::cout << "Settings: \n\t" <<
	"alpha: " << alpha << "\n\t" <<
	"send_acknowledgements: " << send_acknowledgements << "\n\t" <<
	"acknowledgement_type: " << acknowledgement_type << "\n\t" <<
	std::endl;

	std::cout << "Parameters: (" << parameter_set << ") \n\t" <<
	"start_point: " << start_point << "\n\t" <<
	"lambda_max: " << lambda_max << "\n\t" <<
	"lambda_min: " << lambda_min << "\n\t" <<
	"lambda_discount_rate: " << lambda_discount_rate << "\n\t" <<
	"depth: " << depth << "\n\t" <<
	"look_buffer: " << look_buffer << "\n\t" <<
	"dynamism_discount_type: " << dynamism_discount_type << "\n\t" <<
	"dynamism_discount: " << dynamism_discount << "\n\t" <<
	"markov_dependency_switch: " << markov_dependency_switch << "\n\t" <<
	"number_of_dependencies_considered: " << number_of_dependencies_considered << "\n\t" <<
	"moving_window_size: " << moving_window_size << "\n\t" <<
	"min_peak_distance: " << min_peak_distance << "\n\t" <<
	"min_peak_height: " << min_peak_height << "\n\t" <<
	"entropy_threshold: " << entropy_threshold << "\n\t" <<
	"model_error_threshold: " << model_error_threshold << "\n\t" <<
	std::endl;

	std::cout << "Logs: \n\t" <<
	"log_input_sequence: " << log_input_sequence << "\n\t" <<
	"log_joint_probabilities: " << log_joint_probabilities << "\n\t" <<
	"log_markov_probabilities: " << log_markov_probabilities << "\n\t" <<
	"log_dynamism_scores: " << log_dynamism_scores << "\n\t" <<
	"log_AOLN: " << log_AOLN << "\n\t" <<
	"log_change_scores: " << log_change_scores << "\n\t" <<
	"log_change_points: " << log_change_points << "\n\t" <<
	"log_entropy_values: " << log_entropy_values << "\n\t" <<
	"log_regime_markov_models: " << log_regime_markov_models << "\n\t" <<
	"log_regime_errors: " << log_regime_errors << "\n\t" <<
	"log_debug: " << log_debug << "\n\t" <<
	std::endl;

}

// this function starts the server
void run_SCD_server() {

	// socket variables
	int listen_result;
	struct sockaddr_in server_address, client_address;
	socklen_t len;

    // create socket
	listen_result = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (listen_result < 0) {
		std::cerr << "Cannot open socket" << std::endl;
		return;
	}
	bzero((char*) &server_address, sizeof(server_address));
	server_address.sin_family = AF_INET;
	server_address.sin_addr.s_addr = INADDR_ANY;
	server_address.sin_port = htons(port_no_start);

    // bind socket
	while (bind(listen_result, (struct sockaddr *)&server_address, sizeof(server_address)) < 0) {
		std::cerr << "Port " << port_no_start << " could not be bound, trying another." << std::endl;
		port_no_start += 1;
		server_address.sin_port = htons(port_no_start);
	}
	std::cout << "Port " << port_no_start << " is bound. " << std::flush;

	// wait for a client
	listen(listen_result, 1);
	len = sizeof(client_address);
	while (true) {
		std::cout << "Waiting for a connection... " << std::endl << std::flush;
		len = sizeof(client_address);
		connection_result = accept(listen_result, (struct sockaddr *)&client_address, &len);
		if (connection_result < 0) {
			std::cerr << "Cannot accept connection." << std::endl;
			return;
		}
		else {
			std::cout << "\n" << "########## Connection established. ##########" << std::endl;
			break;
		}
	}

	// std::cout << "SCD is now listening... \n" << std::endl;

	// start SCD thread
	std::thread SCD_tread(SCD);

	// token holders
	char token_array[BUFFER_SIZE], token_char, acknowledgement;
	int token, number_of_tokens = 0, read_result;

	// for each bulk of tokens read
	while (true) {

		// reset token array
		bzero(token_array, BUFFER_SIZE + 1);

		// read from socket
		read_result = read(connection_result, token_array, BUFFER_SIZE);
    	if (read_result < 0) {
			std::cerr << "ERROR reading from socket" << std::endl;
		}

		// for each token in the bulk
		for (size_t tokenWalker = 0; tokenWalker < read_result; tokenWalker++) {

			// get current token
			token_char = token_array[tokenWalker];

			// check termination signal
			if (token_char == 'X') {
				std::unique_lock <std::mutex> sequence_lock(sequence_mutex);
				sequence.push_front(first_token);
				sequence.push_front(second_token);
				sequence_lock.unlock();
				terminate_SCD = true;
				sequence_ready.notify_one();
				SCD_tread.join();
				break;
			}

			// convert current token from char to int
			token = token_char - '0';
			if (token > 9) {
				token -= 7;
			}

			// bound check
			if (token < 0 || token >= alpha) {
				std::cerr << "Error: Unexpected token received. (" << token << " > " << alpha << ")" << std::endl;
				exit(1);
			}

			// increment the counter
			number_of_tokens++;

			// save the first and second tokens
			if (number_of_tokens == 1) {
				first_token = token;
			}
			if (number_of_tokens == 2) {
				second_token = token;
			}

			// push the token to the sequence
			std::unique_lock <std::mutex> sequence_lock(sequence_mutex);
			sequence.push_front(token);
			sequence_lock.unlock();

			// notify the slwe estimator thread
			if (number_of_tokens > 2) {
				sequence_ready.notify_one();
			}
			else {
				if (send_acknowledgements) {
					if (acknowledgement_type == "only_ack" || acknowledgement_type == "changes") {
						acknowledgement = '0';
					}
					else if (acknowledgement_type == "model_id") {
						acknowledgement = '@';
					}
					else {
                        // TODO: add paranoia
					}
					write(connection_result, &acknowledgement, sizeof(acknowledgement));
				}
			}

		}

		// still handling termination
		if (terminate_SCD) {
			break;
		}

	}

    // close connection
	std::cout << "\n" << "Terminating SCD..." << std::endl;
	close(connection_result);

}

// error calculator
double calculate_error (std::string error_type, markov_chain_parameter_set reference, markov_chain_parameter_set target) {
	if (error_type == "ZWFSME"){ // Zero weighted first and second markov errors
		int start_index, final_index;
		int alphaSqr = alpha*alpha;
		arma::Row <double> diffZero = abs(reference.parameter_set_vector(arma::span(0, alpha - 1))-target.parameter_set_vector(arma::span(0, alpha - 1)));
		arma::Row <double> diffFirst = abs(reference.parameter_set_vector(arma::span(alpha, alphaSqr+alpha-1))-target.parameter_set_vector(arma::span(alpha, alphaSqr+alpha-1)));
		arma::Row <double> diffSecond = abs(reference.parameter_set_vector(arma::span(alphaSqr+alpha, 2*alphaSqr+alpha-1))-target.parameter_set_vector(arma::span(alphaSqr+alpha, 2*alphaSqr+alpha-1)));
		double firstOrderError = 0;
		double secondOrderError = 0;
		for (int i = 1; i <= alpha; i++){
			start_index = alpha*i - alpha;
			final_index = start_index + alpha - 1;
			firstOrderError += arma::sum(arma::dot(diffZero, diffFirst(arma::span(start_index, final_index))));
			secondOrderError += arma::sum(arma::dot(diffZero, diffSecond(arma::span(start_index, final_index))));
		}
		return firstOrderError + secondOrderError;
	}
	else if (error_type == "mae") {
		return arma::mean(arma::abs(target.parameter_set_vector - reference.parameter_set_vector));
	}
}

// this function clears the queues (buffers), intended to call after there is a change detected
void clear_queues () {
	jp_queue.clear();
	mp_queue.clear();
	dyn_queue.clear();
	AOLN_queue.clear();
	change_score_queue.clear();
}

// not for Scottish Country Dancing
void SCD () {

    // let the party begin
	joint_probability new_jp;
	markov_probability new_mp, current_mp, reference_mp;
	dynamism_score current_dynamism_score;
	entropy_value new_ev;
	markov_chain_parameter_set current_markov_chain_parameter_set, dummy_markov_chain_parameter_set;
	arma::Row <double> joint_probability_vector = arma::Row <double> (alpha + std::pow(alpha, 2) + std::pow(alpha, 3), arma::fill::zeros);
	arma::Row <double> markov_probability_vector = arma::Row <double> (alpha + std::pow(alpha, 2) + std::pow(alpha, 2), arma::fill::zeros);
	arma::Row <double> dyn_vector_ordered = arma::Row <double> (alpha + std::pow(alpha, 2) + std::pow(alpha, 2), arma::fill::zeros);
	arma::Row <double> entropy_value_vector = arma::Row <double> (1 + alpha + alpha, arma::fill::zeros);
	arma::Row <double> current_parameter_set_vector = arma::Row <double> (alpha + std::pow(alpha, 2) + std::pow(alpha, 2), arma::fill::zeros);
	bool empty_jp_queue = false, terminate_now = false;
	int current_queue_size, current_token, next_token, next_next_token, closest_markov_model_id, model_count = 0;
	size_t queue_walker;
	double dummy_sum, current_AOLN, discount_factor, current_error, min_error;
	char acknowledgement, previous_acknowledgement = '@';
	std::string experiment_name;

	// experiment date & time, result folder creation
	time_t now = time(0);
	tm *experiment_date = localtime(&now);
	experiment_name = std::to_string(1900 + experiment_date->tm_year) + "-" + std::to_string(1 + experiment_date->tm_mon) + "-" + std::to_string(experiment_date->tm_mday) + "_" + std::to_string(experiment_date->tm_hour) + "-" + std::to_string(experiment_date->tm_min) + "-" + std::to_string(experiment_date->tm_sec);
	std::cout << "\n" << "  Experiment Date: " << experiment_name << "\n" << std::endl;
	boost::filesystem::path current_path(boost::filesystem::current_path());
	boost::filesystem::path parent_path = current_path.parent_path();
	boost::filesystem::path results_path(parent_path.string() + "/out_files");
	boost::filesystem::path current_result_folder(results_path / experiment_name);
	create_directory(current_result_folder);

	// decide queue size
	queue_size = std::max({depth, moving_window_size, look_buffer});

	// initialize lambda
	lambda_current = lambda_max;

    // declare log files
	std::ofstream input_sequence_log_file;
	std::ofstream joint_probabilities_log_file;
	std::ofstream markov_probabilities_log_file;
	std::ofstream dynamism_scores_log_file;
	std::ofstream AOLNs_log_file;
	std::ofstream change_scores_log_file;
	std::ofstream change_points_log_file;
	std::ofstream entropy_values_log_file;
	std::ofstream markov_chain_parameters_log_file;
	std::ofstream debug_log_file;
	std::ofstream regime_errors_log_file;

	// clear log files
	if (log_input_sequence) {
		input_sequence_log_file.open("../out_files/" + experiment_name + "/input_sequence.txt", std::ofstream::out | std::ofstream::trunc);
	}
	if (log_joint_probabilities) {
		joint_probabilities_log_file.open("../out_files/" + experiment_name + "/joint_probabilities.txt", std::ofstream::out | std::ofstream::trunc);
	}
	if (log_markov_probabilities) {
		markov_probabilities_log_file.open("../out_files/" + experiment_name + "/markov_probabilities.txt", std::ofstream::out | std::ofstream::trunc);
	}
	if (log_dynamism_scores) {
		dynamism_scores_log_file.open("../out_files/" + experiment_name + "/dynamism_scores.txt", std::ofstream::out | std::ofstream::trunc);
	}
	if (log_AOLN) {
		AOLNs_log_file.open("../out_files/" + experiment_name + "/AOLNs.txt", std::ofstream::out | std::ofstream::trunc);
	}
	if (log_change_scores) {
		change_scores_log_file.open("../out_files/" + experiment_name + "/change_scores.txt", std::ofstream::out | std::ofstream::trunc);
	}
	if (log_change_points) {
		change_points_log_file.open("../out_files/" + experiment_name + "/change_points.txt", std::ofstream::out | std::ofstream::trunc);
	}
	if (log_entropy_values) {
		entropy_values_log_file.open("../out_files/" + experiment_name + "/entropy_values.txt", std::ofstream::out | std::ofstream::trunc);
	}
	if (log_regime_markov_models) {
		markov_chain_parameters_log_file.open("../out_files/" + experiment_name + "/markov_chain_parameters.txt", std::ofstream::out | std::ofstream::trunc);
	}
	if (log_debug) {
		debug_log_file.open("../out_files/" + experiment_name + "/debug_log.txt", std::ofstream::out | std::ofstream::trunc);
	}
	if (log_regime_errors) {
		regime_errors_log_file.open("../out_files/" + experiment_name + "/regime_errors.txt", std::ofstream::out | std::ofstream::trunc);
	}

	// tick for time measurement (begin)
	auto begin_scd = std::chrono::high_resolution_clock::now();

	// this loop iterates for each token
	while (true) {

        // wait for the signal & wake up if sequence has at least 2 tokens
		std::unique_lock <std::mutex> sequence_lock(sequence_mutex);
		sequence_ready.wait(sequence_lock, []{return sequence.size() > 2;});

        // get tokens (current, next & nextNext for zero-, first- and second-order Markov probabilities)
		current_token = sequence[sequence.size() - 1];
		next_token = sequence[sequence.size() - 2];
		next_next_token = sequence[sequence.size() - 3];

        // remove the last element in the buffer & unlock mutex
		sequence.pop_back();
		sequence_lock.unlock();

		// log the current token
		if (log_input_sequence) {
			input_sequence_log_file << current_token << "\n";
		}

        // initialize joint probabilities
		if (jp_queue.size() == 0) {
			new_jp.zero.fill(1 / std::pow(alpha, 1));
			new_jp.first.fill(1 / std::pow(alpha, 2));
			new_jp.second.fill(1 / std::pow(alpha, 3));
			jp_queue.push_front(new_jp);
			empty_jp_queue = true;
		}

        // apply penalty and rewarding to joint probabilities #############################################################################
		new_jp.zero = jp_queue.front().zero * lambda_current;
		new_jp.zero(current_token) += 1 - lambda_current;
		new_jp.first = jp_queue.front().first * lambda_current;
		new_jp.first(current_token, next_token) += 1 - lambda_current;
		new_jp.second = jp_queue.front().second * lambda_current;
		new_jp.second(current_token, next_token, next_next_token) += 1 - lambda_current;
		if (lambda_current > lambda_min) {
			lambda_current *= lambda_discount_rate;
		}
		jp_queue.push_front(new_jp);
		if (empty_jp_queue || jp_queue.size() > queue_size) {
			jp_queue.pop_back();
			empty_jp_queue = false;
		}
		current_queue_size = jp_queue.size();
		for (size_t i = 0; i < alpha; i++) {
			queue_walker = 0;
			dummy_sum = 0;
			while (queue_walker < depth && queue_walker < current_queue_size) {
				dummy_sum += jp_queue[queue_walker].zero(i);
				queue_walker++;
			}
			jp_queue.front().zero(i) = dummy_sum / queue_walker;
			for (size_t j = 0; j < alpha; j++) {
				queue_walker = 0;
				dummy_sum = 0;
				while (queue_walker < depth && queue_walker < current_queue_size) {
					dummy_sum += jp_queue[queue_walker].first(i, j);
					queue_walker++;
				}
				jp_queue.front().first(i, j) = dummy_sum / queue_walker;
				for (size_t k = 0; k < alpha; k++) {
					queue_walker = 0;
					dummy_sum = 0;
					while (queue_walker < depth && queue_walker < current_queue_size) {
						dummy_sum += jp_queue[queue_walker].second(i, j, k);
						queue_walker++;
					}
					jp_queue.front().second(i, j, k) = dummy_sum / queue_walker;
				}
			}
		}
		if (log_joint_probabilities) {
			joint_probability_vector(arma::span(0, alpha - 1)) = jp_queue.front().zero;
			joint_probability_vector(arma::span(alpha, alpha + std::pow(alpha, 2) - 1)) = vectorise(jp_queue.front().first).t();
			joint_probability_vector(arma::span(alpha + std::pow(alpha, 2), alpha + std::pow(alpha, 2) + std::pow(alpha, 3) - 1)) = vectorise(jp_queue.front().second).t();
			joint_probabilities_log_file << joint_probability_vector;
		}
        // ################################################################################################################################

        // calculate (and log) markov probabilities & push ################################################################################
		new_mp.zero = new_jp.zero;
		for (size_t i = 0; i < alpha; i++) {
			for (size_t j = 0; j < alpha; j++ ) {
				new_mp.first(i, j) = new_jp.first(j, i) / new_jp.zero(j);
				new_mp.second(i, j) = arma::accu(new_jp.second(arma::span(j, j), arma::span(), arma::span(i, i))) / arma::accu(new_jp.first(j, arma::span()));
			}
		}
		mp_queue.push_front(new_mp);
		if (mp_queue.size() > queue_size) {
			mp_queue.pop_back();
		}
		if (log_markov_probabilities) {
			markov_probability_vector(arma::span(0, alpha - 1)) = mp_queue.front().zero;
			markov_probability_vector(arma::span(alpha, alpha + std::pow(alpha, 2) - 1)) = vectorise(mp_queue.front().first).t();
			markov_probability_vector(arma::span(alpha + std::pow(alpha, 2), alpha + std::pow(alpha, 2) + std::pow(alpha, 2) - 1)) = vectorise(mp_queue.front().second).t();
			markov_probabilities_log_file << markov_probability_vector;
		}
		// ################################################################################################################################

        // calculate error for each model in the memory ###################################################################################
		if (model_parameter_type == "mp") {
			current_parameter_set_vector(arma::span(0, alpha - 1)) = mp_queue.front().zero;
			current_parameter_set_vector(arma::span(alpha, alpha + std::pow(alpha, 2) - 1)) = vectorise(mp_queue.front().first).t();
			current_parameter_set_vector(arma::span(alpha + std::pow(alpha, 2), alpha + std::pow(alpha, 2) + std::pow(alpha, 2) - 1)) = vectorise(mp_queue.front().second).t();
		}
		else {
            // tba
		}
		current_markov_chain_parameter_set.parameter_set_vector(arma::span::all) = current_parameter_set_vector;
		current_markov_chain_parameter_set.jp = jp_queue.front();
		min_error = 1000;
        // TODO: maybe inf?
		if (markov_chain_parameter_set_queue.size() > 0) {
			min_error = calculate_error("ZWFSME", markov_chain_parameter_set_queue[0], current_markov_chain_parameter_set);
			if (log_regime_errors) {
				regime_errors_log_file << std::fixed << std::setprecision(5) << min_error;
			}
			closest_markov_model_id = 0;
			for (queue_walker = 1; queue_walker < markov_chain_parameter_set_queue.size(); queue_walker++) {
				current_error = calculate_error("ZWFSME", markov_chain_parameter_set_queue[queue_walker], current_markov_chain_parameter_set);
				if (log_regime_errors) {
					regime_errors_log_file << "\t" << std::fixed << std::setprecision(5) << current_error;
				}
				if (current_error < min_error) {
					min_error = current_error;
					closest_markov_model_id = queue_walker;
				}
			}
			if (log_regime_errors) {
				regime_errors_log_file << "\n";
			}
		}
        // ################################################################################################################################

		// calculate (and log) entropy values #############################################################################################
		new_ev.zero = -arma::accu(mp_queue.front().zero % log2(mp_queue.front().zero));
		for (size_t column_walker = 0; column_walker < alpha; column_walker++) {
			new_ev.first(column_walker) = -arma::accu(mp_queue.front().first(arma::span(), column_walker) % log2(mp_queue.front().first(arma::span(), column_walker)));
			new_ev.second(column_walker) = -arma::accu(mp_queue.front().second(arma::span(), column_walker) % log2(mp_queue.front().second(arma::span(), column_walker)));
		}
		entropy_value_queue.push_front(new_ev);
		if (entropy_value_queue.size() > queue_size) {
			entropy_value_queue.pop_back();
		}
		entropy_value_vector(0) = entropy_value_queue.front().zero;
		entropy_value_vector(arma::span(1, alpha)) = entropy_value_queue.front().first;
		entropy_value_vector(arma::span(alpha + 1, 2 * alpha)) = entropy_value_queue.front().second;
		if (log_entropy_values) {
			entropy_values_log_file << entropy_value_vector;
		}
        // ################################################################################################################################

		// dynamism amplification #########################################################################################################
		queue_walker = 1;
		current_mp = mp_queue[0];
		current_dynamism_score.zero.zeros();
		current_dynamism_score.first.zeros();
		current_dynamism_score.second.zeros();
		while (queue_walker < look_buffer && queue_walker < mp_queue.size()) {
			if (dynamism_discount_type == "none") {
				discount_factor = 1;
			}
			else if (dynamism_discount_type == "linear") {
				discount_factor = (double) (dynamism_discount - queue_walker) / dynamism_discount;
			}
			else if (dynamism_discount_type == "exponential") {
				discount_factor = dynamism_discount;
			}
			if (discount_factor < 0) {
				discount_factor = 0;
			}
			reference_mp = mp_queue[queue_walker];
			current_dynamism_score.zero += abs(current_mp.zero - reference_mp.zero) * discount_factor;
			current_dynamism_score.first += abs(current_mp.first - reference_mp.first) * discount_factor;
			current_dynamism_score.second += abs(current_mp.second - reference_mp.second) * discount_factor;
			queue_walker++;
		}
		current_dynamism_score.zero /= queue_walker;
		current_dynamism_score.first /= queue_walker;
		current_dynamism_score.second /= queue_walker;
		dyn_queue.push_front(current_dynamism_score);
		if (dyn_queue.size() > queue_size) {
			dyn_queue.pop_back();
		}
        // ################################################################################################################################

		// average of largest N ###########################################################################################################
		if (markov_dependency_switch[0] == '1') {
			dyn_vector_ordered(arma::span(0, alpha - 1)) = dyn_queue.front().zero;
		}
		if (markov_dependency_switch[1] == '1') {
			dyn_vector_ordered(arma::span(alpha, alpha + std::pow(alpha, 2) - 1)) = vectorise(dyn_queue.front().first).t();
		}
		if (markov_dependency_switch[2] == '1') {
			dyn_vector_ordered(arma::span(alpha + std::pow(alpha, 2), alpha + std::pow(alpha, 2) + std::pow(alpha, 2) - 1)) = vectorise(dyn_queue.front().second).t();
		}
		if (log_dynamism_scores) {
			dynamism_scores_log_file << dyn_vector_ordered;
		}
		dyn_vector_ordered = sort(dyn_vector_ordered, "descend");
		current_AOLN = mean(dyn_vector_ordered.head(number_of_dependencies_considered));
		AOLN_queue.push_front(current_AOLN);
		if (AOLN_queue.size() > queue_size) {
			AOLN_queue.pop_back();
		}
		if (AOLNs_log_file) {
			AOLNs_log_file << current_AOLN << "\n";
		}
        // ################################################################################################################################

		// moving average #################################################################################################################
		queue_walker = 0;
		dummy_sum = 0;
		while (queue_walker < moving_window_size && queue_walker < AOLN_queue.size()) {
			dummy_sum += AOLN_queue[queue_walker];
			queue_walker++;
		}
		change_score_queue.push_front(dummy_sum / queue_walker);
		if (change_score_queue.size() > queue_size) {
			change_score_queue.pop_back();
		}
		if (log_change_scores) {
			change_scores_log_file << (dummy_sum / queue_walker) << "\n";
		}
		// ################################################################################################################################

		// peak detection #################################################################################################################
		number_of_tokens_processed++;
		distance_from_last_peak = number_of_tokens_processed - change_point_queue.front();
		if (change_score_queue.front() > min_peak_height && distance_from_last_peak > min_peak_distance && number_of_tokens_processed > start_point) {
			change_point_queue.push_front(number_of_tokens_processed);
			lambda_current = lambda_max;
			std::cout << "  >> change detected at " << number_of_tokens_processed << std::endl;
			if (log_change_points) {
				change_points_log_file << number_of_tokens_processed << "\n";
			}
			if (log_regime_markov_models) {
				markov_chain_parameters_log_file << current_parameter_set_vector;
			}
		}
        // ################################################################################################################################

        // do these before termination
		if (terminate_now && sequence.size() == 2) {

			// log the final markov model
			current_markov_chain_parameter_set.model_id = model_count++;
			current_markov_chain_parameter_set.parameter_set_vector(arma::span::all) = current_parameter_set_vector;
			current_markov_chain_parameter_set.jp = jp_queue.front();
			markov_chain_parameter_set_queue.push_back(current_markov_chain_parameter_set);
			if (log_regime_markov_models) {
				markov_chain_parameters_log_file << current_parameter_set_vector;
			}

			// console message
			std::cout << "\n" << "########## All tokens are processed. ##########" << std::endl;

			// log results to console
			auto end_scd = std::chrono::high_resolution_clock::now();
    		auto duration = end_scd - begin_scd;
    		auto duration_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
			double duretion_seconds = (double) duration_milliseconds / 1000;
			std::cout << "\n" << "  Total number of tokens processed: " << number_of_tokens_processed << std::endl;
			std::cout << "  Total duration: " << duretion_seconds << " seconds" << std::endl;
			std::cout << "  Throughput: " << number_of_tokens_processed / duretion_seconds << " tokens/second" << std::endl;
			std::cout << "\n" << "  Total number of change points detected: " << change_point_queue.size() << std::endl;

			// close file connections
			if (log_input_sequence) {
				input_sequence_log_file.close();
			}
			if (log_joint_probabilities) {
				joint_probabilities_log_file.close();
			}
			if (log_markov_probabilities) {
				markov_probabilities_log_file.close();
			}
			if (log_dynamism_scores) {
				dynamism_scores_log_file.close();
			}
			if (log_AOLN) {
				AOLNs_log_file.close();
			}
			if (log_change_scores) {
				change_scores_log_file.close();
			}
			if (log_change_points) {
				change_points_log_file.close();
			}
			if (log_entropy_values) {
				entropy_values_log_file.close();
			}
			if (log_regime_markov_models) {
				markov_chain_parameters_log_file.close();
			}
			if (log_regime_errors) {
				regime_errors_log_file.close();
			}
			if (log_debug) {
				debug_log_file.close();
			}

			// bye
			return;

		}

		// send acknowledgements
		if (send_acknowledgements && !terminate_SCD && sequence.size() >= 2) {
			if (acknowledgement_type == "only_ack") {
				acknowledgement = '0';
			}
			else if (acknowledgement_type == "changes") {
				if (change_point_queue.front() == number_of_tokens_processed) {
					acknowledgement = '1';
				}
				else {
					acknowledgement = '0';
				}
			}
			else if (acknowledgement_type == "model_id") {

                // change score is high
				if (change_score_queue.front() > min_peak_height && distance_from_last_peak > min_peak_distance) {

                    // from unsaved model to unsaved model (@, ?, @)
					if (previous_acknowledgement == '@') {
						acknowledgement = '?';
						current_markov_chain_parameter_set.model_id = model_count++;
						current_markov_chain_parameter_set.parameter_set_vector(arma::span::all) = current_parameter_set_vector;
						current_markov_chain_parameter_set.jp = jp_queue.front();
						markov_chain_parameter_set_queue.push_back(current_markov_chain_parameter_set);
						clear_queues();
						dummy_markov_chain_parameter_set = {};
					}

                    // from a saved model to unsaved model (A, ?, @)
					else {
						acknowledgement = '>';
						markov_chain_parameter_set_queue[previous_acknowledgement - 'A'].parameter_set_vector(arma::span::all) = current_parameter_set_vector;
						markov_chain_parameter_set_queue[previous_acknowledgement - 'A'].jp = jp_queue.front();
						clear_queues();
						dummy_markov_chain_parameter_set = {};
					}
				}

                // change score is not high
				else {

                    // min error is high
					if (min_error > model_error_threshold) {

                        // from unsaved model to unsaved model (@, @)
						if (previous_acknowledgement == '@' || previous_acknowledgement == '?' || previous_acknowledgement == '>') {
							acknowledgement = '@';
						}

                        // from a saved model to unsaved model (A, @)
						else {
							acknowledgement = '@';
							clear_queues();
							jp_queue.push_front(dummy_markov_chain_parameter_set.jp);
						}
					}

                    // min error is not high
					else {

                        // from unsaved model to a saved model (@, A)
						if (previous_acknowledgement == '@') {
							acknowledgement = closest_markov_model_id + 'A';
							current_markov_chain_parameter_set.parameter_set_vector(arma::span::all) = current_parameter_set_vector;
							current_markov_chain_parameter_set.jp = jp_queue.front();
							dummy_markov_chain_parameter_set = current_markov_chain_parameter_set;
							// clear_queues();
							// jp_queue.push_front(markov_chain_parameter_set_queue[closest_markov_model_id].jp);
							// empty_jp_queue = true;
						}

                        // from a saved model to a saved model (A, X)
						else {

                            // from a saved model to the same saved model (A, A)
							if (previous_acknowledgement == closest_markov_model_id + 'A') {
								acknowledgement = closest_markov_model_id + 'A';
							}

                            // from a saved model to another saved model (A, B)
							else {
								acknowledgement = closest_markov_model_id + 'A';
								// clear_queues();
								// jp_queue.push_front(markov_chain_parameter_set_queue[closest_markov_model_id].jp);
								// empty_jp_queue = true;
							}

						}

					}

				}

			}
			if (log_debug) {
				debug_log_file << std::fixed << std::setprecision(8) << "#models: " << markov_chain_parameter_set_queue.size() << " \t lambda: " << lambda_current << " \t merr: " << min_error << " \t ent: " << arma::mean(entropy_value_vector) << " \t ack: " << acknowledgement << std::endl;
			}
			write(connection_result, &acknowledgement, sizeof(acknowledgement));
			previous_acknowledgement = acknowledgement;
		}

		// terminate in the next iteration
		if (terminate_SCD) {
			terminate_now = true;
		}

	}

}

// because we always need a main function
int main (int argc, char* argv[]) {

	// argument check (if no port number is given, default is 20000)
	if (argc == 1) {
		port_no_start = 20000;
	}
	else if (argc == 2) {
		port_no_start = atoi(argv[1]);
	}
	else {
		std::cerr << "Syntax: ./scd <port (OPTIONAL)>" << std::endl;
		return 0;
	}

	// read settings (mostly parameters)
	read_settings();

	// show settings
	display_settings();

	// run the change detection server
	run_SCD_server();

	// terminate
	return 0;

}
