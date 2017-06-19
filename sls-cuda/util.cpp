#include "defs.h"

#include <boost/config.hpp>
#include <boost/program_options/detail/config_file.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

program_options_t get_program_options(int argc, char *argv[])
{
    program_options_t options;

    // default values
    options.input = "";
    options.n_threads = 256;
    options.n_iterations = 256;
    options.flip_probability = 0.25;

    po::options_description description("runsat: ./runsat --input <file> [options]");

    description.add_options()
        ("help,h", "displays this help message")
        ("version,v", "displays version number")
        ("input,i", po::value<std::string>(), "input CNF file")
        ("iterations,j", po::value<int>(), "sets number of iterations (default=threads*256)")
        ("threads,t", po::value<int>(), "sets number of threads (default=256)")
        ("flip,f", po::value<double>(), "sets frequency of random flips (default=0.25)");

    po::positional_options_description pos;
    pos.add("input", -1);

    po::variables_map vars;

    try {
        po::store(po::command_line_parser(argc, argv).options(description).positional(pos).run(), vars);
        po::notify(vars);
    }
    catch (const po::error& e) {
        std::cerr << e.what() << std::endl;
        exit(0);
    }

    if (vars.count("help") || vars.size() == 0) {
        std::cout << description;
        exit(0);
    }

    if (vars.count("version")) {
        std::cout << "v0.01\n";
        exit(0);
    }

    if (vars.count("input")) {
        options.input = vars["input"].as<std::string>();
    }

    if (vars.count("threads")) {
        options.n_threads = vars["threads"].as<int>();
    }

    if (vars.count("iterations")) {
        options.n_iterations = vars["iterations"].as<int>();
    }

    if (vars.count("flip")) {
        options.flip_probability = vars["flip"].as<double>();
    }

    if (options.input.size() == 0) {
        std::cout << description;
    }

    printf("v listing parameters ...\n");
    printf("v    input file:\t\t\t%s\n", options.input.c_str());
    printf("v    number of threads:\t\t\t%i\n", options.n_threads);
    printf("v    number of iterations:\t\t%i\n", options.n_iterations);
    printf("v    frequency of random flips:\t\t%f\n", options.flip_probability);

    return options;
}

std::tuple<variable_map, clause_vector>
read_cnf(std::string const& filename)
{
    bool redundant_literal = false;

    std::ifstream file(filename);
    if (!file) {
        fprintf(stderr, "ifstream::open() failed\n");
        exit(EXIT_FAILURE);
    }

    int line_counter = 0;

    variable_map variables;
    clause_vector clauses;

    while (!file.eof()) {
        line_counter++;

        std::string line;
        std::getline(file, line);

        if (line.empty()) {
            continue;
        }

        /* Skip problem / comment */
        if (line[0] == 'p' || line[0] == 'c') {
            continue;
        }

        /* XOR clauses */
        if (line[0] == 'x') {
            fprintf(stderr, "Cannot read XOR clauses\n");
            exit(EXIT_FAILURE);
        }

        clause_t clause;
        std::stringstream ss(line);
        std::map<int, size_t> repeats;

        while (!ss.eof()) {
            literal_t literal;
            ss >> literal;

            if (literal == 0) {
                break;
            }

            if (repeats.count(literal) == 0) {
                variable_t var = std::abs(literal);

                // remap of variables to the range [1, n],
                // where n is the total number of variables
                variable_t remapped_var;
                auto it = variables.find(var);

                if (it == variables.end()) {
                    remapped_var = variables.size() + 1;
                    variables[var] = remapped_var;
                }
                else {
                    remapped_var = it->second;
                }

                clause.push_back(remapped_var * (literal < 0 ? -1 : 1));
            }
            else {
                if (!redundant_literal) {
                    printf("v parsing information ...\n");
                    redundant_literal = true;
                }

                printf("v literal %i occuring multiple times "
                    "in clause in line %i\n",
                    literal, line_counter);
            }

            repeats[literal]++;
        }

        clauses.push_back(clause);
    }

    if (redundant_literal) {
        printf("v ... done\n");
    }

    return std::make_tuple(variables, clauses);
}
