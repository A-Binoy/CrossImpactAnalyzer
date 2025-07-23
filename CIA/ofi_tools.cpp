#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
namespace py = pybind11;

struct LOBSTERRow {
    double time;
    int type;
    double price;
    int direction;
    std::vector<double> bids;
    std::vector<double> asks;
    std::vector<int> bid_sizes;
    std::vector<int> ask_sizes;
};

std::vector<LOBSTERRow> read_lobster_files(const std::string &message_file, const std::string &orderbook_file, int levels = 10) {
    std::ifstream msg(message_file), ob(orderbook_file);
    std::vector<LOBSTERRow> data;
    std::string msg_line, ob_line;

    while (std::getline(msg, msg_line) && std::getline(ob, ob_line)) {
        std::stringstream msg_ss(msg_line), ob_ss(ob_line);
        std::string val;
        std::vector<std::string> msg_tokens, ob_tokens;

        while (std::getline(msg_ss, val, ',')) msg_tokens.push_back(val);
        while (std::getline(ob_ss, val, ',')) ob_tokens.push_back(val);

        LOBSTERRow row;
        row.time = std::stod(msg_tokens[0]);
        row.type = std::stoi(msg_tokens[1]);
        row.price = std::stod(msg_tokens[4]) / 10000.0;
        row.direction = std::stoi(msg_tokens[5]);

        for (int i = 0; i < levels; ++i) {
            row.asks.push_back(std::stod(ob_tokens[i * 4 + 0]));
            row.ask_sizes.push_back(std::stoi(ob_tokens[i * 4 + 1]));
            row.bids.push_back(std::stod(ob_tokens[i * 4 + 2]));
            row.bid_sizes.push_back(std::stoi(ob_tokens[i * 4 + 3]));
        }
        data.push_back(row);
    }
    return data;
}

std::vector<std::vector<double>> compute_ofi(const std::vector<LOBSTERRow> &data) {
    std::vector<std::vector<double>> ofi_list;
    ofi_list.reserve(data.size() - 1);

    for (size_t i = 1; i < data.size(); ++i) {
        std::vector<double> ofi_per_level;
        for (size_t level = 0; level < data[i].bids.size(); ++level) {
            int delta_bid = data[i].bid_sizes[level] - data[i - 1].bid_sizes[level];
            int delta_ask = data[i - 1].ask_sizes[level] - data[i].ask_sizes[level];
            double ofi = delta_bid + delta_ask;
            ofi_per_level.push_back(ofi);
        }
        ofi_list.push_back(ofi_per_level);
    }
    return ofi_list;
}
std::vector<double> rolling_ofi_sum(const std::vector<double> &ofi, int window) {
    std::vector<double> result(ofi.size(), 0.0);
    double sum = 0.0;

    for (size_t i = 0; i < ofi.size(); ++i) {
        sum += ofi[i];
        if (i >= static_cast<size_t>(window)) {
            sum -= ofi[i - window];
        }
        result[i] = sum;
    }

    return result;
}

PYBIND11_MODULE(ofi_tools, m) {
    py::class_<LOBSTERRow>(m, "LOBSTERRow")
        .def_readwrite("time", &LOBSTERRow::time)
        .def_readwrite("type", &LOBSTERRow::type)
        .def_readwrite("price", &LOBSTERRow::price)
        .def_readwrite("direction", &LOBSTERRow::direction)
        .def_readwrite("bids", &LOBSTERRow::bids)
        .def_readwrite("asks", &LOBSTERRow::asks)
        .def_readwrite("bid_sizes", &LOBSTERRow::bid_sizes)
        .def_readwrite("ask_sizes", &LOBSTERRow::ask_sizes);

    m.def("read_lobster_files", &read_lobster_files);
    m.def("compute_ofi", &compute_ofi);
    m.def("rolling_ofi_sum", &rolling_ofi_sum);
}
