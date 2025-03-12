#ifndef FILTER_HPP
#define FILTER_HPP


#include <fstream>
#include <jsoncpp/json/json.h>


struct Filter {
    cv::Vec3d v;
    cv::Scalar p0;
    double t1, t2;
    double r;

    Filter() = default;
    Filter(cv::Vec3d& v, cv::Scalar& p0, double t1, double t2, double r) : v{v}, p0{p0}, t1{t1}, t2{t2}, r{r} {}
    
    void writeToJson(std::string path);
    static Filter readFromJson(std::string path);
};


void Filter::writeToJson(std::string path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "unable to open file for writing" << std::endl;
        exit(2);
    }

    Json::Value root;

    Json::Value v_val;
    v_val.append(v[0]);
    v_val.append(v[1]);
    v_val.append(v[2]);
    root["v"] = v_val;

    Json::Value p0_val;
    p0_val.append(p0[0]);
    p0_val.append(p0[1]);
    p0_val.append(p0[2]);
    p0_val.append(p0[3]);
    root["p0"] = p0_val;

    root["t1"] = t1;
    root["t2"] = t2;
    root["r"] = r;

    Json::StreamWriterBuilder writer;
    std::string jsonString = Json::writeString(writer, root);
    file << jsonString;
    file.close();
    std::cout << "data written to " << path << std::endl;
}


Filter Filter::readFromJson(std::string path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "unable to open file for reading" << std::endl;
        exit(3);
    }

    Json::Value root;
    file >> root;
    file.close();

    cv::Vec3d v(
        root["v"][0].asDouble(),
        root["v"][1].asDouble(),
        root["v"][2].asDouble()
    );

    cv::Scalar p0(
        root["p0"][0].asDouble(),
        root["p0"][1].asDouble(),
        root["p0"][2].asDouble(),
        root["p0"][3].asDouble()
    );

    double t1 = root["t1"].asDouble();
    double t2 = root["t2"].asDouble();
    double r = root["r"].asDouble();

    return Filter(v, p0, t1, t2, r);
}


#endif  // FILTER_HPP