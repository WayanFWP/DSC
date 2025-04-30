#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

int classifyPhase(double heel, double toe) {
    if (heel > 0.5 && toe > 0.5) return 0;       // IC
    else if (heel < 0.2 && toe > 0.5) return 1;  // FF
    else if (heel > 0.5 && toe < 0.2) return 2;  // HO
    else if (heel > 0.5 && toe >= 0.2 && toe <= 0.5) return 3; // MSt
    else if (heel < 0.2 && toe >= 0.2 && toe <= 0.5) return 4; // TO
    else return 5;  // Sw
}

int main() {
    std::ifstream infile("data/dataset.txt");
    std::ofstream outfile("data/training_project1_set.txt");

    if (!infile) {
        std::cerr << "File input tidak ditemukan.\n";
        return 1;
    }

    if (!outfile) {
        std::cerr << "Tidak bisa membuka file output.\n";
        return 1;
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        double x1, x2;
        if (!(iss >> x1 >> x2)) continue;

        int label = classifyPhase(x1, x2);
        outfile << x1 << " " << x2 << " " << label << "\n";
    }

    std::cout << "Label otomatis selesai. File disimpan di: data/labeled_dataset.txt\n";
    return 0;
}
