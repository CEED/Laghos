#include <fstream>
#include <iostream>

#include "laghos_utils.hpp"

using namespace std;

void BasisGeneratorFinalSummary(CAROM::SVDBasisGenerator* bg, const double energyFraction, int & cutoff)
{
    const int rom_dim = bg->getSpatialBasis()->numColumns();
    const CAROM::Matrix* sing_vals = bg->getSingularValues();

    MFEM_VERIFY(rom_dim == sing_vals->numColumns(), "");

    double sum = 0.0;
    for (int sv = 0; sv < sing_vals->numColumns(); ++sv) {
        sum += (*sing_vals)(sv, sv);
    }

    double partialSum = 0.0;
    for (int sv = 0; sv < sing_vals->numColumns(); ++sv) {
        partialSum += (*sing_vals)(sv, sv);
        if (partialSum / sum > energyFraction)
        {
            cutoff = sv+1;
            break;
        }
    }

    cout << "Take first " << cutoff << " of " << sing_vals->numColumns() << " basis vectors" << endl;
}

void PrintSingularValues(const int rank, const std::string& basename, const std::string& name, CAROM::SVDBasisGenerator* bg, const bool usingWindows, const int window)
{
    const CAROM::Matrix* sing_vals = bg->getSingularValues();

    char tmp[100];
    sprintf(tmp, ".%06d", rank);

    std::string fullname = (usingWindows) ? basename + "/sVal" + name + std::to_string(window) + tmp : basename + "/sVal" + name + tmp;

    std::ofstream ofs(fullname.c_str(), std::ofstream::out);
    ofs.precision(16);

    for (int sv = 0; sv < sing_vals->numColumns(); ++sv) {
        ofs << (*sing_vals)(sv, sv) << endl;
    }

    ofs.close();
}

int ReadTimeWindows(const int nw, std::string twfile, Array<double>& twep, const bool printStatus)
{
    if (printStatus) cout << "Reading time windows from file " << twfile << endl;

    std::ifstream ifs(twfile.c_str());

    if (!ifs.is_open())
    {
        cout << "Error: invalid file" << endl;
        return 1;  // invalid file
    }

    twep.SetSize(nw);

    string line, word;
    int count = 0;
    while (getline(ifs, line))
    {
        if (count >= nw)
        {
            cout << "Error reading CSV file. Read more than " << nw << " lines" << endl;
            ifs.close();
            return 3;
        }

        stringstream s(line);
        vector<string> row;

        while (getline(s, word, ','))
            row.push_back(word);

        if (row.size() != 1)
        {
            cout << "Error: CSV file does not specify exactly 1 parameter" << endl;
            ifs.close();
            return 2;  // incorrect number of parameters
        }

        twep[count] = stod(row[0]);

        if (printStatus) cout << "Using time window " << count << " with end time " << twep[count] << endl;
        count++;
    }

    ifs.close();

    if (count != nw)
    {
        cout << "Error reading CSV file. Read " << count << " lines but expected " << nw << endl;
        return 3;
    }

    return 0;
}

int ReadTimeWindowParameters(const int nw, std::string twfile, Array<double>& twep, Array2D<int>& twparam, double sFactor[], const bool printStatus, const bool rhs)
{
    if (printStatus) cout << "Reading time window parameters from file " << twfile << endl;

    std::ifstream ifs(twfile.c_str());

    if (!ifs.is_open())
    {
        cout << "Error: invalid file" << endl;
        return 1;  // invalid file
    }

    // Parameters to read for each time window:
    // end time, rdimx, rdimv, rdime
    const int nparamRead = rhs ? 6 : 4; // number of parameters to read for each time window

    // Add 3 more parameters for nsamx, nsamv, nsame
    const int nparam = nparamRead + 3;

    twep.SetSize(nw);
    twparam.SetSize(nw, nparam - 1);

    string line, word;
    int count = 0;
    while (getline(ifs, line))
    {
        if (count >= nw)
        {
            cout << "Error reading CSV file. Read more than " << nw << " lines" << endl;
            ifs.close();
            return 3;
        }

        stringstream s(line);
        vector<string> row;

        while (getline(s, word, ','))
            row.push_back(word);

        if (row.size() != nparamRead)
        {
            cout << "Error: CSV file does not specify " << nparamRead << " parameters" << endl;
            ifs.close();
            return 2;  // incorrect number of parameters
        }

        twep[count] = stod(row[0]);
        for (int i=0; i<nparamRead-1; ++i)
            twparam(count,i) = stoi(row[i+1]);

        // Setting nsamx, nsamv, nsame
        twparam(count, nparamRead-1) = sFactor[0] * twparam(count, 0);
        twparam(count, nparamRead)   = sFactor[1] * twparam(count, rhs ? 3 : 1);
        twparam(count, nparamRead+1) = sFactor[2] * twparam(count, rhs ? 4 : 2);

        if (rhs && printStatus) cout << "Using time window " << count << " with end time " << twep[count] << ", rdimx " << twparam(count,0)
                                         << ", rdimv " << twparam(count,1) << ", rdime " << twparam(count,2) << ", rdimfv " << twparam(count,3)
                                         << ", rdimfe " << twparam(count,4) << ", nsamx " << twparam(count,5)
                                         << ", nsamv " << twparam(count,6) << ", nsame " << twparam(count,7) << endl;
        else if (printStatus) cout << "Using time window " << count << " with end time " << twep[count] << ", rdimx " << twparam(count,0)
                                       << ", rdimv " << twparam(count,1) << ", rdime " << twparam(count,2) << ", nsamx " << twparam(count,3)
                                       << ", nsamv " << twparam(count,4) << ", nsame " << twparam(count,5) << endl;

        count++;
    }

    ifs.close();

    if (count != nw)
    {
        cout << "Error reading CSV file. Read " << count << " lines but expected " << nw << endl;
        return 3;
    }

    return 0;
}
