#ifndef MFEM_LAGHOS_CSV
#define MFEM_LAGHOS_CSV

#include <fstream>
#include <iostream>

#include "mfem.hpp"

using namespace std;


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

        twep[count] = stof(row[0]);

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

int ReadTimeWindowParameters(const int nw, std::string twfile, Array<double>& twep, Array2D<int>& twparam, double sFactor[], const bool printStatus)
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
    const int nparamRead = 4; // number of parameters to read for each time window

    // Add 3 more parameters for nsamx, nsamv, nsame
    const int nparam = 7;

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

        twep[count] = stof(row[0]);
        for (int i=0; i<nparamRead-1; ++i)
            twparam(count,i) = stoi(row[i+1]);

        // Setting nsamx, nsamv, nsame
        twparam(count, 3) = sFactor[0] * twparam(count, 0);
        twparam(count, 4) = sFactor[1] * twparam(count, 1);
        twparam(count, 5) = sFactor[2] * twparam(count, 2);

        if (printStatus) cout << "Using time window " << count << " with end time " << twep[count] << ", rdimx " << twparam(count,0)
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

#endif // MFEM_LAGHOS_CSV
