#include <bits/stdc++.h>
using namespace std;

int main()
{
    // actions 0 left, 1 up, 2 right, 3 down
    // State 0 A, 1 B, 2 C, 3 D
    double reward = 16;
    double delta = 0.2;
    double bellman = 0.01;
    double T[4][4][4];
    double R[4][4][4];
    vector<vector<double>> U;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                T[i][j][k] = 0;
                R[i][j][k] = 0;
            }
        }
    }
    T[1][0][0] = 0.8;
    R[1][0][0] = -1;
    T[1][0][1] = 0.2;
    R[1][0][1] = -1;
    T[0][1][0] = 0.2;
    R[0][1][0] = -1;
    T[0][1][2] = 0.8;
    R[0][1][2] = -1;
    T[1][1][1] = 0.2;
    R[1][1][1] = -1;
    T[1][1][3] = 0.8;
    R[1][1][3] = -4;
    T[0][2][0] = 0.2;
    R[0][2][0] = -1;
    T[0][2][1] = 0.8;
    R[0][2][1] = -1;
    T[2][2][2] = 0.75;
    R[2][2][2] = -1;
    T[2][2][3] = 0.25;
    R[2][2][3] = -3;
    T[2][3][0] = 0.8;
    R[2][3][0] = -1;
    T[2][3][2] = 0.2;
    R[2][3][2] = -1;

    vector<double> temp;
    temp.push_back(0);
    temp.push_back(0);
    temp.push_back(0);
    temp.push_back(reward);
    U.push_back(temp);

    cout << "t = 0, uA = 0, uB = 0, uC = 0, uD = " << reward << endl;

    double diff = 100000;
    int t = 1;
    bool once = false;
    while (diff > bellman)
    {
    last:
        cout << "________________________________________________" << endl;
        cout << "t = " << t << endl;
        cout << "For state A" << endl;
        double a = 0;
        double b = 0;
        cout << "UA" << t << "= max(" << endl;
        for (int i = 0; i < 4; ++i)
        {
            if (T[0][1][i])
            {
                cout << T[0][1][i] << "(" << R[0][1][i] << " + " << delta << "x" << U[t - 1][i] << ") + ";
                a += (T[0][1][i] * (R[0][1][i] + (delta * U[t - 1][i])));
            }
        }
        cout << "(Up)" << endl;
        for (int i = 0; i < 4; ++i)
        {
            if (T[0][2][i])
            {
                cout << T[0][2][i] << "(" << R[0][2][i] << " + " << delta << "x" << U[t - 1][i] << ") + ";
                b += (T[0][2][i] * (R[0][2][i] + (delta * U[t - 1][i])));
            }
        }
        cout << "(Right))" << endl;
        cout << "UA" << t << "= max(" << a << ", " << b << ")" << endl;
        temp[0] = max(a, b);
        cout << "UA" << t << "= " << temp[0] << endl
             << endl;

        cout << "For state B" << endl;
        a = 0;
        b = 0;
        cout << "UB" << t << "= max(" << endl;
        for (int i = 0; i < 4; ++i)
        {
            if (T[1][0][i])
            {
                cout << T[1][0][i] << "(" << R[1][0][i] << " + " << delta << "x" << U[t - 1][i] << ") + ";
                a += (T[1][0][i] * (R[1][0][i] + (delta * U[t - 1][i])));
            }
        }
        cout << "(Left)" << endl;
        for (int i = 0; i < 4; ++i)
        {
            if (T[1][1][i])
            {
                cout << T[1][1][i] << "(" << R[1][1][i] << " + " << delta << "x" << U[t - 1][i] << ") + ";
                b += (T[1][1][i] * (R[1][1][i] + (delta * U[t - 1][i])));
            }
        }
        cout << "(Up))" << endl;
        cout << "UB" << t << "= max(" << a << ", " << b << ")" << endl;
        temp[1] = max(a, b);
        cout << "UB" << t << "= " << temp[1] << endl
             << endl;

        cout << "For state C" << endl;
        a = 0;
        b = 0;
        cout << "UC" << t << "= max(" << endl;
        for (int i = 0; i < 4; ++i)
        {
            if (T[2][2][i])
            {
                cout << T[2][2][i] << "(" << R[2][2][i] << " + " << delta << "x" << U[t - 1][i] << ") + ";
                a += (T[2][2][i] * (R[2][2][i] + (delta * U[t - 1][i])));
            }
        }
        cout << "(Right)" << endl;
        for (int i = 0; i < 4; ++i)
        {
            if (T[2][3][i])
            {
                cout << T[2][3][i] << "(" << R[2][3][i] << " + " << delta << "x" << U[t - 1][i] << ") + ";
                b += (T[2][3][i] * (R[2][3][i] + (delta * U[t - 1][i])));
            }
        }
        cout << "(Down))" << endl;
        cout << "UC" << t << "= max(" << a << ", " << b << ")" << endl;
        temp[2] = max(a, b);
        cout << "UC" << t << "= " << temp[2] << endl
             << endl;

        temp[3] = reward;
        cout << "UD" << t << "= " << temp[3] << endl
             << endl;
        U.push_back(temp);
        double d1 = abs(U[t][0] - U[t - 1][0]);
        double d2 = abs(U[t][1] - U[t - 1][1]);
        double d3 = abs(U[t][2] - U[t - 1][2]);
        diff = max(max(d1, d2), d3);
        cout << "Maximal difference: " << diff << endl;
        ++t;
    }
    if (!once)
    {
        once = !once;
        goto last;
    }
}
