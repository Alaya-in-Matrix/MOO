#include "MOO.h"
#include <iostream>
using namespace std;
using namespace Eigen;
VectorXd testf(const VectorXd& xs)
{
    const double x  = xs(0);
    const double f1 = pow(x-1, 2);
    const double f2 = pow(x-3, 2);
    VectorXd obj(2);
    obj << f1, f2;
    return obj;
}
int main()
{
    VectorXd lb = VectorXd::Constant(1, 1, 0);
    VectorXd ub = VectorXd::Constant(1, 1, 10);
    MOO mo(testf, 2, lb, ub);
    mo.moo();
    cout << mo.pareto_set() << endl;
    return EXIT_SUCCESS;
}
