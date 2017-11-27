#include "MOO.h"
#include "Eigen/Dense"
#include <iostream>
#include <cassert>
#include <cmath>
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
VectorXd zdt1(const VectorXd& xs)
{
    assert(xs.size() > 1);
    const size_t dim = xs.size();
    VectorXd rest_xs = xs.tail(xs.cols() - 1);
    const double f1 = xs(0);
    const double g  = 1.0 + 9.0 / (dim - 1) * rest_xs.sum();
    const double f2 = g * (1.0 - sqrt(f1 / g));
    VectorXd objs(2);
    objs << f1, f2;
    return objs;
}
VectorXd zdt2(const VectorXd& xs)
{
    assert(xs.size() > 1);
    const size_t dim = xs.size();
    VectorXd rest_xs = xs.tail(xs.cols() - 1);
    const double f1 = xs(0);
    const double g  = 1.0 + 9.0 / (dim - 1) * rest_xs.sum();
    const double f2 = g * (1.0 - pow(f1/g, 2));
    VectorXd objs(2);
    objs << f1, f2;
    return objs;
}
VectorXd zdt3(const VectorXd& xs)
{
    assert(xs.size() > 1);
    const size_t dim = xs.size();
    VectorXd rest_xs = xs.tail(xs.cols() - 1);
    const double f1 = xs(0);
    const double g  = 1.0 + 9.0 / (dim - 1) * rest_xs.sum();
    const double f2 = g * (1 - sqrt(f1 / g) - f1 / g * sin(10 * M_PI * f1));
    VectorXd objs(2);
    objs << f1, f2;
    return objs;
}
int main()
{
    VectorXd lb = VectorXd::Constant(10, 1, 0);
    VectorXd ub = VectorXd::Constant(10, 1, 1);
    MOO mo(zdt3, 2, lb, ub);
    mo.set_gen(100);
    mo.set_np(100);
    mo.set_record(true);
    mo.moo();
    cout << mo.pareto_front() << endl;
    return EXIT_SUCCESS;
}
