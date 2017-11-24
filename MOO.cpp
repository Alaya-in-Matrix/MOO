#include "MOO.h"
using namespace std;
using namespace Eigen;
MOO::MOO(MOO::ObjF f, size_t no, const VectorXd& lb, const VectorXd& ub)
    : _dim(lb.size()), _num_obj(no), _lb(lb), _ub(ub), _func(f)
{
    assert(lb.size() == ub.size() and (lb.array() < ub.array()).all());
    _engine    = mt19937_64(_seed);
    _sampled_x = MatrixXd(_dim, 0);
    _sampled_y = MatrixXd(_num_obj, 0);
}
void MOO::set_np(size_t np) { _np = np; }
void MOO::set_gen(size_t gen) { _gen = gen; }
void MOO::set_f(double f) { _f = f; }
void MOO::set_cr(double cr) { _cr = cr; }
void MOO::set_record(bool r) { _record_all = r; }
void MOO::set_seed(size_t seed) 
{ 
    _seed = seed;
    _engine.seed(_seed);
}
size_t MOO::get_seed() const { return _seed; }
void MOO::moo()
{
    _pop_x = _rand_matrix(_lb, _ub, _np);
    _pop_y = _run_func_batch(_pop_x);
    for(size_t i = 0; i < _gen; ++i)
    {
        const MatrixXd mutated    = _mutation(_f, _pop_x);
        const MatrixXd children_x = _crossover(_cr, _pop_x, mutated);
        const MatrixXd children_y = _run_func_batch(children_x);

        MatrixXd merged_x(_dim,  2 * _np);
        MatrixXd merged_y(_num_obj,  2 * _np);
        merged_x << _pop_x, children_x;
        merged_y << _pop_y, children_y;

        const vector<size_t> idxs = _nth_element(merged_y, _np);
        _pop_x = _slice_matrix(merged_x, idxs);
        _pop_y = _slice_matrix(merged_y, idxs);
    }
}
MatrixXd MOO::pareto_set() const
{
    return _record_all ? _slice_matrix(_sampled_x, _extract_pf(_sampled_y))
                       : _slice_matrix(_pop_x, _extract_pf(_pop_y));
}
MatrixXd MOO::pareto_front() const
{
    return _record_all ? _slice_matrix(_sampled_y, _extract_pf(_sampled_y))
                       : _slice_matrix(_pop_y, _extract_pf(_pop_y));
}

MatrixXd MOO::_mutation(double f, const MatrixXd& parent)
{
    const size_t np  = parent.cols();
    MatrixXd mutated = parent;
    std::uniform_int_distribution<long> distr(0, np);
    for(size_t i = 0; i < np; ++i)
    {
        const size_t i1 = distr(_engine);
        const size_t i2 = distr(_engine);
        mutated.col(i)  = parent.col(i) + f * (parent.col(i1) - parent.col(i2));
        mutated.col(i)  = mutated.col(i).cwiseMax(_lb).cwiseMin(_ub);
    }
    return mutated;
}

MatrixXd MOO::_crossover(double cr, const MatrixXd& parent, const MatrixXd& mutated)
{
    assert(mutated.cols() == parent.cols());
    assert(mutated.rows() == parent.rows());
    const size_t np  = parent.cols();
    const size_t dim = parent.rows();
    MatrixXd children(dim, np);
    uniform_real_distribution<double> prob_distr(0, 1);
    uniform_int_distribution<size_t>  idx_distr(0, dim - 1);
    for(size_t i = 0; i < np; ++i)
    {
        const size_t rand_idx = idx_distr(_engine);
        for(size_t j = 0; j < dim; ++j)
        {
            children(j, i) = prob_distr(_engine) <= cr || j == rand_idx ? mutated(j, i) : parent(j, i);
        }
    }
    return children;
}
std::vector<size_t> MOO::_seq_index(size_t n) const
{
    vector<size_t> v(n);
    for(size_t i = 0; i < n; ++i)
        v[i] = i;
    return v;
}

MatrixXd MOO::_slice_matrix(const MatrixXd& m, const std::vector<size_t>& indices) const
{
    assert(*max_element(indices.begin(), indices.end()) < (size_t)m.cols());
    MatrixXd slice(m.rows(), indices.size());
    for(size_t i = 0; i < indices.size(); ++i)
        slice.col(i) = m.col(indices[i]);
    return slice;
}
Eigen::VectorXd MOO::_run_func(const Eigen::VectorXd& param) // wrapper of _func
{
    VectorXd evaluated = _func(param);
    assert((size_t)param.size() == _dim);
    assert((size_t)evaluated.size() == _num_obj);
    if(_record_all)
    {
        _sampled_x.conservativeResize(Eigen::NoChange, _sampled_x.cols() + 1);
        _sampled_y.conservativeResize(Eigen::NoChange, _sampled_y.cols() + 1);
        _sampled_x.rightCols(1) = param;
        _sampled_y.rightCols(1) = evaluated;
    }
    return evaluated;
}
Eigen::MatrixXd MOO::_run_func_batch(const Eigen::MatrixXd& params) // wrapper of _func
{
    MatrixXd evaluated(_num_obj, params.cols());
    for(long i = 0; i < params.cols(); ++i)
        evaluated.col(i) = _run_func(params.col(i));
    return evaluated;
}
MatrixXd MOO::_rand_matrix(const Eigen::VectorXd& lb, const Eigen::VectorXd& ub, size_t num_col)
{
    const size_t dim = lb.size();
    MatrixXd m(dim, num_col);
    uniform_real_distribution<double> distr(-1, 1);
    const VectorXd _a = 0.5 * (ub - lb);
    const VectorXd _b = 0.5 * (ub + lb);
    for(size_t i = 0; i < num_col; ++i)
        for(size_t j = 0; j < dim; ++j)
            m(j, i) = distr(_engine);
    m = _a.replicate(1, num_col).cwiseProduct(m).colwise() + _b;
    return m;
}
    // bool _compare(const Eigen::VectorXd& obj1, const Eigen::VectorXd& obj2) const;
    // std::vector<size_t> _nth_element(const Eigen::MatrixXd& objs, size_t) const;
    // std::vector<size_t> _extract_pf(const Eigen::MatrixXd&) const;
