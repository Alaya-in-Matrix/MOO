#include "MOO.h"
#include <Eigen/Dense>
#include <limits>
using namespace std;
using namespace Eigen;
MOO::MOO(MOO::ObjF f, size_t no, const VectorXd& lb, const VectorXd& ub)
    : _dim(lb.size()), _num_obj(no), _lb(lb), _ub(ub), _func(f)
{
    assert(lb.size() == ub.size() and (lb.array() < ub.array()).all());
    _engine = mt19937_64(_seed);
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
    for (size_t i = 0; i < _gen; ++i)
    {
        const MatrixXd mutated    = _mutation(_f, _pop_x);
        const MatrixXd children_x = _crossover(_cr, _pop_x, mutated);
        const MatrixXd children_y = _run_func_batch(children_x);

        MatrixXd merged_x(_dim, 2 * _np);
        MatrixXd merged_y(_num_obj, 2 * _np);
        merged_x << _pop_x, children_x;
        merged_y << _pop_y, children_y;

        const vector<size_t> idxs = _nth_element(merged_y, _np);
        _pop_x = _slice_matrix(merged_x, idxs).leftCols(_np);
        _pop_y = _slice_matrix(merged_y, idxs).leftCols(_np);
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
    const size_t np = parent.cols();
    MatrixXd mutated = parent;
    uniform_int_distribution<long> distr(0, np - 1);
    for (size_t i = 0; i < np; ++i)
    {
        const size_t i1 = distr(_engine);
        const size_t i2 = distr(_engine);
        mutated.col(i) = parent.col(i) + f * (parent.col(i1) - parent.col(i2));
        mutated.col(i) = mutated.col(i).cwiseMax(_lb).cwiseMin(_ub);
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
    uniform_int_distribution<size_t> idx_distr(0, dim - 1);
    for (size_t i = 0; i < np; ++i)
    {
        const size_t rand_idx = idx_distr(_engine);
        for (size_t j = 0; j < dim; ++j)
        {
            children(j, i) = prob_distr(_engine) <= cr || j == rand_idx ? mutated(j, i) : parent(j, i);
        }
    }
    return children;
}
vector<size_t> MOO::_seq_index(size_t n) const
{
    vector<size_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = i;
    return v;
}

MatrixXd MOO::_slice_matrix(const MatrixXd& m, const vector<size_t>& indices) const
{
    assert(*max_element(indices.begin(), indices.end()) < (size_t)m.cols());
    MatrixXd slice(m.rows(), indices.size());
    for (size_t i = 0; i < indices.size(); ++i) slice.col(i) = m.col(indices[i]);
    return slice;
}
Eigen::VectorXd MOO::_run_func(const Eigen::VectorXd& param)  // wrapper of _func
{
    VectorXd evaluated = _func(param);
    assert((size_t)param.size() == _dim);
    assert((size_t)evaluated.size() == _num_obj);
    if (_record_all)
    {
        _sampled_x.conservativeResize(Eigen::NoChange, _sampled_x.cols() + 1);
        _sampled_y.conservativeResize(Eigen::NoChange, _sampled_y.cols() + 1);
        _sampled_x.rightCols(1) = param;
        _sampled_y.rightCols(1) = evaluated;
    }
    return evaluated;
}
Eigen::MatrixXd MOO::_run_func_batch(const Eigen::MatrixXd& params)  // wrapper of _func
{
    MatrixXd evaluated(_num_obj, params.cols());
    for (long i = 0; i < params.cols(); ++i) evaluated.col(i) = _run_func(params.col(i));
    return evaluated;
}
MatrixXd MOO::_rand_matrix(const Eigen::VectorXd& lb, const Eigen::VectorXd& ub, size_t num_col)
{
    const size_t dim = lb.size();
    MatrixXd m(dim, num_col);
    uniform_real_distribution<double> distr(-1, 1);
    const VectorXd _a = 0.5 * (ub - lb);
    const VectorXd _b = 0.5 * (ub + lb);
    for (size_t i = 0; i < num_col; ++i)
        for (size_t j = 0; j < dim; ++j) m(j, i) = distr(_engine);
    m = _a.replicate(1, num_col).cwiseProduct(m).colwise() + _b;
    return m;
}
bool MOO::_dominate(const VectorXd& obj1, const VectorXd& obj2) const
{
    return (obj1.array() <= obj2.array()).all() and obj1 != obj2;
}
vector<size_t> MOO::_extract_pf(const Eigen::MatrixXd& pnts) const
{
    vector<size_t> idxs;
    for (long i = 0; i < pnts.cols(); ++i)
    {
        bool dominated = false;
        for (long j = 0; j < pnts.cols(); ++j)
        {
            if (_dominate(pnts.col(j), pnts.col(i)))
            {
                dominated = true;
                break;
            }
        }
        if (not dominated) idxs.push_back(i);
    }
    return idxs;
}

VectorXi MOO::_dom_rank(const Eigen::MatrixXd& objs) const
{
    vector<size_t> num_dom(objs.cols(), 0);     // num_dom[i]: number of points that dominate point i
    vector<vector<long>> dom_set(objs.cols());  // dom_set[i]: set of points that are dominated by i
    vector<long> curr_front;
    VectorXi ranks(objs.cols());
    for (long i = 0; i < objs.cols(); ++i)
    {
        for (long j = 0; j < objs.cols(); ++j)
        {
            if (_dominate(objs.col(i), objs.col(j)))
                dom_set[i].push_back(j);
            else if (_dominate(objs.col(j), objs.col(i)))
                ++num_dom[i];
        }
        if (num_dom[i] == 0) curr_front.push_back(i);
    }
    size_t rank = 0;
    while (not curr_front.empty())
    {
        vector<long> tmp_front;
        for (auto idx : curr_front)
        {
            ranks(idx) = rank;
            for (auto dominated : dom_set[idx])
            {
                assert(num_dom[dominated] > 0);
                --num_dom[dominated];
                if (num_dom[dominated] == 0) tmp_front.push_back(dominated);
            }
        }
        ++rank;
        curr_front = tmp_front;
    }
    return ranks;
}

VectorXd MOO::_crowding_dist(const MatrixXd& objs, const VectorXi& ranks) const
{
    assert(objs.cols() == ranks.size());
    const size_t max_rank = ranks.maxCoeff();
    vector<vector<size_t>> rank_record(max_rank + 1);  // rank start from 0
    for (long i = 0; i < ranks.size(); ++i) rank_record[ranks[i]].push_back(i);

    VectorXd crowding_dist(objs.cols());
    for (size_t i = 0; i <= max_rank; ++i)
    {
        const MatrixXd fronts = _slice_matrix(objs, rank_record[i]);
        const VectorXd dists = _front_crowding_dist(fronts);
        for (size_t j = 0; j < rank_record[i].size(); ++j)
        {
            crowding_dist(rank_record[i][j]) = dists[j];
        }
    }
    return crowding_dist;
}
VectorXd MOO::_front_crowding_dist(const MatrixXd& front_objs) const
{
    // XXX: this is not the original crowding_dist in the NSGA-II paper
    const size_t num_spec = front_objs.rows();
    const size_t num_pnts = front_objs.cols();
    VectorXd dists = VectorXd::Ones(num_pnts);
    for (size_t i = 0; i < num_spec; ++i)
    {
        const RowVectorXd obj_vals = front_objs.row(i);
        vector<size_t> idxs = _seq_index(num_pnts);
        std::sort(idxs.begin(), idxs.end(), [&](long i1, long i2) -> bool { return obj_vals(i1) < obj_vals(i2); });

        for (size_t j = 0; j < num_pnts; ++j)
        {
            const long idx = idxs[j];
            if (num_pnts == 1)
                dists(idx) = numeric_limits<double>::infinity();  // boundary points are always favored
            else if (j == 0)
            {
                const long next_idx = idxs[j + 1];
                dists(idx) *= (obj_vals(next_idx) - obj_vals(idx));
            }
            else if (j == num_pnts - 1)
            {
                const long prev_idx = idxs[j - 1];
                dists(idx) *= (obj_vals(idx) - obj_vals(prev_idx));
            }
            else
            {
                const long prev_idx = idxs[j - 1];
                const long next_idx = idxs[j + 1];
                dists(idx) *= min(obj_vals(next_idx) - obj_vals(idx), obj_vals(idx) - obj_vals(prev_idx));
            }
        }
    }
    return dists;
}
std::vector<size_t> MOO::_nth_element(const MatrixXd& objs, size_t n) const
{
    const VectorXi ranks    = _dom_rank(objs);
    const VectorXd crow_vol = _crowding_dist(objs, ranks);
    vector<size_t> indices  = _seq_index(objs.cols());
    auto cmp = [&](const size_t i1, size_t i2) -> bool {
        return ranks(i1) < ranks(i2) or (ranks(i1) == ranks(i2) and crow_vol(i1) > crow_vol(i2));
    };
    std::nth_element(indices.begin(), indices.begin() + n, indices.end(), cmp);
    return indices;
}
MatrixXd MOO::dbx() const { return _record_all ? _sampled_x : _pop_x; }
MatrixXd MOO::dby() const { return _record_all ? _sampled_y : _pop_y; }
vector<size_t> MOO::nth_element(size_t n) const
{
    const MatrixXd& dby = _record_all ? _sampled_y : _pop_y; 
    return _nth_element(dby, n);
}
vector<size_t> MOO::sort() const
{
    const MatrixXd& dby = _record_all ? _sampled_y : _pop_y; 
    const VectorXi ranks    = _dom_rank(dby);
    const VectorXd crow_vol = _crowding_dist(dby, ranks);
    vector<size_t> indices  = _seq_index(dby.cols());
    auto cmp = [&](const size_t i1, size_t i2) -> bool {
        return ranks(i1) < ranks(i2) or (ranks(i1) == ranks(i2) and crow_vol(i1) > crow_vol(i2));
    };
    std::sort(indices.begin(), indices.end(), cmp);
    return indices;
}
size_t MOO::best() const { return sort().front(); }
