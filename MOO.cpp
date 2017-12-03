#include "MOO.h"
#include <Eigen/Dense>
#include <limits>
#include <iostream>
#include <unordered_set>
using namespace std;
using namespace Eigen;
MOO::MOO(MOO::ObjF f, size_t no, const VectorXd& lb, const VectorXd& ub)
    : _dim(lb.size()), _num_obj(no), _lb(lb), _ub(ub), _func(f)
{
    assert(lb.size() == ub.size() and (lb.array() < ub.array()).all());
    _engine = mt19937_64(_seed);
    // _sampled_x = MatrixXd(_dim, 0);
    // _sampled_y = MatrixXd(_num_obj, 0);
    _anchor_x  = MatrixXd(_dim, 0);
    _anchor_y  = MatrixXd(_num_obj, 0);
    _elitist_x = MatrixXd(_dim, 0);
    _elitist_y = MatrixXd(_num_obj, 0);
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
void MOO::set_anchor(const MatrixXd& x)
{
    assert((size_t)x.rows() == _dim);
    _anchor_x = x;
    _anchor_y = _run_func_batch(x);
}
size_t MOO::get_seed() const { return _seed; }
void MOO::moo()
{
    // if(_record_all)
    // {
    //     _sampled_x = MatrixXd::Zero(_dim,     _np * (1+_gen));
    //     _sampled_y = MatrixXd::Zero(_num_obj, _np * (1+_gen));
    // }
    _pop_x = _rand_matrix(_lb, _ub, _np);
    _pop_y = _run_func_batch(_pop_x);
    for (size_t i = 0; i < _gen; ++i)
    {
        MatrixXd mutated      = _mutation(_f, _pop_x);
        MatrixXd children_x   = _crossover(_cr, _pop_x, mutated);
        MatrixXd children_y   = _run_func_batch(children_x);

        // Setting from DEMO: Differential Evolution for Multiobjective Optimization
        vector<size_t> extra_child;
        vector<bool> is_child(_np);
        for(size_t j = 0; j < _np; ++j)
        {
            if(_dominate(children_y.col(j), _pop_y.col(j)))
            {
                _pop_x.col(j) = children_x.col(j);
                _pop_y.col(j) = children_y.col(j);
                is_child[j]   = true;
            }
            else if(not _dominate(_pop_y.col(j), children_y.col(j)))
            {
                extra_child.push_back(j);
                is_child.push_back(true);
            }
        }
        assert(is_child.size() == _pop_x.cols() + extra_child.size());

        MatrixXd extra_child_x = _slice_matrix(children_x, extra_child);
        MatrixXd extra_child_y = _slice_matrix(children_y, extra_child);
        MatrixXd merged_x(_dim,     _np + _anchor_x.cols() + extra_child_y.cols());
        MatrixXd merged_y(_num_obj, _np + _anchor_x.cols() + extra_child_y.cols());
        merged_x << _pop_x, extra_child_x, _anchor_x;
        merged_y << _pop_y, extra_child_y, _anchor_y;

        _ranks        = _dom_rank(merged_y);
        _crowding_vol = _crowding_dist(merged_y, _ranks);
        const vector<size_t> idxs = _nth_element(merged_y, _np);
        _pop_x = _slice_matrix(merged_x, idxs).leftCols(_np);
        _pop_y = _slice_matrix(merged_y, idxs).leftCols(_np);
        if(_record_all)
        {
            vector<size_t> best_rank_indices;
            for(long j = 0; j < _pop_x.cols() + extra_child_x.cols(); ++j)
            {
                if(_ranks(j) == 0 and is_child[j])
                    best_rank_indices.push_back(j);
            }
            _elitist_x.conservativeResize(Eigen::NoChange, _elitist_x.cols() + best_rank_indices.size());
            _elitist_y.conservativeResize(Eigen::NoChange, _elitist_y.cols() + best_rank_indices.size());
            _elitist_x.rightCols(best_rank_indices.size()) = _slice_matrix(merged_x, best_rank_indices);
            _elitist_y.rightCols(best_rank_indices.size()) = _slice_matrix(merged_y, best_rank_indices);
        }
    }
}
MatrixXd MOO::pareto_set() const
{
    return _record_all ? _slice_matrix(_elitist_x, _extract_pf(_elitist_y))
                       : _slice_matrix(_pop_x, _extract_pf(_pop_y));
}
MatrixXd MOO::pareto_front() const
{
    return _record_all ? _slice_matrix(_elitist_y, _extract_pf(_elitist_y))
                       : _slice_matrix(_pop_y, _extract_pf(_pop_y));
}

MatrixXd MOO::_mutation(double f, const MatrixXd& parent)
{
    const size_t np  = parent.cols();
    MatrixXd mutated = parent;
    uniform_int_distribution<long> distr(0, np - 1);
    for (size_t i = 0; i < np; ++i)
    {
        const size_t i0 = distr(_engine);
        const size_t i1 = distr(_engine);
        const size_t i2 = distr(_engine);
        mutated.col(i)  = parent.col(i0) + f * (parent.col(i1) - parent.col(i2));
        mutated.col(i)  = mutated.col(i).cwiseMax(_lb).cwiseMin(_ub);
    }
    return mutated;
}

Eigen::MatrixXd MOO::_polynomial_mutation(const Eigen::MatrixXd para, double rate, size_t idx)
{
    const size_t dim     = para.rows();
    const size_t num_pnt = para.cols();
    MatrixXd mutated     = para;
    uniform_real_distribution<double> distr(0, 1);
    for(size_t i = 0; i < num_pnt; ++i)
    {
        for(size_t j = 0; j < dim; ++j)
        {
            if(distr(_engine) <= rate)
            {
                const double r      = distr(_engine);
                const double sigma  = r < 0.5 ? pow(2 * r, 1.0 / (idx + 1)) - 1 : 1 - pow(2 - 2 * r, 1.0 / (idx + 1));
                mutated(j, i)      += sigma * (_ub(j) - _lb(j));
            }
        }
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
    if(indices.size() == 0)
        return MatrixXd(m.rows(), 0);
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
    // if (_record_all)
    // {
    //     _sampled_x.col(_eval_counter) = param;
    //     _sampled_y.col(_eval_counter) = evaluated;
    // }
    ++_eval_counter;
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
vector<size_t> MOO::_extract_pf2(const Eigen::MatrixXd& pnts) const
{
    // special case for bi-objective, O(NlogN) complexity
    vector<size_t> idxs = _seq_index(pnts.cols());
    // sort points according to the first objective
    std::sort(idxs.begin(), idxs.end(), [&](const size_t i1, size_t i2)->bool{
        return pnts(0, i1) < pnts(0, i2);
    });

    vector<size_t> pf{idxs[0]};
    size_t curr = idxs[0];
    for(long i = 1; i < pnts.cols(); ++i)
    {
        if(pnts(1, idxs[i]) < pnts(1, curr))
        {
            curr = idxs[i];
            pf.push_back(idxs[i]);
        }
    }
    return pf;
}
vector<size_t> MOO::_extract_pf(const Eigen::MatrixXd& pnts) const
{
    // Mishra, K. K., and Sandeep Harit. "A fast algorithm for finding the non
    // dominated set in multi objective optimization." International Journal of
    // Computer Applications 1.25 (2010): 35-39.
    // XXX: The paper is poorly written!!!
    if(pnts.rows() == 2)
        return _extract_pf2(pnts);
    vector<size_t> idxs = _seq_index(pnts.cols());

    // sort points according to the first objective
    std::sort(idxs.begin(), idxs.end(), [&](const size_t i1, size_t i2)->bool{
        return pnts(0, i1) < pnts(0, i2);
    });

    unordered_set<size_t> s{idxs[0]};
    for(size_t i = 1; i < idxs.size(); ++i)
    {
        size_t o       = idxs[i];
        bool dominated = false;
        for(auto it = s.begin(); it != s.end();)
        {
            if(_dominate(pnts.col(o), pnts.col(*it)))
            {
                it = s.erase(it);
            }
            else
            {
                if (_dominate(pnts.col(*it), pnts.col(o)))  // o is dominated
                    dominated = true;
                ++it;
            }
        }
        if(not dominated)
            s.insert(o);
    }
    vector<size_t> ns;
    std::copy(s.begin(), s.end(), std::back_inserter(ns));
    return ns;

    // vector<size_t> idxs;
    // for (long i = 0; i < pnts.cols(); ++i)
    // {
    //     bool dominated = false;
    //     for (long j = 0; j < pnts.cols(); ++j)
    //     {
    //         if (_dominate(pnts.col(j), pnts.col(i)))
    //         {
    //             dominated = true;
    //             break;
    //         }
    //     }
    //     if (not dominated) idxs.push_back(i);
    // }
    // return idxs;
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
    vector<size_t> indices  = _seq_index(objs.cols());
    auto cmp = [&](const size_t i1, size_t i2) -> bool {
        return _ranks(i1) < _ranks(i2) or (_ranks(i1) == _ranks(i2) and _crowding_vol(i1) > _crowding_vol(i2));
    };
    std::nth_element(indices.begin(), indices.begin() + n, indices.end(), cmp);
    return indices;
}
MatrixXd MOO::dbx() const { return _record_all ? _elitist_x : _pop_x; }
MatrixXd MOO::dby() const { return _record_all ? _elitist_y : _pop_y; }
vector<size_t> MOO::nth_element(size_t n) const
{
    const MatrixXd& dby = _record_all ? _elitist_y : _pop_y; 
    return _nth_element(dby, n);
}
vector<size_t> MOO::sort() const
{
    const MatrixXd& dby = _record_all ? _elitist_y : _pop_y; 
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
