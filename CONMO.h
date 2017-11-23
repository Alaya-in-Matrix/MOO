#pragma once
#include <Eigen/Dense>
#include <vector>
class CONMO
{
    // if _record_all is set to true, the all the evaluated points would be saved to calculate a pareto front
    bool   _record_all = false; 
    size_t _np         = 100;
    size_t _gen        = 200;
    double _f          = 0.8;
    double _cr         = 0.8;
    Eigen::MatrixXd _pop_x;
    Eigen::MatrixXd _pop_y;
    Eigen::MatrixXd _sampled_x;
    Eigen::MatrixXd _sampled_y;

    Eigen::VectorXd _mutation(double f, const Eigen::VectorXd& parent) const;
    Eigen::VectorXd _crossover(double cr, const Eigen::VectorXd& parent, const Eigen::VectorXd& mutated) const;
    bool _dominate(const Eigen::VectorXd& obj1, const Eigen::VectorXd& obj2) const;
    std::vector<size_t> _extract_pf(const Eigen::MatrixXd&) const;
    std::vector<size_t> _seq_index(size_t) const;
    Eigen::MatrixXd _slice_matrix(const Eigen::MatrixXd&, const std::vector<size_t>&) const;
public:
    typedef std::function<Eigen::VectorXd(const Eigen::VectorXd&)> ObjF;
    CONMO(ObjF, size_t num_o, size_t num_c, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);
    void set_np(size_t);
    void set_gen(size_t);
    void set_f(size_t);
    void set_cr(size_t);
    Eigen::VectorXd conmo();
};
