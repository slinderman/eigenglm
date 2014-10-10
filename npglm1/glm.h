#include <iostream>
#include <Eigen/Dense>
#include <vector>

#include "nptypes.h"

using namespace Eigen;
using namespace std;
using namespace nptypes;

class SpikeTrain
{
private:
    void initialize(int N, int T, double dt, VectorXd S, int D_imp, vector<MatrixXd> filtered_S);

public:
    int N;
    int T;
    double dt;
    VectorXd S;

    //TODO: Stimuli

    // Filtered spike train.
    int D_imp;
    vector<MatrixXd> filtered_S;

    SpikeTrain(int N, int T, double dt, VectorXd S, int D_imp, vector<MatrixXd> filtered_S);

    // Expose a constructor that uses buffers only, for Python.
    SpikeTrain(int N, int T, double dt, double* S_buffer, int D_imp, vector<double*> filtered_S_buffers);
};

class Component
{
public:
    //  Abstract base class for all components
    //
    virtual double log_probability() = 0;

    // Resample the parameters of this component
    virtual void resample() = 0;

    // Destructor
    virtual ~Component() {}
};

// Forward define the GLM class
class Glm;

class BiasCurrent : public Component
//class BiasCurrent
{
    Glm* parent;
public:
    double I_bias;

    BiasCurrent(Glm* glm, double bias);
    ~BiasCurrent() {}

    double log_probability();
    void coord_descent_step(double momentum);
    void resample();
};

class LinearImpulseCurrent : public Component
{
public:
    // A vector of impulse response weights for each presynaptic neuron.
    vector<VectorXd> w_ir;

    LinearImpulseCurrent(int N, int B);

    MatrixXd compute_current(SpikeTrain* st);

    double log_probability() {return 0.0; }
    void resample() {}
};

//class StimulusCurrent : public Component
//{
//public:
//
//    double log_probability();
//
//    void resample();
//};
//
//class NoStimulusCurrent : StimulusCurrent
//{
//}

class SmoothRectLinearLink : public Component
{
public:
    VectorXd compute_firing_rate(VectorXd I);

    VectorXd d_firing_rate_d_I(VectorXd I);

    double log_probability() {return 0.0; }
    void resample() {}
};

class Glm : public Component
{
public:
    // List of datasets
    std::vector<SpikeTrain*> spike_trains;

    // Subcomponents
    BiasCurrent *bias;
    LinearImpulseCurrent *impulse;
    SmoothRectLinearLink *nlin;
    VectorXd A;
    VectorXd W;

    Glm(int N, int D_imp);

    void add_spike_train(SpikeTrain *s);

    void get_firing_rate(SpikeTrain *s, VectorXd *fr);

    void get_firing_rate(SpikeTrain *s, double* fr_buffer);

    double log_likelihood();

    double log_probability();

    double d_ll_d_bias(SpikeTrain* s);

    double d_ll_d_bias(SpikeTrain* s, VectorXd I_stim, VectorXd I_net);

    VectorXd d_ll_d_I_imp(SpikeTrain* s, int n);

    VectorXd d_ll_d_I_imp(SpikeTrain* s, int n, double I_bias, VectorXd I_stim);

    void coord_descent_step(double momentum);

    void resample();
};

class Population : Component
{
public:
    vector<Glm*> glms;

    void add_glm(Glm *g);

    double log_probability();
    void resample();
};
