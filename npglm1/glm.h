#ifndef __GLM_H_INCLUDED__
#define __GLM_H_INCLUDED__

#include <iostream>
#include <Eigen/Dense>
#include <vector>

#include "inference.h"
#include "nptypes.h"

using namespace Eigen;
using namespace std;
using namespace nptypes;

class SpikeTrain
{
private:
    // Does the shared work of the constructors.
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
//class BiasCurrent;

class BiasCurrent : public Component
{
    Glm* parent;

    // Nested class for sampling
    class BiasHmcSampler : public AdaptiveHmcSampler
    {
    BiasCurrent* parent;
    public:
        BiasHmcSampler(BiasCurrent* parent,
                       std::default_random_engine rng,
                       int n_steps=1);
        double logp(MatrixXd x);
        MatrixXd grad(MatrixXd x);
    };

    BiasHmcSampler* sampler;

public:
    double I_bias;

    BiasCurrent(Glm* glm, double bias, std::default_random_engine rng);
    ~BiasCurrent() {}

    double log_probability();
    void coord_descent_step(double momentum);
    void resample();

    double get_bias() { return I_bias; }
};

class LinearImpulseCurrent : public Component
{
private:
    Glm* glm;
    int N, D_imp;

    // Nested class for sampling
    class ImpulseHmcSampler : public AdaptiveHmcSampler
    {
    int n_pre;
    LinearImpulseCurrent* parent;
    public:
        ImpulseHmcSampler(LinearImpulseCurrent* parent,
                          std::default_random_engine rng,
                          int n_steps=1);
        double logp(MatrixXd x);
        MatrixXd grad(MatrixXd x);
        void set_n_pre(int n_pre);
    };

    ImpulseHmcSampler* sampler;

public:
    // A vector of impulse response weights for each presynaptic neuron.
    vector<VectorXd> w_ir;

    // Constructor
    LinearImpulseCurrent(Glm* glm, int N, int D_imp, std::default_random_engine rng);

    // Getters
    MatrixXd compute_current(SpikeTrain* st);
    void get_w(double* w_buffer);
    void set_w(double* w_buffer);
    double log_probability() {return 0.0; }

    // Gradients
    VectorXd d_ll_d_w(SpikeTrain* st, int n);
    void d_ll_d_w(SpikeTrain* st, int n, double* dw_buffer);

    // Inference
    void coord_descent_step(double momentum);
    void resample();

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
private:
    void initialize(int N, int D_imp, int seed);
public:
    // List of datasets
    std::vector<SpikeTrain*> spike_trains;

    // Subcomponents
    BiasCurrent *bias;
    LinearImpulseCurrent *impulse;
    SmoothRectLinearLink *nlin;
    VectorXd A;
    VectorXd W;

    // Constructor
    Glm(int N, int D_imp);
    Glm(int N, int D_imp, int seed);

    ~Glm();

    void add_spike_train(SpikeTrain *s);

    // Getters
    BiasCurrent* get_bias_component() { return bias; }

    LinearImpulseCurrent* get_impulse_component() { return impulse; }

    void get_firing_rate(SpikeTrain *s, VectorXd *fr);

    void get_firing_rate(SpikeTrain *s, double* fr_buffer);

    double log_likelihood();

    double log_probability();

    // Gradient calculations
    double d_ll_d_bias(SpikeTrain* s);

    double d_ll_d_bias(SpikeTrain* s, VectorXd I_stim, VectorXd I_net);

    VectorXd d_ll_d_I_imp(SpikeTrain* s, int n);

    VectorXd d_ll_d_I_imp(SpikeTrain* s, int n, double I_bias, VectorXd I_stim);

    // Coordinate descent inference
    void coord_descent_step(double momentum);

    // MCMC
    void resample();
};


#endif