#ifndef __GLM_H_INCLUDED__
#define __GLM_H_INCLUDED__

#include <iostream>
#include <Eigen/Dense>
#include <vector>

#include "distributions.h"
#include "inference.h"
#include "nptypes.h"

using namespace Eigen;
using namespace std;
using namespace nptypes;

/**
 *  Class that encapsulates the spike train object and its corresponding data.
 */
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

    // Cache the currents
    vector<VectorXd> cached_I_imp_key;
    vector<VectorXd> cached_I_imp_val;

    vector<VectorXd> cached_I_stim_key;
    vector<VectorXd> cached_I_stim_val;

    SpikeTrain(int N, int T, double dt, VectorXd S, int D_imp, vector<MatrixXd> filtered_S);

    // Expose a constructor that uses buffers only, for Python.
    SpikeTrain(int N, int T, double dt, double* S_buffer, int D_imp, vector<double*> filtered_S_buffers);

};

/**
 *  Base class for components of the GLM.
 */
class Component
{
public:
    //  Abstract base class for all components
    //
    virtual double log_probability() = 0;

    // Coordinate descent inference.
    virtual void coord_descent_step(double momentum) = 0;

    // Resample the parameters of this component
    virtual void resample() = 0;

    // Destructor
    virtual ~Component() {}
};

// Forward define the GLM class
class Glm;

class BiasCurrent : public Component
{
    Distribution* prior;
    Glm* glm;

    // Nested class for sampling
    class BiasHmcSampler : public AdaptiveHmcSampler
    {
    BiasCurrent* parent;
    MatrixXd I_net;
    MatrixXd I_stim;

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

    BiasCurrent(Random* random, DiagonalGaussian* prior);
    ~BiasCurrent();

    // Getters and setters
    void set_glm(Glm* glm) { this->glm = glm; }
    double get_bias() { return I_bias; }

    // Component methods
    double log_probability();
    void coord_descent_step(double momentum);
    void resample();

};

class ImpulseCurrent : public Component
{
public:
    virtual MatrixXd compute_current(SpikeTrain* st) = 0;
    virtual MatrixXd compute_current_with_cache(SpikeTrain* st) = 0;
};

class LinearImpulseCurrent : public ImpulseCurrent
{
private:
    DiagonalGaussian* prior;
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
    LinearImpulseCurrent(int N, int D_imp, Random* random, DiagonalGaussian* prior);
    ~LinearImpulseCurrent();

    // Getters and setters
    void set_glm(Glm* glm) { this->glm = glm; }
    MatrixXd compute_current(SpikeTrain* st);
    MatrixXd compute_current_with_cache(SpikeTrain* st);

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

class DirichletImpulseCurrent : public ImpulseCurrent
{
private:
    // Dirichlet parameters (real valued, unconstrained)
    Dirichlet* prior;
    Glm* glm;
    int N, D_imp;

    // Nested class for sampling
    class ImpulseHmcSampler : public AdaptiveHmcSampler
    {
    int n_pre;
    DirichletImpulseCurrent* parent;
    public:
        ImpulseHmcSampler(DirichletImpulseCurrent* parent,
                          std::default_random_engine rng,
                          int n_steps=1);
        double logp(MatrixXd x);
        MatrixXd grad(MatrixXd x);
        void set_n_pre(int n_pre);
    };

    ImpulseHmcSampler* sampler;

public:
    // A vector of impulse response weights for each presynaptic neuron.
    vector<VectorXd> g_ir;
    vector<VectorXd> w_ir;

    // Constructor
    DirichletImpulseCurrent(int N, int D_imp, Random* random, Dirichlet* prior);
    ~DirichletImpulseCurrent();

    // Getters and setters
    void set_glm(Glm* glm) { this->glm = glm; }
    MatrixXd compute_current(SpikeTrain* st);
    MatrixXd compute_current_with_cache(SpikeTrain* st);
    void get_w(double* w_buffer);
    void get_g(double* w_buffer);
    void set_g(double* w_buffer);
    double log_probability() {return 0.0; }

    // Gradients
    VectorXd d_ll_d_g(SpikeTrain* st, int n);
    void d_ll_d_g(SpikeTrain* st, int n, double* dg_buffer);

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

class NetworkColumn : public Component
{
protected:
    Glm* glm;
public:
    VectorXd A;
    VectorXd W;

    // Getters and setters
    void get_A(VectorXd *A_buffer);
    void get_A(double *A_buffer);

    void get_W(VectorXd *W_buffer);
    void get_W(double *W_buffer);

};

class ConstantNetworkColumn : public NetworkColumn
{
public:
    ConstantNetworkColumn(int N);
    double log_probability() { return 0.0; }
    void coord_descent_step(double momentum) {}
    void resample() {}

    // Getters and setters
    void set_glm(Glm* glm) { this->glm = glm; }
};

class GaussianNetworkColumn : public NetworkColumn
{
    DiagonalGaussian* W_prior;
    IndependentBernoulli* A_prior;

public:
    GaussianNetworkColumn(Random* random,
                          DiagonalGaussian* W_prior,
                          IndependentBernoulli* A_prior
                          );

    double log_probability();
    void coord_descent_step(double momentum);
    void resample();

    // Getters and setters
    void set_glm(Glm* glm) { this->glm = glm; }
    void set_A(int n_pre, double a);
    void set_W(int n_pre, double w);
};

class SmoothRectLinearLink : public Component
{
public:
    VectorXd compute_firing_rate(VectorXd I);

    VectorXd d_firing_rate_d_I(VectorXd I);

    double log_probability() {return 0.0; }
    void resample() {}
    void coord_descent_step(double momentum) {}
};

class Glm : public Component
{
public:
    // Identity of this neuron
    int n;

    // Number of presynaptic neurons
    int N;

    // Subcomponents
    BiasCurrent *bias;
    ImpulseCurrent *impulse;
    SmoothRectLinearLink *nlin;
    NetworkColumn* network;

    // List of datasets
    std::vector<SpikeTrain*> spike_trains;

    // This is an abstract base class
    ~Glm();

    void add_spike_train(SpikeTrain *s);

    void get_firing_rate(SpikeTrain *s, VectorXd *fr);
    void get_firing_rate(SpikeTrain *s, double* fr_buffer);
    double log_prior();
    double log_likelihood();
    double log_probability();

    // Gradient calculations
    double d_ll_d_bias(SpikeTrain* s);
    double d_ll_d_bias(SpikeTrain* s, VectorXd I_stim, VectorXd I_net);
    VectorXd d_ll_d_I_imp(SpikeTrain* s, int n);
    VectorXd d_ll_d_I_imp(SpikeTrain* s, int n, double I_bias, VectorXd I_stim);

    // Inference
    void coord_descent_step(double momentum);
    void resample();
};

class StandardGlm : public Glm
{
private:
    void initialize(int n, int N, int D_imp, int seed);

public:
    // Constructor
    StandardGlm(int n, int N,
                Random* random,
                BiasCurrent* bias,
                LinearImpulseCurrent* impulse,
                SmoothRectLinearLink* nlin,
                ConstantNetworkColumn* network);
    ~StandardGlm() {}

};

class NormalizedGlm : public Glm
{
private:

public:
    // Constructor
    NormalizedGlm(int n, int N,
                  Random* random,
                  BiasCurrent* bias,
                  DirichletImpulseCurrent* impulse,
                  SmoothRectLinearLink* nlin,
                  GaussianNetworkColumn* network);
    ~NormalizedGlm() {}

};


#endif