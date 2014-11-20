#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <math.h>

#include "eigenglm.h"

using namespace Eigen;
using namespace std;


void SpikeTrain::initialize(int N, int T, double dt, VectorXd S, int D_imp, vector<MatrixXd> filtered_S)
{
    SpikeTrain::N = N;
    SpikeTrain::T = T;
    SpikeTrain::dt = dt;
    SpikeTrain::S = S;

    SpikeTrain::D_imp = D_imp;
    SpikeTrain::filtered_S = filtered_S;
}

SpikeTrain::SpikeTrain(int N, int T, double dt, VectorXd S, int D_imp, vector<MatrixXd> filtered_S)
{
    SpikeTrain::initialize(N, T, dt, S, D_imp, filtered_S);
}

SpikeTrain::SpikeTrain(int N, int T, double dt, double* S_buffer, int D_imp, vector<double*> filtered_S_buffers)
{
    // Copy spike train into new buffer
    NPVector<double> np_S(S_buffer, T);
    VectorXd S(T);
    S = np_S;

    vector<MatrixXd> filtered_S;
    for (int n=0; n<N; n++)
    {
        // Copy filtered_S[n] into a new matrix
        NPMatrix<double> np_fS(filtered_S_buffers[n], T, D_imp);
        MatrixXd fS(T, D_imp);
        fS = np_fS;
        filtered_S.push_back(fS);

        // Initialize the caches
        cached_I_imp_key.push_back(VectorXd(D_imp));
        cached_I_imp_val.push_back(VectorXd(T));

        cached_I_stim_key.push_back(VectorXd(D_imp));
        cached_I_stim_val.push_back(VectorXd(T));
    }

    SpikeTrain::initialize(N, T, dt, S, D_imp, filtered_S);
}


/**
 *  Bias current.
 */
BiasCurrent::BiasCurrent(Random* random, DiagonalGaussian* prior)
{
    // Initialize the bias
    this->prior = prior;
    I_bias = prior->sample()(0);

    // Initialize the sampler. The number of steps is set in glm.h
    sampler = new BiasHmcSampler(this, random->rng);
}

BiasCurrent::~BiasCurrent() {}

double BiasCurrent::log_probability()
{
    NPVector<double> np_I_bias(&I_bias, 1);
    return prior->logp(np_I_bias);
}

void BiasCurrent::coord_descent_step(double momentum)
{
    MatrixXd mI_bias = MatrixXd::Constant(1, 1, I_bias);
    MatrixXd mgrad = MatrixXd::Zero(1,1);

    // Overwrite grad with the prior gradient
    prior->grad(mI_bias, &mgrad);

    double grad = mgrad(0);

    // Get the gradient with respect to each spike train
    for (vector<SpikeTrain*>::iterator s = glm->spike_trains.begin();
         s != glm->spike_trains.end();
         ++s)
    {
        grad += glm->d_ll_d_bias(*s);
    }

    // Update bias
    I_bias += momentum * grad;
}

/**
 *  Bias HMC Sampler
 */
BiasCurrent::BiasHmcSampler::BiasHmcSampler(BiasCurrent* parent,
                                            std::default_random_engine rng,
                                            int n_steps) :
                                            AdaptiveHmcSampler(rng, n_steps)
{
    this->parent = parent;
}


double BiasCurrent::BiasHmcSampler::logp(MatrixXd x)
{
    // Set the bias
    parent->I_bias = x(0,0);

    return parent->glm->log_probability();
}

MatrixXd BiasCurrent::BiasHmcSampler::grad(MatrixXd x)
{
    // Set the bias
    parent->I_bias = x(0);

    // Initialize the output
    MatrixXd mI_bias = MatrixXd::Constant(1, 1, parent->I_bias);
//    Map<MatrixXd> mI_bias(&I_bias, 1,1);
    MatrixXd grad = MatrixXd::Zero(1,1);

    // Overwrite grad with the prior gradient
    parent->prior->grad(mI_bias, &grad);

    Glm* glm = parent->glm;

    // Get the gradient with respect to each spike train
    for (vector<SpikeTrain*>::iterator s = glm->spike_trains.begin();
         s != glm->spike_trains.end();
         ++s)
    {
        grad.array() += glm->d_ll_d_bias(*s);
    }

    return grad;
}

void BiasCurrent::resample()
{
    // Use the BiasHmcSampler to update the bias
    MatrixXd x_curr(1,1);
    x_curr(0,0) = I_bias;
    MatrixXd x_next(1,1);

    sampler->sample(x_curr, &x_next);

    // Set the new bias
    I_bias = x_next(0,0);
}



/**
 *  Nonlinearity classes.
 */

inline double rect(double x) { return (x < 0.0 ? exp(x) : 1+x); }
inline double drect_dx(double x) { return (x < 0.0 ? exp(x) : 1.0); }

VectorXd SmoothRectLinearLink::compute_firing_rate(VectorXd I)
{
//    return (1.0 + I.array().exp()).log();
    return I.unaryExpr(std::ptr_fun(rect));
}

VectorXd SmoothRectLinearLink::d_firing_rate_d_I(VectorXd I)
{
    // Gradient of the firing rate with respect to I
//    return I.array().exp() / (1.0 + I.array().exp());
    return I.unaryExpr(std::ptr_fun(drect_dx));
}

/**
 *  Network classes.
 */
void NetworkColumn::get_A(VectorXd* A_buffer)
{
    *A_buffer = A;
}

void NetworkColumn::get_A(double* A_buffer)
{
    NPVector<double> np_A(A_buffer, A.size());

    // Copy the result back to A
    np_A = A;
}

void NetworkColumn::get_W(VectorXd* W_buffer)
{
    *W_buffer = W;
}

void NetworkColumn::get_W(double* W_buffer)
{
    NPVector<double> np_W(W_buffer, W.size());

    // Copy the result back to W
    np_W = W;
}

ConstantNetworkColumn::ConstantNetworkColumn(int N)
{
    // Initialize a constant
    A = VectorXd::Ones(N);
    W = VectorXd::Ones(N);
}

GaussianNetworkColumn::GaussianNetworkColumn(Random* random,
                                             DiagonalGaussian* W_prior,
                                             IndependentBernoulli* A_prior)
{
    this->W_prior = W_prior;
    this->A_prior = A_prior;

    // Sample weight and adjacency matrices from the prior
    A = A_prior->sample();
    W = W_prior->sample();
}

double GaussianNetworkColumn::log_probability()
{
    return A_prior->logp(A) + W_prior->logp(W);
}

void GaussianNetworkColumn::resample() {}
void GaussianNetworkColumn::coord_descent_step(double momentum) {}

void GaussianNetworkColumn::set_A(int n_pre, double a)
{
    A(n_pre) = a;
}

void GaussianNetworkColumn::set_W(int n_pre, double w)
{
    W(n_pre) = w;
}

/**
 *  GLM class
 */
Glm::~Glm() {}

void Glm::add_spike_train(SpikeTrain *s)
{
    spike_trains.push_back(s);
}

void Glm::get_firing_rate(SpikeTrain* s, VectorXd *fr)
{
    // Compute the total current for this spike train.
    VectorXd I = VectorXd::Constant(s->T, 0.0);

    // Bias is a constant.
    I = I.array() + bias->I_bias;

    // Add the weighted impulse responses
    MatrixXd I_imp = impulse->compute_current_with_cache(s);
    I += I_imp * (network->A.array() * network->W.array()).matrix();

    // Compute the firing rate and its log.
    *fr = nlin->compute_firing_rate(I);
}

void Glm::get_firing_rate(SpikeTrain* s, double* fr_buffer)
{
    NPVector<double> fr(fr_buffer, s->T);
    VectorXd vec_fr;
    Glm::get_firing_rate(s, &vec_fr);

    // Copy the result back to fr
    fr = vec_fr;
}

double Glm::log_prior()
{
    double lp = 0.0;

    // Add subcomponent priors
    lp += bias->log_probability();
    lp += impulse->log_probability();
    lp += nlin->log_probability();
    lp += network->log_probability();
    return lp;
}

double Glm::log_likelihood()
{
    double ll = 0;
    for (vector<SpikeTrain*>::iterator it = spike_trains.begin();
         it != spike_trains.end();
         ++it)
    {
        SpikeTrain* s = *it;
        VectorXd lam;
        Glm::get_firing_rate(s, &lam);
        VectorXd loglam = lam.array().log();

        // Compute the Poisson likelihood.
        ll += -1 * s->dt * lam.sum() + s->S.dot(loglam);

    }
    return ll;
}

double Glm::log_probability()
{
    // Add the prior.
    double lp = log_prior();

    // Add the likelihood.
    lp += log_likelihood();

    return lp;
}

// Getters and setters


// Gradients
double Glm::d_ll_d_bias(SpikeTrain* s)
{
    // Overloaded function if we don't have I_stim or I_net precomputed
    MatrixXd I_imp = impulse->compute_current(s);
    VectorXd I_net = I_imp * (network->A.array() * network->W.array()).matrix();
    VectorXd I_stim = VectorXd::Constant(I_net.size(), 0);

    return d_ll_d_bias(s, I_stim, I_net);
}

double Glm::d_ll_d_bias(SpikeTrain* s, VectorXd I_stim, VectorXd I_net)
{
    // Compute the gradient of the log likelihood with respect to the
    // First, compute the total current for this spike train.
    VectorXd I = VectorXd(I_stim + I_net);

    // Bias is a constant.
    I = I.array() + bias->I_bias;

    // Compute the firing rate and its log.
    VectorXd lam = nlin->compute_firing_rate(I);
    VectorXd loglam = lam.array().log();

    // Now compute the gradients.
    // TODO: Avoid divide by zero
    VectorXd d_ll_d_lam = -1*(s->dt) + s->S.array()/lam.array();
    VectorXd d_lam_d_I = nlin->d_firing_rate_d_I(I);
    double d_I_d_bias = 1.0;

    // Multiply em up!
    double d_ll_d_bias = d_ll_d_lam.dot(d_lam_d_I) * d_I_d_bias;

    return d_ll_d_bias;
}

VectorXd Glm::d_ll_d_I_imp(SpikeTrain* s, int n)
{
    VectorXd I_stim = VectorXd::Constant(s->T, 0);
    double I_bias = bias->I_bias;

    return d_ll_d_I_imp(s, n, I_bias, I_stim);
}

VectorXd Glm::d_ll_d_I_imp(SpikeTrain* s, int n, double I_bias, VectorXd I_stim)
{
    // Compute the gradient of the log likelihood with respect to the
    // First, compute the total current for this spike train.
    VectorXd I = I_bias + I_stim.array();

    // Add in the impulse current
    MatrixXd I_imp = impulse->compute_current(s);
    VectorXd I_net = I_imp * (network->A.array() * network->W.array()).matrix();
    I += I_net;

    // Compute the firing rate and its log.
    VectorXd lam = nlin->compute_firing_rate(I);
    VectorXd loglam = lam.array().log();

    // Now compute the gradients.
    // TODO: Avoid divide by zero
    VectorXd d_ll_d_lam = VectorXd::Constant(s->T, -s->dt).array() + s->S.array()/lam.array();
    VectorXd d_lam_d_I = nlin->d_firing_rate_d_I(I);
    double d_I_d_I_imp_n = network->A(n)  * network->W(n);

    // Multiply em up!
    VectorXd d_ll_d_I_imp_n = d_ll_d_lam.array() * d_lam_d_I.array() * d_I_d_I_imp_n;

    return d_ll_d_I_imp_n;
}

void Glm::coord_descent_step(double momentum)
{
    // Call subcomponent resample methods.
    bias->coord_descent_step(momentum);
    impulse->coord_descent_step(momentum);
    network->coord_descent_step(momentum);
}

void Glm::resample()
{
    // Call subcomponent resample methods.
    bias->resample();
    impulse->resample();
    network->resample();
}

/**
 *  Standard GLM implementation
 */
StandardGlm::StandardGlm(int n, int N,
                         Random* random,
                         BiasCurrent* bias,
                         LinearImpulseCurrent* impulse,
                         SmoothRectLinearLink* nlin,
                         ConstantNetworkColumn* network)
{
    this->n = n;
    this->N = N;

    // Set the child links
    this->bias = bias;
    this->impulse = impulse;
    this->nlin = nlin;
    this->network = network;

    // Set the parent links
    bias->set_glm(this);
    impulse->set_glm(this);

}

/**
 *  Normalized GLM implementation
 */
NormalizedGlm::NormalizedGlm(int n, int N,
                             Random* random,
                             BiasCurrent* bias,
                             DirichletImpulseCurrent* impulse,
                             SmoothRectLinearLink* nlin,
                             GaussianNetworkColumn* network)
{
    this->n = n;
    this->N = N;

    // Set the child links
    this->bias = bias;
    this->impulse = impulse;
    this->nlin = nlin;
    this->network = network;

    // Set the parent links
    bias->set_glm(this);
    impulse->set_glm(this);
    network->set_glm(this);

}

