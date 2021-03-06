#include <Eigen/Dense>
#include <vector>

#include "eigenglm.h"

using namespace Eigen;
using namespace std;

/**
 *  Impulse response classes.
 */
LinearImpulseCurrent::LinearImpulseCurrent(int N, int D_imp, Random* random, DiagonalGaussian* prior)
{
    this->prior = prior;

    this->N = N;
    this->D_imp = D_imp;

    // Initialize impulse response weights.
    for (int n=0; n < N; n++)
    {
        w_ir.push_back(prior->sample());
    }

    // Initialize the sampler. The number of steps is set in glm.h
    sampler = new ImpulseHmcSampler(this, random->rng);
}

LinearImpulseCurrent::~LinearImpulseCurrent()
{
    if (sampler) { delete sampler; }
}

VectorXd LinearImpulseCurrent::d_ll_d_w(SpikeTrain* st, int n)
{
    // For the linear impulse response I_imp = filtered_S * w
    // so d_I_imp_d_w = filtered_S
    return (glm->d_ll_d_I_imp(st, n).transpose() * st->filtered_S[n]);
}

void LinearImpulseCurrent::d_ll_d_w(SpikeTrain* st, int n, double* dw_buffer)
{
    // Copy the impulse response weights into a buffer
    NPVector<double> dw(dw_buffer, LinearImpulseCurrent::D_imp);
    dw = LinearImpulseCurrent::d_ll_d_w(st, n);
}

void LinearImpulseCurrent::coord_descent_step(double momentum)
{
    // Update each of the impulse response weights
    for (int n=0; n<N; n++)
    {
        MatrixXd grad(D_imp,1);
        prior->grad(w_ir[n], &grad);

        // Get the gradient with respect to each spike train
        for (vector<SpikeTrain*>::iterator it = glm->spike_trains.begin();
             it != glm->spike_trains.end();
             ++it)
        {
            grad += LinearImpulseCurrent::d_ll_d_w(*it, n);
        }

        // Update w_ir
        w_ir[n] += (momentum * grad.array()).matrix();
    }
}

MatrixXd LinearImpulseCurrent::compute_current(SpikeTrain* st)
{
    MatrixXd I_imp(st->T, st->N);

    // Each column of the output matrix is a matrix vector product
    // of the filtered spike train for neuron n and the impulse
    // response weights for that connection.
    for (int n=0; n < st->N; n++)
    {
        I_imp.col(n) = st->filtered_S[n] * w_ir[n];
    }
    return I_imp;
}

MatrixXd LinearImpulseCurrent::compute_current_with_cache(SpikeTrain* st)
{
    MatrixXd I_imp(st->T, st->N);

    // Each column of the output matrix is a matrix vector product
    // of the filtered spike train for neuron n and the impulse
    // response weights for that connection.
    for (int n=0; n < st->N; n++)
    {
        // Check if the current has been cached
        if (!allclose(st->cached_I_imp_key[n], w_ir[n]))
        {
            // Cache miss. Update the cache.
            st->cached_I_imp_key[n] = w_ir[n];
            st->cached_I_imp_val[n] = st->filtered_S[n] * w_ir[n];
        }
        I_imp.col(n) = st->cached_I_imp_val[n];
    }
    return I_imp;
}

void LinearImpulseCurrent::get_w(double* w_buffer)
{
    // Copy the impulse response weights into a buffer
    NPMatrix<double> w(w_buffer, LinearImpulseCurrent::N, LinearImpulseCurrent::D_imp);
    for (int n=0; n<LinearImpulseCurrent::N; n++)
    {
        w.row(n) = w_ir[n];
    }
}

void LinearImpulseCurrent::set_w(double* w_buffer)
{
    // Copy the impulse response weights into a buffer
    NPMatrix<double> w(w_buffer, LinearImpulseCurrent::N, LinearImpulseCurrent::D_imp);
    for (int n=0; n<LinearImpulseCurrent::N; n++)
    {
        w_ir[n] = w.row(n);
    }
}

/**
 *  Impulse HMC Sampler
 */
LinearImpulseCurrent::ImpulseHmcSampler::ImpulseHmcSampler(LinearImpulseCurrent* parent,
                                                           std::default_random_engine rng,
                                                           int n_steps) :
                                                           AdaptiveHmcSampler(rng, n_steps)
{
    this->parent = parent;
}

double LinearImpulseCurrent::ImpulseHmcSampler::logp(MatrixXd x)
{
    // Set the impulse response
    parent->w_ir[n_pre] = x;

    // Prior
    double logp = parent->prior->logp(x);
    // Likelihood
    logp += parent->glm->log_probability();
    return logp;
}

MatrixXd LinearImpulseCurrent::ImpulseHmcSampler::grad(MatrixXd x)
{
    // Set the impulse response
    parent->w_ir[n_pre] = x;

    // Initialize the output
    MatrixXd grad(parent->D_imp, 1);
    parent->prior->grad(x, &grad);

    // Get the gradient with respect to each spike train
    Glm* glm = parent->glm;
    for (vector<SpikeTrain*>::iterator it = glm->spike_trains.begin();
         it != glm->spike_trains.end();
         ++it)
    {
        grad += parent->d_ll_d_w(*it, n_pre);
    }

    return grad;
}

void LinearImpulseCurrent::ImpulseHmcSampler::set_n_pre(int n_pre)
{
    this->n_pre = n_pre;
}

void LinearImpulseCurrent::resample()
{
    // Sample each presynaptic impulse response in turn.
    for (int n_pre=0; n_pre<N; n_pre++)
    {
        // Only sample if there is a synapse from n_pre.
        if (glm->network->A(n_pre))
        {
            // Tell the sampler which impulse to sample.
            sampler->set_n_pre(n_pre);

            // Use the ImpulseHmcSampler to update the bias
            MatrixXd x_next(D_imp,1);

            sampler->sample(w_ir[n_pre], &x_next);

            // Set the new bias
            w_ir[n_pre] = x_next;
        }
        else
        {
            w_ir[n_pre] = prior->sample();
        }
    }
}

/**
 *  Dirichlet distributed impulse response coefficients.
 */
DirichletImpulseCurrent::DirichletImpulseCurrent(int N, int D_imp, Random* random, Dirichlet* prior)
{
    this->N = N;
    this->D_imp = D_imp;
    this->prior = prior;

    // Initialize impulse response weights.
    for (int n=0; n < N; n++)
    {
        g_ir.push_back(prior->sample());
        w_ir.push_back(prior->as_dirichlet(g_ir[n]));
    }

    // Initialize the sampler. The number of steps is set in glm.h
    sampler = new ImpulseHmcSampler(this, random->rng);
}

DirichletImpulseCurrent::~DirichletImpulseCurrent()
{
    if (sampler) { delete sampler; }
}

VectorXd DirichletImpulseCurrent::d_ll_d_g(SpikeTrain* st, int n)
{
    // For the linear impulse response I_imp = filtered_S * w
    // so d_I_imp_d_w = filtered_S
    VectorXd dll_dw = (glm->d_ll_d_I_imp(st, n).transpose() * st->filtered_S[n]);
    VectorXd dll_dg = dll_dw.transpose() * prior->dw_dg(g_ir[n]);
    return dll_dg;
}

void DirichletImpulseCurrent::d_ll_d_g(SpikeTrain* st, int n, double* dg_buffer)
{
    // Copy the impulse response weights into a buffer
    NPVector<double> dg(dg_buffer, DirichletImpulseCurrent::D_imp);
    dg = DirichletImpulseCurrent::d_ll_d_g(st, n);
}

MatrixXd DirichletImpulseCurrent::compute_current(SpikeTrain* st)
{
    MatrixXd I_imp = MatrixXd::Random(st->T, st->N);

    // Each column of the output matrix is a matrix vector product
    // of the filtered spike train for neuron n and the impulse
    // response weights for that connection.
    for (int n=0; n < st->N; n++)
    {
        I_imp.col(n) = st->filtered_S[n] * w_ir[n];
    }
    return I_imp;
}

MatrixXd DirichletImpulseCurrent::compute_current_with_cache(SpikeTrain* st)
{
    MatrixXd I_imp(st->T, st->N);

    // Each column of the output matrix is a matrix vector product
    // of the filtered spike train for neuron n and the impulse
    // response weights for that connection.
    for (int n=0; n < st->N; n++)
    {
        // Check if the current has been cached
        if (!allclose(st->cached_I_imp_key[n], w_ir[n]))
        {
            // Cache miss. Update the cache.
            st->cached_I_imp_key[n] = w_ir[n];
            st->cached_I_imp_val[n] = st->filtered_S[n] * w_ir[n];
        }
        I_imp.col(n) = st->cached_I_imp_val[n];
    }
    return I_imp;
}

void DirichletImpulseCurrent::get_w(double* w_buffer)
{
    // Copy the impulse response weights into a buffer
    NPMatrix<double> w(w_buffer, N, D_imp);
    for (int n=0; n<N; n++)
    {
        w.row(n) = w_ir[n];
    }
}

void DirichletImpulseCurrent::get_g(double* g_buffer)
{
    // Copy the impulse response weights into a buffer
    NPMatrix<double> g(g_buffer, N, D_imp);
    for (int n=0; n < N; n++)
    {
        g.row(n) = g_ir[n];
    }
}

void DirichletImpulseCurrent::set_g(double* g_buffer)
{
    // Copy the impulse response weights into a buffer
    NPMatrix<double> g(g_buffer, N, D_imp);
    for (int n=0; n < N; n++)
    {
        g_ir[n] = g.row(n);
        // Update w accordingly
        w_ir[n] = prior->as_dirichlet(g_ir[n]);
    }
}


/**
 *  Normalized Impulse HMC Sampler
 */
DirichletImpulseCurrent::ImpulseHmcSampler::ImpulseHmcSampler(DirichletImpulseCurrent* parent,
                                                           std::default_random_engine rng,
                                                           int n_steps) :
                                                           AdaptiveHmcSampler(rng, n_steps)
{
    this->parent = parent;
}

double DirichletImpulseCurrent::ImpulseHmcSampler::logp(MatrixXd x)
{
    // Set the impulse response
    parent->g_ir[n_pre] = x;
    parent->w_ir[n_pre] = parent->prior->as_dirichlet(x);

    // Prior
    double logp = parent->prior->logp(x);
    // Likelihood
    logp += parent->glm->log_probability();
    return logp;

    return parent->glm->log_probability();
}

MatrixXd DirichletImpulseCurrent::ImpulseHmcSampler::grad(MatrixXd x)
{
    // Set the impulse response
    parent->g_ir[n_pre] = x;
    parent->w_ir[n_pre] = parent->prior->as_dirichlet(x);

    // Initialize the output
    MatrixXd grad(parent->D_imp, 1);
    parent->prior->grad(x, &grad);

    // Get the gradient with respect to each spike train
    Glm* glm = parent->glm;
    for (vector<SpikeTrain*>::iterator it = glm->spike_trains.begin();
         it != glm->spike_trains.end();
         ++it)
    {
        grad += parent->d_ll_d_g(*it, n_pre);
    }

    return grad;
}

void DirichletImpulseCurrent::ImpulseHmcSampler::set_n_pre(int n_pre)
{
    this->n_pre = n_pre;
}

void DirichletImpulseCurrent::coord_descent_step(double momentum)
{
    // Update each of the impulse response weights
    for (int n=0; n<N; n++)
    {
        MatrixXd grad(D_imp,1);
        prior->grad(g_ir[n], &grad);

        // Get the gradient with respect to each spike train
        for (vector<SpikeTrain*>::iterator it = glm->spike_trains.begin();
             it != glm->spike_trains.end();
             ++it)
        {
            grad += DirichletImpulseCurrent::d_ll_d_g(*it, n);
        }

        // Update w_ir
        g_ir[n] += (momentum * grad.array()).matrix();
        w_ir[n] = prior->as_dirichlet(g_ir[n]);
    }
}

void DirichletImpulseCurrent::resample()
{
    // Sample each presynaptic impulse response in turn.
    for (int n_pre=0; n_pre<N; n_pre++)
    {
        // Only sample if there is a synapse from n_pre.
        if (glm->network->A(n_pre))
        {
            // Tell the sampler which impulse to sample.
            sampler->set_n_pre(n_pre);

            // Use the ImpulseHmcSampler to update the bias
            MatrixXd x_next(D_imp,1);

            sampler->sample(g_ir[n_pre], &x_next);

            // Set the new bias
            g_ir[n_pre] = x_next;
            w_ir[n_pre] = prior->as_dirichlet(g_ir[n_pre]);
        }
        else
        {
            g_ir[n_pre] = prior->sample();
            w_ir[n_pre] = prior->as_dirichlet(g_ir[n_pre]);
        }
    }
}