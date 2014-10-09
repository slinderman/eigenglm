#include <iostream>
#include <Eigen/Dense>
#include <vector>

#include "glm.h"

using namespace Eigen;
using namespace std;

SpikeTrain::SpikeTrain(int N, int T, double dt, VectorXd S, int D_imp, vector<MatrixXd> filtered_S)
{
    SpikeTrain::N = N;
    SpikeTrain::T = T;
    SpikeTrain::dt = dt;
    SpikeTrain::S = S;

    SpikeTrain::D_imp = D_imp;
    SpikeTrain::filtered_S = filtered_S;
}


BiasCurrent::BiasCurrent(Glm* glm, double bias)
{
    parent = glm;
    I_bias = bias;
}

double BiasCurrent::log_probability()
{
    return 0.0;
}

void BiasCurrent::coord_descent_step(double momentum)
{
    double grad = 0;

    // Get the gradient with respect to each spike train
    for (vector<SpikeTrain*>::iterator s = parent->spike_trains.begin();
         s != parent->spike_trains.end();
         ++s)
    {
        grad += parent->d_ll_d_bias(*s);
    }

    // Update bias
    I_bias += momentum * grad;
}

void BiasCurrent::resample() {}

LinearImpulseCurrent::LinearImpulseCurrent(int N, int B)
{
    // Initialize impulse response weights.
    for (int n=0; n < N; n++)
    {
        w_ir.push_back(VectorXd::Random(B));
    }
}

MatrixXd LinearImpulseCurrent::compute_current(SpikeTrain* st)
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


VectorXd SmoothRectLinearLink::compute_firing_rate(VectorXd I)
{
    return (1.0 + I.array().exp()).log();
}

VectorXd SmoothRectLinearLink::d_firing_rate_d_I(VectorXd I)
{
    // Gradient of the firing rate with respect to I
    return I.array().exp() / (1.0 + I.array().exp());
}

Glm::Glm(int N, int D_imp)
{
    // Standard GLM
    Glm::bias = new BiasCurrent(this, 1.0);
    Glm::impulse = new LinearImpulseCurrent(N, D_imp);
    Glm::nlin = new SmoothRectLinearLink();

    // Make a network
    Glm::A = VectorXd::Ones(N);
    Glm::W = VectorXd::Ones(N);
}

void Glm::add_spike_train(SpikeTrain *s)
{
    spike_trains.push_back(s);
}

void Glm::firing_rate(SpikeTrain* s, VectorXd *fr)
{
//    fr.array() *= 0;

    // Compute the total current for this spike train.
    VectorXd I = VectorXd::Constant(s->T, 0.0);

    // Bias is a constant.
    I = I.array() + bias->I_bias;

    // Add the weighted impulse responses
    MatrixXd I_imp = impulse->compute_current(s);
    I += I_imp * (A.array() * W.array()).matrix();

    // Compute the firing rate and its log.
    *fr = nlin->compute_firing_rate(I);
}

double Glm::log_likelihood()
{
    double ll = 0;
    for (vector<SpikeTrain*>::iterator s = spike_trains.begin();
         s != spike_trains.end();
         ++s)
    {
//        // Compute the total current for this spike train.
//        VectorXd I = VectorXd::Constant((*s)->T, 0.0);
//
//        // Bias is a constant.
//        I = I.array() + bias->I_bias;
//
//        // Add the weighted impulse responses
//        MatrixXd I_imp = impulse->compute_current(*s);
//        I += I_imp * (A.array() * W.array()).matrix();
//
//        // Compute the firing rate and its log.
//        VectorXd lam = nlin->compute_firing_rate(I);
//        VectorXd lam = VectorXd::Constant((*s)->T, 0);
        VectorXd lam;
        Glm::firing_rate(*s, &lam);
        VectorXd loglam = lam.array().log();

        // Compute the Poisson likelihood.
        ll += -1 * (*s)->dt * lam.sum() + (*s)->S.dot(loglam);

    }
    return ll;
}

double Glm::log_probability()
{
    double lp = 0.0;

    // Add subcomponent priors
    lp += bias->log_probability();
    lp += impulse->log_probability();
    lp += nlin->log_probability();

    // Add the likelihood.
    lp += log_likelihood();

    return lp;
}

void Glm::coord_descent_step(double momentum)
{
    // Call subcomponent resample methods.
    bias->coord_descent_step(momentum);
//    impulse->coord_descent_step();
//    nlin->coord_descent_step();
}

void Glm::resample()
{
    // Call subcomponent resample methods.
    bias->resample();
    impulse->resample();
    nlin->resample();
}

double Glm::d_ll_d_bias(SpikeTrain* s)
{
    // Overloaded function if we don't have I_stim or I_net precomputed
    MatrixXd I_imp = impulse->compute_current(s);
    VectorXd I_net = I_imp * (A.array() * W.array()).matrix();
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
    VectorXd I = VectorXd(I_bias + I_stim.array());

    // Bias is a constant.
    I = I.array() + bias->I_bias;

    // Compute the firing rate and its log.
    VectorXd lam = nlin->compute_firing_rate(I);
    VectorXd loglam = lam.array().log();

    // Now compute the gradients.
    // TODO: Avoid divide by zero
    VectorXd d_ll_d_lam = VectorXd::Constant(s->T, -s->dt).array() + s->S.array()/lam.array();
    VectorXd d_lam_d_I = nlin->d_firing_rate_d_I(I);
    double d_I_d_I_imp_n = A(n)  * W(n);

    // Multiply em up!
    VectorXd d_ll_d_I_imp_n = d_ll_d_lam.array() * d_lam_d_I.array() * d_I_d_I_imp_n;

    return d_ll_d_I_imp_n;
}


void Population::add_glm(Glm *g)
{
    glms.push_back(g);
}

double Population::log_probability()
{
    double lp = 0.0;
    for (vector<Glm*>::iterator g = glms.begin();
         g != glms.end();
         ++g)
    {
        lp += (*g)->log_probability();
    }
    return lp;
}

void Population::resample()
{
}



int main()
{
    // Make some fake data sets.
    vector<SpikeTrain*> spike_trains;
    int N = 2;
    int B = 5;
    for (int m=0; m < 10; m++)
    {
        int T = 10;
        double dt = 0.1;
        VectorXd S = (1 + VectorXd::Random(T).array()) * 10;

        vector<MatrixXd> filtered_S;
        for (int n=0; n<N; n++)
        {
            filtered_S.push_back(MatrixXd::Random(T,B));
        }

        SpikeTrain *s = new SpikeTrain(N, T, dt, S, B, filtered_S);
        spike_trains.push_back(s);
    }

    // Make a network
    MatrixXd A = MatrixXd::Constant(N,N,1);
    MatrixXd W = MatrixXd::Constant(N,N,1);

    // Make a population of GLMs
    Population population = Population();

    for (int n=0; n<N; n++)
    {
        Glm *g = new Glm(N, B);

        // Add data
        for (vector<SpikeTrain*>::iterator s = spike_trains.begin();
                 s != spike_trains.end();
                 ++s)
        {
            g->add_spike_train(*s);
        }

        // Add GLM to population.
        population.add_glm(g);
    }



    // Compute log likelihood
//    cout << "ll = " << g->log_likelihood() << endl;
    cout << "lp = " << population.log_probability() << endl;

//    cout << "dll_dbias = " << population.glms[0]->d_ll_d_bias(spike_trains[0]) << endl;
//    cout << "dll_dI_imp = " << population.glms[0]->d_ll_d_I_imp(spike_trains[0], 0) << endl;

    // Cleanup?
//    delete spike_trains;
//    delete bias;
//    delete link;
//    delete g;
}

