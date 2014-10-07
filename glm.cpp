#include <iostream>
#include <Eigen/Dense>
#include <vector>
using namespace Eigen;
using namespace std;

class SpikeTrain
{
public:
    int N;
    int T;
    double dt;
    VectorXd S;

    //TODO: Stimuli

    // Filtered spike train.
    vector<MatrixXd> filtered_S;

    SpikeTrain(int N, int T, double dt, VectorXd S, vector<MatrixXd> filtered_S)
    {
        SpikeTrain::N = N;
        SpikeTrain::T = T;
        SpikeTrain::dt = dt;
        SpikeTrain::S = S;
        SpikeTrain::filtered_S = filtered_S;
    }
};

class Component
{
public:
    //  Abstract base class for all components
    //
    virtual double log_probability() = 0;

    // Resample the parameters of this component
    virtual void resample() = 0;

};


class BiasCurrent : public Component
{
public:
    double I_bias;

    BiasCurrent(double bias)
    {
        I_bias = bias;
    }

    double log_probability()
    {
        return 0.0;
    }

    void resample()
    {
    }
};

class LinearImpulseCurrent : public Component
{
public:
    // A vector of impulse response weights for each presynaptic neuron.
    vector<VectorXd> w_ir;

    LinearImpulseCurrent(int N, int B)
    {
        // Initialize impulse response weights.
        for (int n=0; n < N; n++)
        {
            w_ir.push_back(VectorXd::Random(B));
        }
    }

    MatrixXd compute_current(SpikeTrain* st)
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

    double log_probability()
    {
        return 0.0;
    }

    void resample()
    {
    }
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
    VectorXd compute_firing_rate(VectorXd I)
    {
        return (1.0 + I.array().exp()).log();
    }

    VectorXd d_firing_rate_d_I(VectorXd I)
    {
        // Gradient of the firing rate with respect to I
        return I.array().exp() / (1.0 + I.array().exp());
    }

    double log_probability()
    {
        return 0.0;
    }

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

    Glm(BiasCurrent *bias,
        LinearImpulseCurrent *impulse,
        VectorXd A,
        VectorXd W,
        SmoothRectLinearLink *link)
    {
        Glm::bias = bias;
        Glm::impulse = impulse;
        Glm::nlin = link;
        Glm::A = A;
        Glm::W = W;

    }

    void add_spike_train(SpikeTrain *s)
    {
        spike_trains.push_back(s);
    }

    double log_likelihood()
    {
        double ll = 0;
        for (vector<SpikeTrain*>::iterator s = spike_trains.begin();
             s != spike_trains.end();
             ++s)
        {
            // Compute the total current for this spike train.
            VectorXd I = VectorXd::Constant((*s)->T, 0.0);

            // Bias is a constant.
            I = I.array() + bias->I_bias;

            // Add the weighted impulse responses
            MatrixXd I_imp = impulse->compute_current(*s);
            I += I_imp * (A.array() * W.array()).matrix();

            // Compute the firing rate and its log.
            VectorXd lam = nlin->compute_firing_rate(I);
            VectorXd loglam = lam.array().log();

            // Compute the Poisson likelihood.
            ll += -(*s)->dt * lam.sum() + (*s)->S.dot(loglam);

        }
        return ll;
    }

    double log_probability()
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

    void resample()
    {
        // Call subcomponent resample methods.
        bias->resample();
        nlin->resample();
    }

    double d_ll_d_bias(SpikeTrain* s)
    {
        // Overloaded function if we don't have I_stim or I_net precomputed
        MatrixXd I_imp = impulse->compute_current(s);
        VectorXd I_net = I_imp * (A.array() * W.array()).matrix();
        VectorXd I_stim = VectorXd::Constant(I_net.size(), 0);

        return d_ll_d_bias(s, I_stim, I_net);
    }

    double d_ll_d_bias(SpikeTrain* s, VectorXd I_stim, VectorXd I_net)
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
        VectorXd d_ll_d_lam = VectorXd::Constant(s->T, -s->dt).array() + s->S.array()/lam.array();
        VectorXd d_lam_d_I = nlin->d_firing_rate_d_I(I);
        double d_I_d_bias = 1.0;

        // Multiply em up!
        double d_ll_d_bias = d_ll_d_lam.dot(d_lam_d_I) * d_I_d_bias;

        return d_ll_d_bias;
    }

    VectorXd d_ll_d_I_imp(SpikeTrain* s, double I_bias, VectorXd I_stim)
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
        double d_I_d_bias = 1.0;

        // Multiply em up!
        double d_ll_d_bias = d_ll_d_lam.dot(d_lam_d_I) * d_I_d_bias;

        return d_ll_d_bias;
    }

};

class Population : Component
{
public:
    vector<Glm*> glms;

    void add_glm(Glm *g)
    {
        glms.push_back(g);
    }

    double log_probability()
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

    void resample()
    {
    }

};


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

        SpikeTrain *s = new SpikeTrain(N, T, dt, S, filtered_S);
        spike_trains.push_back(s);
    }

    // Make a network
    MatrixXd A = MatrixXd::Constant(N,N,1);
    MatrixXd W = MatrixXd::Constant(N,N,1);

    // Make a population of GLMs
    Population population = Population();

    for (int n=0; n<N; n++)
    {
        BiasCurrent *bias = new BiasCurrent(1.0);
        LinearImpulseCurrent *impulse = new LinearImpulseCurrent(N,B);
        SmoothRectLinearLink *link = new SmoothRectLinearLink();
        Glm *g = new Glm(bias, impulse, A.col(n), W.col(n), link);

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

    // Cleanup?
//    delete spike_trains;
//    delete bias;
//    delete link;
//    delete g;
}

