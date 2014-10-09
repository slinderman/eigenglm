#include <iostream>
#include <vector>

#include "nptypes.h"
#include "glm.h"

// Wrapper for the EigenGlm that takes numpy maps
using namespace nptypes;
using namespace std;

class NpSpikeTrain
{
    SpikeTrain* st;
public:
    NpSpikeTrain(int N, int T, double dt, double* S_buffer, int D_imp, vector<double*> filtered_S_buffers)
    {
        NPVector<double> S(S_buffer, T);

        vector<MatrixXd> filtered_S;
        for (int n=0; n<N; n++)
        {
            // TODO: Get filtered_S size
            NPMatrix<double> fS(filtered_S_buffers[n], T, D_imp);
            filtered_S.push_back(fS);
        }

        st = new SpikeTrain(N, T, dt, S, D_imp, filtered_S);
    }

    SpikeTrain* get_spike_train()
    {
        return st;
    }
};

/*
class Component
{
public:
    //  Abstract base class for all components
    //
    virtual double log_probability() = 0;

    // Resample the parameters of this component
    virtual void resample() = 0;

};
*/
//
//class NpBiasCurrent
//{
//public:
//    BiasCurrent* biasCurrent;
//
//    NpBiasCurrent(double bias)
//    {
//        biasCurrent = new BiasCurrent(bias);
//    }
//
//    ~NpBiasCurrent()
//    {
//        if (biasCurrent)
//        {
//            delete biasCurrent;
//        }
//    }
//
//    double log_probability()
//    {
//        return biasCurrent->log_probability();
//    }
//
//    void resample()
//    {
//        biasCurrent->resample();
//    }
//
//};

/*
class LinearImpulseCurrent : public Component
{
public:
    // A vector of impulse response weights for each presynaptic neuron.
    vector<VectorXd> w_ir;

    LinearImpulseCurrent(int N, int B);

    MatrixXd compute_current(SpikeTrain* st);

    double log_probability();
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

    double log_probability();
    void resample();
};
*/
class NpGlm
{
public:
    Glm* glm;

    NpGlm(int N, int D_imp)
    {
        glm = new Glm(N, D_imp);
    }

    void add_spike_train(NpSpikeTrain *s)
    {
        glm->add_spike_train(s->get_spike_train());
    }

    double log_likelihood()
    {
        return glm->log_likelihood();
    }

    double log_probability()
    {
        return glm->log_probability();
    }

    void coord_descent_step(double momentum)
    {
        glm->coord_descent_step(momentum);
    }

    void get_firing_rate(NpSpikeTrain* s, double* fr)
    {
        NPVector<double> np_fr(fr, s->get_spike_train()->T);
        VectorXd vec_fr;
        glm->firing_rate(s->get_spike_train(), &vec_fr);

        // Copy the result back to fr
        np_fr = vec_fr;
    }


    double get_bias()
    {
        return glm->bias->I_bias;
    }

};

/*
class Population : Component
{
public:
    vector<Glm*> glms;

    void add_glm(Glm *g);

    double log_probability();
    void resample();
};
*/
