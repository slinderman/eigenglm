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

    SpikeTrain(int N, int T, double dt, VectorXd S, vector<MatrixXd> filtered_S);
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


//class BiasCurrent : public Component
class BiasCurrent 
{
public:
    double I_bias;

    BiasCurrent(double bias);
    ~BiasCurrent() {}

    double log_probability();
    void resample();
};

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
        SmoothRectLinearLink *link);

    void add_spike_train(SpikeTrain *s);

    double log_likelihood();

    double log_probability();

    double d_ll_d_bias(SpikeTrain* s);

    double d_ll_d_bias(SpikeTrain* s, VectorXd I_stim, VectorXd I_net);

    VectorXd d_ll_d_I_imp(SpikeTrain* s, int n);

    VectorXd d_ll_d_I_imp(SpikeTrain* s, int n, double I_bias, VectorXd I_stim);

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
