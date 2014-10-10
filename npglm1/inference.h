#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <math.h>

using namespace Eigen;

class AdaptiveHmcSampler
{

public:
    double step_sz;
    int n_steps;
    bool adaptive;
    double avg_accept_rate;
    double tgt_accept_rate;
    double avg_accept_time_const;
    double min_step_sz;
    double max_step_sz;

    std::default_random_engine rng;
    std::normal_distribution<double> normal;
    std::uniform_real_distribution<double> uniform;

    // TODO: Template this class
    AdaptiveHmcSampler(std::default_random_engine rng)
    {
        // Initialize random number generators
        // This needs to be done up front. If we create them fresh
        // each time sample is called we won't get any stochasticity.
        this->rng = rng;
        normal = std::normal_distribution<double>(0.0, 1.0);
        uniform = std::uniform_real_distribution<double>(0.0,1.0);

        // TODO: Initialize with constructor
        step_sz = 0.1;
        n_steps = 10;

        // Adaptive step size parameters
        adaptive = true;
        tgt_accept_rate = 0.9;
        avg_accept_time_const = 0.9;
        avg_accept_rate = tgt_accept_rate;
        min_step_sz = 0.001;
        max_step_sz = 1.0;
    }

    // Base classes must override these functions
    virtual double logp(MatrixXd x) = 0;
    virtual MatrixXd grad(MatrixXd x) = 0;

    // Generic sampling method
    void sample(MatrixXd q_curr, MatrixXd* q_next)
    {
        // Start at current state
        MatrixXd q;
        q = q_curr;

        // Momentum is simplest for a standard normal rv
        MatrixXd p(q.rows(), q.cols());
        for (int i=0; i<q.rows(); i++)
        {
            for (int j=0; j<q.cols(); j++)
            {
                p(i,j) = normal(rng);
            }
        }

        // Set a prefactor of -1 since we are using log probs instead of neg log probs
        double pre = -1.0;

        // Evaluate potential and kinetic energies at start of trajectory
        double U_curr = pre * logp(q_curr);
        double K_curr = p.array().pow(2).sum()/2.0;

        // Make a half step in the momentum variable
        p -= step_sz * pre * grad(q)/2.0;

        // Alternate full steps for position and momentum
        for (int i=0; i<n_steps; i++)
        {
            q += step_sz*p;

            // Full step for momentum except for last iteration
            if (i < n_steps-1)
                p -= step_sz * pre * grad(q);
            else
                p -= step_sz * pre * grad(q)/2.0;
        }

        // Negate the momentum at the end of the trajectory to make proposal symmetric?
        p = -p;

        // Evaluate potential and kinetic energies at end of trajectory
        double U_prop = pre * logp(q);
        double K_prop = p.array().pow(2).sum()/2.0;

        // Accept or reject new state with probability proportional to change in energy.
        // Ideally this will be nearly 0, but forward Euler integration introduced errors.
        // Exponentiate a value near zero and get nearly 100% chance of acceptance.
        bool accept = log(uniform(rng)) < U_curr-U_prop + K_curr-K_prop;
        if (accept)
            *q_next = q;
        else
            *q_next = q_curr;


        // Do adaptive step size updates
        if (adaptive)
        {
            avg_accept_rate = avg_accept_time_const * avg_accept_rate +
                              (1.0-avg_accept_time_const) * accept;
            if (avg_accept_rate > tgt_accept_rate)
                step_sz *= 1.02;
            else
                step_sz *= 0.98;

            if (step_sz > max_step_sz)
                step_sz = max_step_sz;
            if (step_sz < min_step_sz)
                step_sz = min_step_sz;
        }

    }
};

