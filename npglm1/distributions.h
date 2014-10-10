#ifndef __DISTRIBUTIONS_H_INCLUDED__
#define __DISTRIBUTIONS_H_INCLUDED__

/**
 *  Wrapper for some distribution objects that will be used frequently.
 */

#include <Eigen/Dense>
#include <random>
#include <math.h>

using namespace Eigen;

class Distribution
{
protected:
    std::default_random_engine rng;
public:
    Distribution(std::default_random_engine rng)
    {
        this->rng = rng;
    }

    virtual ~Distribution() {}

    virtual double logp(MatrixXd x) = 0;
    virtual void grad(MatrixXd x, MatrixXd* dx) = 0;
    virtual MatrixXd sample() = 0;
};

class DiagonalGuassian : public Distribution
{
    int D;
    VectorXd mu;
    VectorXd sigma;
    std::normal_distribution<double> normal;

public:
    DiagonalGuassian(VectorXd mu,
                     VectorXd sigma,
                     std::default_random_engine rng) :
                     Distribution(rng)
    {
        this->mu = mu;
        this->sigma = sigma;
        this->D = mu.size();
        this->normal = std::normal_distribution<double>(0.0, 1.0);
    }

    ~DiagonalGuassian() {}

    double logp(MatrixXd x)
    {
        // Calculate the normalization constant
        double Z = -0.5*x.size() * log(2 * M_PI) - sigma.array().log().sum();
        // Calculate the exponent
        return Z + -0.5*(((x-mu).array()/sigma.array()).pow(2)).sum();
    }

    void grad(MatrixXd x, MatrixXd* dx)
    {
        *dx = -(x-mu).array()/sigma.array();
    }

    MatrixXd sample()
    {
        VectorXd x(D);
        for (int d=0; d<D; d++)
        {
            x(d) = mu(d) + sigma(d) * normal(rng);
        }
        return x;
    }
};



#endif