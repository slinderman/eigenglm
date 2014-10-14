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


/*
 *  Dirichlet distribution parameterized by gamma r.v.'s. To unconstrain the parameters,
 *  we use real valued r.v.'s and make sure their absolute value to gamma distributed.
 *
 *  x = [g_1, ..., g_D]
 *  |g_d| ~ Gamma(\alpha_d, 1)
 *   p    ~ Dirichlet(|g_1| / \sum{|g_d|}, ... , |g_D| / \sum{|g_d|})
 */
class Dirichlet : public Distribution
{
    int D;
    VectorXd alpha;
    std::vector<std::gamma_distribution<double>> gammas;

public:
    Dirichlet(VectorXd alpha,
              std::default_random_engine rng) :
              Distribution(rng)
    {
        this->alpha = alpha;
        this->D = alpha.size();

        for (int d=0; d < this->D; d++)
        {
            this->gammas.push_back(std::gamma_distribution<double>(alpha(d)));
        }
    }

    ~Dirichlet() {}

    double logp(MatrixXd x)
    {
        // (self.alpha-1.0) * T.sum(T.log(abs(g))) - T.sum(abs(g))
        return ((alpha.array() - 1.) * x.array().abs().log()).sum() -
               x.array().abs().sum();
    }

    void grad(MatrixXd x, MatrixXd* dx)
    {
        // dlp/dx_d = sign(g_d) * (alpha_d - 1)/ |g_d| - sign(g_d)
        //          = sign(g_d) * ((alpha_d - 1)/ |g_d| - 1)
        ArrayXd xsign = -1.0 + 2.0*(x.array() > 0).cast<double>();
        *dx = xsign * ((alpha.array()-1)/x.array().abs() - 1.);
    }

    MatrixXd sample()
    {
        VectorXd x(D);
        for (int d=0; d<D; d++)
        {
            x(d) = gammas[d](rng);
        }
        return x;
    }

    MatrixXd as_dirichlet(MatrixXd x)
    {
        MatrixXd p(x.rows(), x.cols());
        double Z = x.array().abs().sum();

        for (int i=0; i<x.rows(); i++)
        {
            for (int j=0; j<x.cols(); j++)
            {
                p(i,j) = abs(x(i,j)) / Z;
            }
        }
        return p;
    }
};


#endif