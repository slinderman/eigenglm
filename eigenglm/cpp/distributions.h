#ifndef __DISTRIBUTIONS_H_INCLUDED__
#define __DISTRIBUTIONS_H_INCLUDED__

/**
 *  Wrapper for some distribution objects that will be used frequently.
 */

#include <Eigen/Dense>
#include <random>
#include <math.h>
#include "nptypes.h"

inline int sign(double x) { return (x < 0.0 ? -1 : 1); }

using namespace Eigen;
using namespace nptypes;

class Distribution
{
protected:
    std::default_random_engine rng;
public:
    Distribution() {}

    Distribution(std::default_random_engine rng)
    {
        this->rng = rng;
    }

    virtual ~Distribution() {}

    virtual double logp(MatrixXd x) = 0;
    virtual void grad(MatrixXd x, MatrixXd* dx) = 0;
    virtual MatrixXd sample() = 0;
};

class IndependentBernoulli : public Distribution
{
    int D;
    VectorXd rho;
    VectorXd logrho;
    VectorXd lognotrho;
    std::uniform_real_distribution<double> uniform;

private:
    void initialize(VectorXd rho)
    {
        this->rho = rho;
        this->logrho = rho.array().log();
        this->lognotrho = (1.0-rho.array()).log();
        this->D = rho.size();
        uniform = std::uniform_real_distribution<double>(0.0,1.0);
    }
public:
    IndependentBernoulli()
    {
        rho = VectorXd::Constant(this->D, 0.5);
        initialize(rho);
    }

    IndependentBernoulli(VectorXd rho,
                         std::default_random_engine rng) :
                         Distribution(rng)
    {
        initialize(rho);
    }

    IndependentBernoulli(double rho,
                     std::default_random_engine rng) :
                     Distribution(rng)
    {
        VectorXd rhov = VectorXd::Constant(1, rho);
        initialize(rhov);
    }

    ~IndependentBernoulli() {}

    double logp(MatrixXd x)
    {
        // TODO: input checking?
        return (x.array() * logrho.array()).sum() +
               ((1.0-x.array()) * lognotrho.array()).sum();
    }

    double logp(double* x_buffer)
    {
        NPVector<double> x_np(x_buffer, D);
        VectorXd x = x_np;
        return logp(x);
    }

    void grad(MatrixXd x, MatrixXd* dx)
    {
        *dx = VectorXd::Zero(D);
    }

    void grad(double* x_buffer, double* dx_buffer)
    {
        NPVector<double> dx_np(dx_buffer, D, 1);
//        dx_np.array() = 0;
    }

    MatrixXd sample()
    {
        VectorXd x(D);
        for (int d=0; d<D; d++)
        {
            x(d) = (uniform(rng) < rho(d));
        }
        return x;
    }
};



/**
 *  Diagonal Gaussian distribution, but not necessarily spherical.
 */
class DiagonalGuassian : public Distribution
{
    int D;
    VectorXd mu;
    VectorXd sigma;
    std::normal_distribution<double> normal;

public:
    DiagonalGuassian()
    {
        this->D = 1;
        mu = VectorXd::Zero(D);
        sigma = VectorXd::Ones(D);

        // Initialize random number generator
    //        std::default_random_engine rng;
    //        Distribution::Distribution(rng);
    //        this->rng = rng;

        this->normal = std::normal_distribution<double>(0.0, 1.0);
    }

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

    DiagonalGuassian(double mu,
                     double sigma,
                     std::default_random_engine rng) :
                     Distribution(rng)
    {
        this->mu = VectorXd::Constant(1, mu);
        this->sigma = VectorXd::Constant(1, sigma);
        this->D = 1;
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

    double logp(double* x_buffer)
    {
        NPMatrix<double> x_np(x_buffer, D, 1);
        MatrixXd x = x_np;
        return DiagonalGuassian::logp(x);
    }

    void grad(MatrixXd x, MatrixXd* dx)
    {
        *dx = -(x-mu).array()/sigma.array();
    }

    void grad(double* x_buffer, double* dx_buffer)
    {
        NPMatrix<double> x_np(x_buffer, D, 1);
        NPMatrix<double> dx_np(dx_buffer, D, 1);
        MatrixXd x = x_np;
        MatrixXd dx(x.rows(), x.cols());
        DiagonalGuassian::grad(x, &dx);
        dx_np = dx;
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
    Dirichlet(int D)
    {
        this->D = D;
        this->alpha = VectorXd::Constant(this->D, 0.1);

        for (int d=0; d < this->D; d++)
        {
            this->gammas.push_back(std::gamma_distribution<double>(alpha(d), 1.0));
        }
    }

    Dirichlet(VectorXd alpha,
              std::default_random_engine rng) :
              Distribution(rng)
    {
        this->alpha = alpha;
        this->D = alpha.size();

        for (int d=0; d < this->D; d++)
        {
            this->gammas.push_back(std::gamma_distribution<double>(alpha(d), 1.0));
        }
    }

    ~Dirichlet() {}

    double logp(MatrixXd x)
    {
        // sum_d (alpha_d -1)*log(abs(x_d)) - sum(abs(x_d))
        return ((alpha.array() - 1.) * x.array().abs().log()).sum()
               - x.array().abs().sum();
    }

    double logp(double* x_buffer)
    {
        NPVector<double> x_np(x_buffer, D);
        MatrixXd x = x_np;
        return Dirichlet::logp(x);
    }

    void grad(MatrixXd x, MatrixXd* dx)
    {
        // dlp/dx_d = sign(x_d) * (alpha_d-1) / abs(x_d) - sign(x_d)
        //          = sign(g_d) * ((alpha_d - 1) / abs(g_d) - 1)
        //          = (alpha_d-1)/x_d - sign(x_d)
        ArrayXd xsign = -1.0 + 2.0*(x.array() > 0).cast<double>();
        *dx = (alpha.array()-1)/x.array() - xsign;
//        *dx = xsign * ((alpha.array()-1)/x.array().abs() - 1.);
    }

    void grad(double* x_buffer, double* dx_buffer)
    {
        NPMatrix<double> x_np(x_buffer, D, 1);
        NPMatrix<double> dx_np(dx_buffer, D, 1);
        MatrixXd x = x_np;
        MatrixXd dx(x.rows(), x.cols());
        Dirichlet::grad(x, &dx);
        dx_np = dx;
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

    VectorXd as_dirichlet(VectorXd x)
    {
        VectorXd p(D);
        double Z = x.array().abs().sum();

        for (int d=0; d<D; d++)
        {
            p(d) = fabs(x(d)) / Z;
        }
        return p;
    }

    void as_dirichlet(double* g_buffer, double* w_buffer)
    {
        NPVector<double> g_np(g_buffer, D);
        NPVector<double> w_np(w_buffer, D);
        VectorXd g = g_np;
        VectorXd w = Dirichlet::as_dirichlet(g);
        w_np = w;
    }

    MatrixXd dw_dg(VectorXd g)
    {
        // w_d = |g_d| / \sum_{d'} |g_d'|
        // d_wd/d_gd = sign(g_d) * sum_{d'\neq d} |g_d'| / Z**2

        // d_wd/d_gd' = -|g|*sign(g_d') / Z**2
        MatrixXd dwdg(D,D);
        double Z = g.array().abs().sum();

        for (int d1=0; d1<D; d1++)
        {
            for (int d2=0; d2<D; d2++)
            {
                if (d1==d2)
                {
                    dwdg(d1,d2) = sign(g(d1)) * (Z - fabs(g(d1))) / (Z*Z);
                }
                else
                {
                    dwdg(d1,d2) = - fabs(g(d1)) * sign(g(d2)) / (Z*Z);
                }
            }
        }
        return dwdg;
    }

    void dw_dg(double* g_buffer, double* dwdg_buffer)
    {
        NPVector<double> g_np(g_buffer, D);
        NPMatrix<double> dwdg_np(dwdg_buffer, D, D);
        VectorXd g = g_np;
        MatrixXd dwdg = Dirichlet::dw_dg(g);
        dwdg_np = dwdg;
    }
};


#endif