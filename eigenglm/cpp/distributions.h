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

/**
 *  A silly little wrapper to share the random number generator among components
 */
class Random
{
public:
    std::default_random_engine rng;

    Random(int seed=0)
    {
        // Initialize random number generator
        this->rng = std::default_random_engine(seed);
    }
};


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
    IndependentBernoulli(double* rho_buffer,
                         int D,
                         Random* random) :
                         Distribution(random->rng)
    {
        NPVector<double> rho(rho_buffer, D);
        initialize(rho);
    }

    IndependentBernoulli(double rho,
                         Random* random) :
                         Distribution(random->rng)
    {
        VectorXd rhov = VectorXd::Constant(1, rho);
        initialize(rhov);
    }

    ~IndependentBernoulli() {}

    // Getters and setters
    void get_rho(double* buffer)
    {
        // Copy the parameter into a buffer
        NPVector<double> b(buffer, D);
        b = rho;
    }

    void set_rho(double* buffer)
    {
        // Copy the parameter into a buffer
        NPVector<double> b(buffer, D);
        rho = b;

        // Update helper variables
        this->logrho = rho.array().log();
        this->lognotrho = (1.0-rho.array()).log();
    }

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
class DiagonalGaussian : public Distribution
{
protected:
    int D;
    VectorXd mu;
    VectorXd sigma;
    std::normal_distribution<double> normal;

    void initialize(VectorXd mu,
                    VectorXd sigma,
                    std::default_random_engine rng)
    {
        this->D = mu.size();
        this->mu = mu;
        this->sigma = sigma;
        this->normal = std::normal_distribution<double>(0.0, 1.0);
    }

public:
    DiagonalGaussian(double* mu_buffer,
                     double* sigma_buffer,
                     int D,
                     Random* random) :
                     Distribution(random->rng)
    {
        NPVector<double> mu(mu_buffer, D);
        NPVector<double> sigma(sigma_buffer, D);
        initialize(mu, sigma, random->rng);
    }

    DiagonalGaussian(double mu,
                     double sigma,
                     std::default_random_engine rng) :
                     Distribution(rng)
    {
        VectorXd mu_vec = VectorXd::Constant(1, mu);
        VectorXd sigma_vec = VectorXd::Constant(1, sigma);
        initialize(mu_vec, sigma_vec, rng);
    }

    ~DiagonalGaussian() {}

    // Getters and setters
    void get_mu(double* buffer)
    {
        // Copy the parameter into a buffer
        NPVector<double> b(buffer, D);
        b = mu;
    }

    void set_mu(double* buffer)
    {
        // Copy the parameter into a buffer
        NPVector<double> b(buffer, D);
        mu = b;
    }

    void get_sigma(double* buffer)
    {
        // Copy the parameter into a buffer
        NPVector<double> b(buffer, D);
        b = sigma;
    }

    void set_sigma(double* buffer)
    {
        // Copy the parameter into a buffer
        NPVector<double> b(buffer, D);
        sigma = b;
    }

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
        return DiagonalGaussian::logp(x);
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
        DiagonalGaussian::grad(x, &dx);
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


/**
 *  Scalar Gaussian subclass of diagonal Gaussian
 */
//class ScalarGaussian : public DiagonalGaussian
//{
//public:
//    ScalarGaussian(double mu,
//                   double sigma,
//                   std::default_random_engine rng) :
//                   Distribution(rng)
//    {
//        VectorXd mu_vec = VectorXd::Constant(1, mu);
//        VectorXd sigma_vec = VectorXd::Constant(1, sigma);
//        initialize(mu_vec, sigma_vec, rng);
//    }
//
//    ~ScalarGaussian() {}
//
//
//};

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

    Dirichlet(double* alpha_buffer,
              int D,
              Random* random) :
              Distribution(random->rng)
    {
        this->D = D;
        NPVector<double> alpha(alpha_buffer, D);
        this->alpha = alpha;

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

    // Getters and setters
    void get_alpha(double* buffer)
    {
        // Copy the parameter into a buffer
        NPVector<double> b(buffer, D);
        b = alpha;
    }

    void set_alpha(double* buffer)
    {
        // Copy the parameter into a buffer
        NPVector<double> b(buffer, D);
        alpha = b;
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

    void sample(double* buffer)
    {
        NPVector<double> s(buffer, D);
        s = sample();
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