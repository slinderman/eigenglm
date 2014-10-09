#include <Eigen/Core>
#include "nptypes.h"

// TODO: Namespace?

namespace glm {
    using namespace nptypes;
    using namespace Eigen;
    using namespace std;

    /**
     * Compute the log likelihood of a Poisson GLM with a soft rectified linear
     * inverse link function.
     */
    template <typename Type>
    Type log_likelihood(int T, Type dt, Type* S, Type I_bias, Type* I_stim, Type* I_net)
    {
        // Cast inputs to Eigen vectors.
        NPVector<Type> np_S(S, T);
        NPVector<Type> np_I_stim(I_stim, T);
        NPVector<Type> np_I_net(I_net, T);

        // Compute the total current.
        Matrix<Type, Dynamic, 1> I(T);
        I = I_bias + np_I_stim.array() + np_I_net.array();

        // Compute the firing rate and its log.
        Matrix<Type, Dynamic, 1> lam = (1.0 + I.array().exp()).log();
        Matrix<Type, Dynamic, 1> loglam = lam.array().log();

        // Compute the log probability.
        return -dt * lam.sum() + np_S.dot(loglam);
    }

    /**
     * Compute the log likelihood of a Poisson GLM with a soft rectified linear
     * inverse link function.
     */
    template <typename Type>
    void compute_I_net(int T, int D_imp, Type* ir, Type* w_ir, Type* I_net)
    {
        // Cast inputs to Eigen vectors.
        NPMatrix<Type> np_ir(ir, T, D_imp);
        NPVector<Type> np_w_ir(w_ir, D_imp);
        NPVector<Type> np_I_net(I_net, T);

        np_I_net = np_ir * np_w_ir;
    }
}

// Dummy class to expose the template functions.
// This is required because Cython only supports template classes.
template <typename Type>
class NPGlm
{
    public:
    static Type log_likelihood(int T, Type dt, Type* S, Type I_bias, Type* I_stim, Type* I_net)
    { return glm::log_likelihood(T, dt, S, I_bias, I_stim, I_net); }

    static void compute_I_net(int T, int D_imp, Type* ir, Type* w_ir, Type* I_net)
    { glm::compute_I_net(T, D_imp, ir, w_ir, I_net); }

};