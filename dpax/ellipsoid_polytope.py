import os
os.environ["JAX_PLATFORM_NAME"] = "cpu" 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import jax
import numpy as np
import ecos
from scipy.sparse import csc_matrix
# from jax.scipy.sparse import csc_matrix

from jax import grad 
from jax import jit 
from jax import custom_jvp
from jax import jvp 
import jax.numpy as jnp 
from jax.test_util import check_grads
from jax.scipy.spatial.transform import Rotation as R
import dpax
from dpax.mrp import dcm_from_mrp
from dpax.polytope import polytope_problem_matrices,create_rect_prism
from dpax.ellipsoid import ellipsoid_problem_matrices



## naming follows the julia code

## since current Conic programming does not support socp solver, suspend this function
@jit
def combine_problem_matrices(P1,r1,q1,A2,b2,r2,q2):

    #[]     []      
    G_ort1, h_ort1, G_soc1, h_soc1 = ellipsoid_problem_matrices(P1, r1, q1)

    #                []      []
    G_ort2, h_ort2, G_soc2, h_soc2 = polytope_problem_matrices( A2, b2, r2, q2)

    ## reshape empty constraints to the same size
    # G_ort1=jnp.empty(G_ort2.shape)
    # h_ort1=jnp.empty(h_ort2.shape)

    # G_soc2=jnp.empty(G_soc1.shape)
    # h_soc2=jnp.empty(h_soc1.shape)

    h_ort1 = jnp.reshape(h_ort1,(-1,1))
    h_ort2 = jnp.reshape(h_ort2,(-1,1))
    h_soc1 = jnp.reshape(h_soc1,(-1,1))
    h_soc2 = jnp.reshape(h_soc2,(-1,1))

    G=jnp.vstack((G_ort2,G_soc1))
    h=jnp.vstack((h_ort2,h_soc1))
    c = jnp.array([0,0,0,1.])
    return c,G,h

## solve the conic problem with SOCP, by ECOS solver
def solve_conic(c,G,h):
    ## dims of the orthogonal constraints
    c = np.array(c)
    h = np.array(h).reshape(-1)
    G = csc_matrix(G)
    
 
    sol = ecos.solve(c,G,h,dims={'l':6,'q':[4]},verbose=True)
    # 4,
    x = jnp.array(sol['x'])
    
    # 10,
    s = jnp.array(sol['s'])

    # 10,
    z = jnp.array(sol['z'])
    return x,s,z
@custom_jvp
def ellipsoid_polytope_proximity(P1,r1,q1,A2,b2,r2,q2):
    c,G,h=combine_problem_matrices(P1,r1,q1,A2,b2,r2,q2)
    x,s,z=solve_conic(c,G,h)
    return x[3]

@jit
def ellipsoid_polytope_lagrangian(P1,r1,q1,A2,b2,r2,q2, x, s, z):
    c,G,h=combine_problem_matrices(P1,r1,q1,A2,b2,r2,q2)

    return z.dot(G @ x - h.reshape(-1))

def ellipsoid_polytope_proximity_grads(P1,r1,q1,A2,b2,r2,q2):
    c,G,h=combine_problem_matrices(P1,r1,q1,A2,b2,r2,q2)
    x,s,z = solve_conic(c,G,h)
    
    alpha = x[3]

    lag_grad = jit(grad(ellipsoid_polytope_lagrangian, argnums = (0,1,2,3,4,5,6)))
    grads = lag_grad(P1, r1, q1, A2, b2, r2, q2, x, s, z)

    gP1, gr1, gq1, gA2, gb2, gr2, gq2 = grads 

    return alpha, gP1, gr1, gq1, gA2, gb2, gr2, gq2

## Jacobian-vector product
## for customized autodiff
@ellipsoid_polytope_proximity.defjvp
def _ellipsoid_polytope_proximity_gradient(primals, targets):
  P1,r1,q1,A2,b2,r2,q2 = primals 
  dP1,dr1,dq1,dA2,db2,dr2,dq2 = targets 

  grads = ellipsoid_polytope_proximity_grads(P1,r1,q1,A2,b2,r2,q2)

  alpha, gP1, gr1, gq1, gA2, gb2, gr2, gq2 = grads

  primal_out = alpha 

  tangent_out = (jnp.sum(dP1 * gP1) + dr1.dot(gr1) + dq1.dot(gq1) + 
                 jnp.sum(dA2 * gA2) + db2.dot(gb2) + dr2.dot(gr2) + dq2.dot(gq2)) 
  
  return primal_out, tangent_out 

grad_f = grad(ellipsoid_polytope_proximity, argnums =(1,2))#(0,1,2,3,4,5,6,7)
if __name__ == "__main__":

    P = jnp.eye(3)
    r = jnp.array([1.0, 2.0, 3.0]) 
    q = jnp.array([1.0, 0.0, 0.0, 0.0])  

    G_ort, h_ort, G_soc, h_soc = ellipsoid_problem_matrices(P, r, q)
    # print("G_soc:", G_soc)
    # print("h_soc:", h_soc)

    A2, b2 = create_rect_prism(1.0,2.0,3.0)
    r2 = jnp.array([-1,0.1,2.])
    # p2 = jnp.array([-.3,.3,-.2])
    # Q2 = dcm_from_mrp(p2)
    # q2=R.from_matrix(Q2).as_quat()
    q2=jnp.array([0.0,0.0,0.0,1.0])
    q2=jnp.roll(q2,1)

 
    alpha = ellipsoid_polytope_proximity(P,r,q,A2,b2,r2,q2)
    print("alpha: ", alpha)

    
    grads = grad_f(P,r,q,A2,b2,r2,q2)
    print(grads)
    # check gradients 
    # check_grads(ellipsoid_polytope_proximity,(P,r,q,A2,b2,r2,q2), order=1, atol = 2e-1)