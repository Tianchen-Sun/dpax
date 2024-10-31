import jax 
from jax import grad 
from jax import jit 
from jax import custom_jvp
from jax import jvp 
import jax.numpy as jnp 
from jax.scipy.spatial.transform import Rotation as R
from dpax.pdip_solver import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
function to create standard form min c'x st Gx<=h from 
the description of two polytopes using the DCOL alg. 

https://arxiv.org/abs/2207.00669

each polytope is described in a body frame (B) by Ax<=b, 
with a position r in some world frame (W), and an attitude 
described by a 3x3 rotation matrix W_Q_B. 

args:
	A1: [n1,3] jnp.array, polytope 1 description (Ax <= b)
	b1: [n1] jnp.array, polytope 1 description (Ax <= b)
	r1: [3] jnp.array, position of polytope 1 in world frame 
	Q1: [3,3] jnp.array, W_Q_B rotation matrix for poly 1 
	A2: [n2,3] jnp.array, polytope 2 description (Ax <= b)
	b2: [n2] jnp.array, polytope 2 description (Ax <= b)
	r2: [3] jnp.array, position of polytope 2 in world frame 
	Q2: [3,3] jnp.array, W_Q_B rotation matrix for poly 2

outputs:
	c: [4] jnp.array, linear cost term 
	G: [n1 + n2, 4] jnp.array, inequality constraint Gx<=h
	h: [n1 + n2] jnp.array, inequality constraint Gx<=h

"""
def problem_matrices(A1, b1,r1,q1,A2,b2,r2,q2):
  q1=jnp.roll(q1,-1)
  q2=jnp.roll(q2,-1)
  Q1=R.from_quat(q1).as_matrix()
  Q2=R.from_quat(q2).as_matrix()

  
  c = jnp.array([0,0,0,1.])

  G = jnp.vstack((
      jnp.hstack((A1 @ Q1.T, -jnp.reshape(b1,(-1,1)) )),
      jnp.hstack((A2 @ Q2.T, -jnp.reshape(b2,(-1,1)) ))
  ))
  h = jnp.concatenate((
      A1 @ Q1.T @ r1, 
      A2 @ Q2.T @ r2, 
  ))

  return c, G, h


@custom_jvp
@jit 
def polytope_proximity(A1,b1,r1,q1,A2,b2,r2,q2):
  
  c, G, h = problem_matrices(A1,b1,r1,q1,A2,b2,r2,q2)
  x,s,z = solve_lp(c,G,h)
  return x[3]

@jit 
def polytope_lagrangian(A1,b1,r1,q1,A2,b2,r2,q2, x, s, z):

  c, G, h = problem_matrices(A1,b1,r1,q1,A2,b2,r2,q2)

  # ommit the cost term since c'x doesn't depend on problem data 
  return z.dot(G @ x - h)

@jit 
def polytope_proximity_grads(A1,b1,r1,q1,A2,b2,r2,q2):

  c, G, h = problem_matrices(A1,b1,r1,q1,A2,b2,r2,q2)
  x,s,z = solve_lp(c,G,h)

  alpha = x[3]

  lag_grad = grad(polytope_lagrangian, argnums = (0,1,2,3,4,5,6,7))
  grads = lag_grad(A1,b1,r1,q1,A2,b2,r2,q2, x, s, z)

  gA1, gb1, gr1, gq1, gA2, gb2, gr2, gq2 = grads 

  return alpha, gA1, gb1, gr1, gq1, gA2, gb2, gr2, gq2

## Jacobian-vector product
## for customized autodiff
@polytope_proximity.defjvp
@jit 
def _polytope_proximity_gradient(primals, targets):
  A1,b1,r1,q1,A2,b2,r2,q2 = primals 
  dA1,db1,dr1,dq1,dA2,db2,dr2,dq2 = targets 

  grads = polytope_proximity_grads(A1,b1,r1,q1,A2,b2,r2,q2)

  alpha, gA1, gb1, gr1, gq1, gA2, gb2, gr2, gq2 = grads

  primal_out = alpha 

  tangent_out = (jnp.sum(dA1 * gA1) + db1.dot(gb1) + dr1.dot(gr1) + dq1.dot(gq1) + 
                 jnp.sum(dA2 * gA2) + db2.dot(gb2) + dr2.dot(gr2) + dq2.dot(gq2)) 
  
  return primal_out, tangent_out 
  
grad_f = jit(grad(polytope_proximity, argnums =(2,3) ))#(0,1,2,3,4,5,6,7)