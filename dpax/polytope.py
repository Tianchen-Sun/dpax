import jax
import ecos

from jax import grad 
from jax import jit 
from jax import custom_jvp
from jax import jvp 
import jax.numpy as jnp 
from jax.scipy.spatial.transform import Rotation as R
import dpax
from dpax.mrp import dcm_from_mrp

def create_rect_prism(length, width, height):

  A = jnp.array([
      [1,0,0.],
      [0,1,0.],
      [0,0,1.],
      [-1,0,0.],
      [0,-1,0.],
      [0,0,-1.]
  ])

  cs = jnp.array([
      [length/2,0,0.],
      [0,width/2,0.],
      [0.,0,height/2],
      [-length/2,0,0.],
      [0,-width/2,0.],
      [0.,0,-height/2]
  ])

  # b[i] = dot(A[i,:], b[i,:]) 
  b = jax.vmap(jnp.dot, in_axes = (0,0))(A, cs)

  return A, b 
def polytope_problem_matrices(A,b,r,q):
    """
    Parameters:
    A: 3x3 matrix
    b: 3x1 vector
    r: 3x1 vector
    Q: 3x3 matrix

    Returns:
    G_ort: Empty matrix (no orthogonal constraint) A^T@B = 0
    h_ort: Empty vector (no orthogonal constraint)
    G_soc: Second-order cone constraint matrix
    h_soc: Second-order cone constraint vector
    """
    q=jnp.roll(q,-1)
    
    Q=R.from_quat(q).as_matrix()

    # [6 x 3 : 6 x 1]
    G_ort = jnp.hstack((A @ Q.T, -jnp.reshape(b,(-1,1))))
      
    
    h_ort = A @ Q.T @ r
    

    G_soc = jnp.empty([])
    h_soc = jnp.empty([])

    return G_ort, h_ort, G_soc, h_soc