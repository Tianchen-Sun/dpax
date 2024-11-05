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

@jit
def ellipsoid_problem_matrices(P, r, q):
    """
    Parameters:
    P: 3x3 matrix
    r: 3x1 vector
    q: 4x1 quaternion w,x,y,z

    Returns:
    G_ort: Empty matrix (no orthogonal constraint) A^T@B = 0
    h_ort: Empty vector (no orthogonal constraint)
    G_soc: Second-order cone constraint matrix
    h_soc: Second-order cone constraint vector
    """
    U=jnp.linalg.cholesky(P)
    q=jnp.roll(q,-1)
    Q=R.from_quat(q).as_matrix()


    h_ort = jnp.empty([])
    G_ort = jnp.empty([])

    ## 1 x 4 vector
    h_soc = jnp.concatenate([jnp.array([0]), -U @ Q.T @ r])

    
    G_soc_top = jnp.array([[0, 0, 0, -1]])  
    G_soc_bot = jnp.hstack([-U @ Q.T, jnp.zeros((3, 1))])  


    G_soc = jnp.vstack([G_soc_top, G_soc_bot])  
    
    return G_ort, h_ort, G_soc, h_soc

## naming follows the julia code


if __name__ == "__main__":

    P = jnp.eye(3)  
    r = jnp.array([1.0, 2.0, 3.0]) 
    Q = jnp.array([1,0,0,0])  

    G_ort, h_ort, G_soc, h_soc = ellipsoid_problem_matrices(P, r, Q)

    print("G_ort:", G_ort)
    print("h_ort:", h_ort)
