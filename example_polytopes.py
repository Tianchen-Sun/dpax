import jax
import jax.numpy as jnp 
from jax import jit, grad, vmap 
from jax.test_util import check_grads
from jax.scipy.spatial.transform import Rotation as R
import dpax
from dpax.mrp import dcm_from_mrp
from dpax.polytopes import polytope_proximity
 
# rectangular prism in Ax<=b form (halfspace form)
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

# create polytopes 
A1, b1 = create_rect_prism(1,2,3)
A2, b2 = create_rect_prism(2,4,3)

# position and attitude for each polytope 
r1 = jnp.array([1,3,-2.])
p1 = jnp.array([.1,.3,.4])
Q1 = dcm_from_mrp(p1)

r2 = jnp.array([-1,0.1,2.])
p2 = jnp.array([-.3,.3,-.2])
Q2 = dcm_from_mrp(p2)

# calculate proximity (alpha <= 1 means collision) 

q1=R.from_matrix(Q1).as_quat()
q2=R.from_matrix(Q2).as_quat()
q1=jnp.roll(q1,1)
q2=jnp.roll(q2,1)
alpha = polytope_proximity(A1,b1,r1,q1,A2,b2,r2,q2)


print("alpha: ", alpha)

# calculate all the gradients 
grad_f = jit(grad(polytope_proximity, argnums =(3) ))#(0,1,2,3,4,5,6,7)
grads = grad_f(A1,b1,r1,q1,A2,b2,r2,q2)
print(grads)
# check gradients 
check_grads(polytope_proximity,  (A1,b1,r1,q1,A2,b2,r2,q2), order=1, atol = 2e-1)