# hyper-rational-games
  This project is a solver for the Cauchy problem proposed on my master's degree final project, still to be published. 
  [hrgames.py](hrgames.py) is a module with two functions, one that receives the adjacency matrix, payoff matrices, relationship matrix and initial state of the system to return
  the increments in each strategy for each vertex, the second receives the same parameters plus a time interval and number of steps and returns the state of the system after the game is played through that time interval. [examples.py](examples.py) has a collection of examples made by myself and contributors.
  
  ## Usage
  To run the [hrgames](hrgames.py) module you must have Numpy installed. To run any examples from [examples.py](examples.py) you'll need Numpy, Matplotlib and the [hrgames](hrgames.py) module.
  You can import the hyper-rational evolutionary game module using:
  ```
  import hrgames as hrg
  ```
  You can run the examples from [examples.py](examples.py) using:
  ```
  python examples.py example_name
  ```
  
  Where `example_name` can be any of the following:
  - classic_pd
  - classic_pd_negative
  - hinge_love_triangle
  - k3_love_triangle
  - closed_star_pd
  - closed_star_coex
  
  You can change any parameter directly on the script code. It's all commented and easily accessible for anyone.
  
  Alternatively, you can import the examples script as a module using:
  ```
  import examples as ex
  ```
  Then run any example as a function. This way you can send in the optional parameters `R`, `tf` and `xini` for the relationship matrix, final time and initial conditions, respectively. For example, one could use:
  ```
  >>> import examples as ex
  >>> finalTime = 200
  >>> initialCond = np.divide(np.ones(3,2), 2)
  >>> ex.k3_love_triangle(R = np.eye(3), tf = finalTime, xini = initialCond)
  ```
  With this you'd get the plotted results from a love triangle game where all players are selfish with half probability of using either strategy for 200 time iterations.
  
  There are many interesting games out there that were still no simulated as a hyper-rational game, so feel free to add any new or different examples!
  
  ## Problem description
  This script solves the replicator equation Cauchy problem given by:
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\bg_white&space;\left\{\begin{matrix}&space;\Dot{x}_{v,s}(t)=x_{v,s}(t)\left(\rho_{v,s}(t)-\phi_v(t)\right)\\&space;\Dot{x}_{v,s}(0)=c_{v,s}&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;\left\{\begin{matrix}&space;\Dot{x}_{v,s}(t)=x_{v,s}(t)\left(\rho_{v,s}(t)-\phi_v(t)\right)\\&space;\Dot{x}_{v,s}(0)=c_{v,s}&space;\end{matrix}\right." title="\left\{\begin{matrix} \Dot{x}_{v,s}(t)=x_{v,s}(t)\left(\rho_{v,s}(t)-\phi_v(t)\right)\\ \Dot{x}_{v,s}(0)=c_{v,s} \end{matrix}\right." /></a>
  
  Where <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\bg_white&space;x_{v,s}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;x_{v,s}" title="x_{v,s}" /></a>
  is the chance of the vertex player *v* to use strategy *s* and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\bg_white&space;c_{v,s}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;c_{v,s}" title="c_{v,s}" /></a>
  is the initial condition for every vertex and strategy. The hyper-rational payoff function is given by:
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\pi_v(x_v,x_{-v})=r_{v,v}x_v^TB_v&space;\left(\frac{1}{d_v}\sum_{u=1}^N&space;a_{v,u}x_u\right)&plus;&space;\left(\frac{1}{d_{v,r}}\sum_{u=1}^N&space;r_{v,u}a_{v,u}x_u^TB_u\right)&space;x_v" target="_blank"><img src="https://latex.codecogs.com/png.latex?\bg_white&space;\pi_v(x_v,x_{-v})=r_{v,v}x_v^TB_v&space;\left(\frac{1}{d_v}\sum_{u=1}^N&space;a_{v,u}x_u\right)&plus;&space;\left(\frac{1}{d_{v,r}}\sum_{u=1}^N&space;r_{v,u}a_{v,u}x_u^TB_u\right)&space;x_v" title="\pi_v(x_v,x_{-v})=r_{v,v}x_v^TB_v \left(\frac{1}{d_v}\sum_{u=1}^N a_{v,u}x_u\right)+ \left(\frac{1}{d_{v,r}}\sum_{u=1}^N r_{v,u}a_{v,u}x_u^TB_u\right) x_v" /></a>
  
  We define <a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\rho_{v,s}=\pi_v(e_s,x_{-v})" target="_blank"><img src="https://latex.codecogs.com/png.latex?\bg_white&space;\rho_{v,s}=\pi_v(e_s,x_{-v})" title="\rho_{v,s}=\pi_v(e_s,x_{-v})" /></a>
  as the payment of the pure strategy *s* and <a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\phi_v=\pi_v(x_v,x_{-v})" target="_blank"><img src="https://latex.codecogs.com/png.latex?\bg_white&space;\phi_v=\pi_v(x_v,x_{-v})" title="\phi_v=\pi_v(x_v,x_{-v})" /></a>
  as the expected payoff of the mixed strategy <a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;x_v" target="_blank"><img src="https://latex.codecogs.com/png.latex?\bg_white&space;x_v" title="x_v" /></a>
  of *v*.
