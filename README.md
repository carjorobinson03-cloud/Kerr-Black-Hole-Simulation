# Kerr-Black-Hole-Simulation
Python and GLSL simulation of Kerr Black holes. 

<img src="Maximal%20Spin,%20Mid%20Temp.png" width="700" alt="Maximal spin, medium temperature regime">

<img src="Swarzchild%20Collapse.png" width="700" alt="Schwarzschild collapse">

Null geodesic integration via RK4 over an affine parameter. The integrator uses
Kerr–Schild coordinates with a Boyer–Lindquist fallback, with geodesics
initialized at the ZAMO position. Disc colouring uses a lookup table built from
Planck's law, with the redshift factor applied. The goal was to keep the render
as physical as possible; all varied parameters are in natural units.
