# Kerr-Black-Hole-Simulation
Python and GLSL simulation of Kerr Black holes. 

<img src="Maximal%20Spin,%20Mid%20Temp.png" width="700" alt="Maximal spin, medium temperature regime">

<img src="Swarzchild%20Collapse.png" width="700" alt="Schwarzschild collapse">

Null geodesic integration via RK4 over an affine parameter. The integrator uses
Kerr–Schild coordinates with a Boyer–Lindquist fallback, with geodesics
initialized at the ZAMO position. Disc colouring uses a lookup table built from
Planck's law, with the redshift factor applied. The goal was to keep the render
as physical as possible; all varied parameters are in natural units.



## References

1. Bardeen, J. M., Press, W. H., & Teukolsky, S. A. (1972). Rotating black holes: locally nonrotating frames, energy extraction, and scalar synchrotron radiation. *ApJ* **178**, 347. [doi:10.1086/151796](https://doi.org/10.1086/151796)
2. Novikov, I. D., & Thorne, K. S. (1973). Astrophysics of black holes. In *Black Holes* (Les Houches 1972), eds. C. DeWitt & B. DeWitt, Gordon & Breach, 343.
3. Nash, D. HYG Stellar Database, v4.2. [astronexus.com/projects/hyg](https://www.astronexus.com/projects/hyg) — licensed CC BY-SA 4.0.
