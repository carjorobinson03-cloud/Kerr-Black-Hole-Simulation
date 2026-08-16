#Starfield background
import os
import pandas as pd
import numpy as np
from colour import bb_to_rgb

url  = "https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/CURRENT/hygdata_v41.csv"
path = "hygdata_v41.csv"

if not os.path.exists(path):
    pd.read_csv(url).to_csv(path, index=False)   # download once, only if missing
df = pd.read_csv(path) 
if not os.path.exists("starfield.npy"):
    print("baking starfield (one-time)...")

#Physical data taken from HYG database for starfield
df = df[df.id != 0].dropna(subset = ["mag"]) #sun has posn, 0, 0 ,0. we can see the problem.
dist = df[["x", "y", "z"]].to_numpy() # star position vectors, (N, 3) shape
nmlzd_dist = dist / np.linalg.norm(dist, axis=1, keepdims=True)
apprnt_magnitude = df["mag"].to_numpy()
b_v = df["ci"].fillna(0.65).to_numpy() #NAN guards

T_stars   = 4600.0 * (1.0/(0.92*b_v + 1.7) + 1.0/(0.92*b_v + 0.62)) #Bastelleros
T_lut_grid = np.linspace(1000.0, 40000.0, 1024)
lut_rgb    = np.array([bb_to_rgb(T) for T in T_lut_grid], dtype=np.float32)
rgb_stars  = np.stack([np.interp(T_stars, T_lut_grid, lut_rgb[:, i]) for i in range(3)],
                      axis=1).astype(np.float32)

m_ref = -1.46 #choose sirius as reference because relasitically it should be brightest
stellar_brightness = 10.0**(-0.4*(apprnt_magnitude - m_ref)) #Pogson, 1856
stellar_colour = rgb_stars*stellar_brightness[:, None] # increase dimensionality

#direction mapping
Height_bkg = 2048
Width_bkg = 4096
u = 0.5 + np.arctan2(nmlzd_dist[:,1], nmlzd_dist[:,0]) / (2*np.pi)
v = np.arccos(np.clip(nmlzd_dist[:,2], -1, 1)) / np.pi
px = (u * Width_bkg).astype(int) % Width_bkg
py = np.clip((v * Height_bkg).astype(int), 0, Height_bkg-1) #all just clipping to background space.

#Building a convulution model for the background stars, should fix the errors currently plaguing them.
sigma = 1.5 #for the love of everything holy dont touch this (Unless youre me teehee)
k = int(np.ceil(3*sigma))
gy, gx = np.mgrid[-k:k+1, -k:k+1]
ker = np.exp((-(gx**2 + gy**2)) / (2*sigma**2)).astype(np.float32)
ker /= np.sum(ker) #This whole method seems funky, but if you want to spread the light across neighbouring texels, u need to 
                   #consider a grid of all possible values of light seeping over, done through a meshgrid and gaassian across it


sky_map = np.zeros((Height_bkg, Width_bkg, 3), np.float32)
for dy in range(-k, k+1):
    for dx in range(-k, k+1):
        w = ker[dy+k, dx+k]
        yk = np.clip(py+dy, 0, Height_bkg - 1)
        xk = (px + dx) % Width_bkg
        np.add.at(sky_map, (yk, xk), stellar_colour * w) #trivial to see how matrix power convolution.

np.save("starfield.npy", sky_map)   # freeze it for the GL upload
print("starfield baked.")
