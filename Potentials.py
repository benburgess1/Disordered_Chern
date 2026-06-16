import numpy as np

def V_sq_sep(x, y, beta=1, a=1, V=1, phi_x=0., phi_y=0., **kwargs):
    G = 2 * np.pi * beta / a
    return V * (np.cos(x*G + phi_x) + np.cos(y*G + phi_y))

def V_sq_nonsep(x, y, beta=1, a=1, V=1, phi_1=0., phi_2=0., **kwargs):
    G = 2 * np.pi / a
    return V * (np.cos(beta * G * (x+y) + phi_1) + np.cos(beta * G * (x-y) + phi_2))

def V_pw(x, beta=1, a=1, V=1, phi_x=0., **kwargs):
    G = 2 * np.pi * beta / a
    return V * np.cos(x*G + phi_x)

def V_hex_wpl(x, y, beta=1, V=1, a=1, phi_1=0, phi_2=0, **kwargs):
    G = beta * 4 * np.pi / (np.sqrt(3) * a)
    G1 = G * np.array([np.sqrt(3)/2, -1/2])
    G2 = G * np.array([0, 1])
    return V * (np.cos((G1[0]+G2[0])*x + (G1[1]+G2[1])*y + phi_1) + np.cos((G1[0]-G2[0])*x + (G1[1]-G2[1])*y + phi_2))

def V_hex_sep(x, y, beta=1, V=1, a=1, phi_1=0, phi_2=0, **kwargs):
    G = beta * 4 * np.pi / (np.sqrt(3) * a)
    G1 = G * np.array([np.sqrt(3)/2, -1/2])
    G2 = G * np.array([0, 1])
    return V * (np.cos(G1[0]*x + G1[1]*y + phi_1) + np.cos(G2[0]*x + G2[1]*y + phi_2))

def V_hex_rot(x, y, beta=1, V=1, a=1, phi_1=0, phi_2=0, theta=0., **kwargs):
    G = beta * 4 * np.pi / (np.sqrt(3) * a)
    G1 = G * np.array([np.sqrt(3)/2, -1/2])
    G2 = G * np.array([0, 1])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    G1 = R @ G1
    G2 = R @ G2
    return V * (np.cos((G1[0]+G2[0])*x + (G1[1]+G2[1])*y + phi_1) + np.cos((G1[0]-G2[0])*x + (G1[1]-G2[1])*y + phi_2))