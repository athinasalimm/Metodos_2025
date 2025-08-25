import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

#Les dejo este codigo que quizas les sirve para el ejercicio de la guia donde newton es con 3 ecuaciones! (Recuerden que les queda un jacobiano 3x3)
#Este es el caso con 2 sensores, el ejercicio es con 3 sensores, pero sirve para darse una idea de como al introducir el tercer sensor se reduce la interseccion de las esferas a dos puntos.

def ortho_basis_from_normal(n):
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ey = np.cross(n, a); ey /= np.linalg.norm(ey)
    ez = np.cross(n, ey); ez /= np.linalg.norm(ez)
    return ey, ez

def sphere_wireframe(center, r, n_mer=36, n_par=18):
    cx, cy, cz = center
    traces = []
    phi = np.linspace(0, np.pi, n_par)
    for t in np.linspace(0, 2*np.pi, n_mer):
        x = cx + r*np.sin(phi)*np.cos(t)
        y = cy + r*np.sin(phi)*np.sin(t)
        z = cz + r*np.cos(phi)
        traces.append(go.Scatter3d(x=x, y=y, z=z, mode="lines", showlegend=False))
    u = np.linspace(0, 2*np.pi, n_mer)
    for p in np.linspace(0, np.pi, n_par):
        x = cx + r*np.sin(p)*np.cos(u)
        y = cy + r*np.sin(p)*np.sin(u)
        z = cz + r*np.cos(p)*np.ones_like(u)   # <- CLAVE: array, no escalar
        traces.append(go.Scatter3d(x=x, y=y, z=z, mode="lines", showlegend=False))
    return traces

def intersection_circle_two_spheres(c1, r1, c2, r2):
    p1, p2 = np.array(c1, float), np.array(c2, float)
    d = np.linalg.norm(p2 - p1)
    if d == 0 or d > r1 + r2 or d < abs(r1 - r2):
        return None, None, None
    ex = (p2 - p1) / d
    a = (r1**2 - r2**2 + d**2) / (2*d)
    pc = p1 + a*ex
    h = np.sqrt(max(r1**2 - a**2, 0.0))
    ey, ez = ortho_basis_from_normal(ex)
    return pc, h, (ey, ez)

c1, r1 = (0, 0, 0), 3.0
c2, r2 = (4.0, 1.0, 0.5), 2.6

pc, rc, basis = intersection_circle_two_spheres(c1, r1, c2, r2)

traces = []
traces += sphere_wireframe(c1, r1)
traces += sphere_wireframe(c2, r2)
traces.append(go.Scatter3d(x=[c1[0]], y=[c1[1]], z=[c1[2]], mode="markers+text", text=["S1"]))
traces.append(go.Scatter3d(x=[c2[0]], y=[c2[1]], z=[c2[2]], mode="markers+text", text=["S2"]))

if pc is not None:
    t = np.linspace(0, 2*np.pi, 400)
    ey, ez = basis
    circle = pc.reshape(3,1) + rc*(ey.reshape(3,1)*np.cos(t) + ez.reshape(3,1)*np.sin(t))
    traces.append(go.Scatter3d(x=circle[0], y=circle[1], z=circle[2], mode="lines", name="Intersección"))

fig = go.Figure(data=traces)
fig.update_layout(title="Intersección de 2 esferas (3D)", scene_aspectmode="data")
fig.show()