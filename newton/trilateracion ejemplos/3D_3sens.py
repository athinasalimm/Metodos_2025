import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

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
        z = cz + r*np.cos(p)*np.ones_like(u)
        traces.append(go.Scatter3d(x=x, y=y, z=z, mode="lines", showlegend=False))
    return traces

def _basis_from_normal(n):
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ey = np.cross(n, a); ey /= np.linalg.norm(ey)
    ez = np.cross(n, ey); ez /= np.linalg.norm(ez)
    return ey, ez

def intersection_circle_two_spheres(c1, r1, c2, r2, n_pts=500):
    p1, p2 = np.array(c1, float), np.array(c2, float)
    d = np.linalg.norm(p2 - p1)
    if d == 0 or d > r1 + r2 or d < abs(r1 - r2):
        return None  
    ex = (p2 - p1) / d
    a = (r1**2 - r2**2 + d**2) / (2*d)
    pc = p1 + a*ex                      
    rc = np.sqrt(max(r1**2 - a**2, 0.0)) 
    ey, ez = _basis_from_normal(ex)     

    t = np.linspace(0, 2*np.pi, n_pts)
    circle = pc.reshape(3,1) + rc*(ey.reshape(3,1)*np.cos(t) + ez.reshape(3,1)*np.sin(t))
    return circle  


def trilaterate_3spheres(c1, r1, c2, r2, c3, r3, tol=1e-9):
    P1, P2, P3 = np.array(c1, float), np.array(c2, float), np.array(c3, float)
    ex = P2 - P1
    d = np.linalg.norm(ex)
    if d < tol: return []
    ex /= d
    i = np.dot(ex, P3 - P1)
    temp = P3 - P1 - i*ex
    temp_norm = np.linalg.norm(temp)
    if temp_norm < tol: return []  
    ey = temp / temp_norm
    ez = np.cross(ex, ey)

    j = np.dot(ey, P3 - P1)
    x = (r1**2 - r2**2 + d**2) / (2*d)
    y = (r1**2 - r3**2 + i**2 + j**2 - 2*i*x) / (2*j)

    z2 = r1**2 - x**2 - y**2
    if z2 < -tol: return []
    z = np.sqrt(max(z2, 0.0))

    p_base = P1 + x*ex + y*ey
    sol1 = p_base + z*ez
    if z < tol:
        return [sol1]
    sol2 = p_base - z*ez
    return [sol1, sol2]

c1, r1 = (0.0, 0.0, 0.0), 3.0
c2, r2 = (4.0, 0.5, 0.0), 2.7
c3, r3 = (1.5, 3.5, 2.0), 3.1

sols = trilaterate_3spheres(c1, r1, c2, r2, c3, r3)

traces = []
traces += sphere_wireframe(c1, r1)
traces += sphere_wireframe(c2, r2)
traces += sphere_wireframe(c3, r3)

for i, c in enumerate([c1, c2, c3], start=1):
    traces.append(go.Scatter3d(x=[c[0]], y=[c[1]], z=[c[2]],
                               mode="markers+text", text=[f"S{i}"], textposition="top center"))

circle = intersection_circle_two_spheres(c1, r1, c2, r2)
if circle is not None:
    traces.append(go.Scatter3d(
        x=circle[0], y=circle[1], z=circle[2],
        mode="lines",
        name="Intersección S1–S2",
        line=dict(width=5, color="black")  
    ))

circle = intersection_circle_two_spheres(c1, r1, c3, r3)
if circle is not None:
    traces.append(go.Scatter3d(
        x=circle[0], y=circle[1], z=circle[2],
        mode="lines",
        name="Intersección S1–S3",
        line=dict(width=5, color="black") 
    ))

circle = intersection_circle_two_spheres(c2, r2, c3, r3)
if circle is not None:
    traces.append(go.Scatter3d(
        x=circle[0], y=circle[1], z=circle[2],
        mode="lines",
        name="Intersección S2–S3",
        line=dict(width=5, color="black")  
    ))

if len(sols) == 1:
    s = sols[0]
    traces.append(go.Scatter3d(x=[s[0]], y=[s[1]], z=[s[2]],
                               mode="markers+text", text=["Solución única"]))
elif len(sols) == 2:
    s1, s2 = sols
    traces.append(go.Scatter3d(x=[s1[0]], y=[s1[1]], z=[s1[2]],
                               mode="markers+text", text=["Solución A"]))
    traces.append(go.Scatter3d(x=[s2[0]], y=[s2[1]], z=[s2[2]],
                               mode="markers+text", text=["Solución B"]))

fig = go.Figure(data=traces)
fig.update_layout(title="Trilateración 3D: círculo S1–S2 y soluciones con 3 esferas",
                  scene_aspectmode="data")
fig.show()