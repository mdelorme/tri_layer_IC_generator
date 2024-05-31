import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from matplotlib.widgets import Button, Slider, RadioButtons

log_name = 'log.txt'
log_out = open(log_name, 'w')
def stdout(*args):
  line = ' '.join(args)
  print(line)
  log_out.write(line + '\n')



def inputd(prompt, default_val):
  '''
  Input method with a default value as float
  '''
  try:
    val = float(input(prompt + f'[{default_val}]: '))
  except ValueError:
    val = default_val
  return val

def inputi(prompt, default_val):
  '''
  Input method with a default value as integer
  '''
  try:
    val = int(input(prompt + f'[{default_val}]: '))
  except ValueError:
    val = default_val
  return val

def inputs(promp, default_val):
  '''
  Input method with a default value as string
  '''
  val = input(promp + f'[{default_val}]: ')
  if val == '':
    val = default_val
  return val


stdout(''' ███████████            ███             ████                                             █████   █████████ 
░█░░░███░░░█           ░░░             ░░███                                            ░░███   ███░░░░░███
░   ░███  ░  ████████  ████             ░███   ██████   █████ ████  ██████  ████████     ░███  ███     ░░░ 
    ░███    ░░███░░███░░███  ██████████ ░███  ░░░░░███ ░░███ ░███  ███░░███░░███░░███    ░███ ░███         
    ░███     ░███ ░░░  ░███ ░░░░░░░░░░  ░███   ███████  ░███ ░███ ░███████  ░███ ░░░     ░███ ░███         
    ░███     ░███      ░███             ░███  ███░░███  ░███ ░███ ░███░░░   ░███         ░███ ░░███     ███
    █████    █████     █████            █████░░████████ ░░███████ ░░██████  █████        █████ ░░█████████ 
   ░░░░░    ░░░░░     ░░░░░            ░░░░░  ░░░░░░░░   ░░░░░███  ░░░░░░  ░░░░░        ░░░░░   ░░░░░░░░░  
                                                         ███ ░███                                          
                                                        ░░██████                                           
                                                         ░░░░░░                                            ''')
stdout('1. Generating model')
stdout('===================')

params = namedtuple("Parameters", "theta0 theta1 theta2 dz0 dz1 dz2 chi_rho m0 m1 m2 rho0 Nz yscale")
gamma0 = 5.0/3.0

params.theta0 = 4.0
params.theta1 = 10.0
params.theta2 = 4.0
params.dz0 = 1.0
params.dz1 = 1.0
params.dz2 = 1.0
params.chi_rho = 3.0
params.m0 = 4.0
params.m1 = 1.0
params.m2 = 4.0
params.rho0  = 1.0
params.Nz = 256
params.yscale = 'linear'

def get_profile(params):
  z0   = 0.0
  z1   = params.dz0
  z2   = params.dz0+params.dz1
  zmax = params.dz0+params.dz1+params.dz2
  
  dz = (zmax - z0) / params.Nz
  z  = np.linspace(z0 + 0.5*dz, zmax - 0.5*dz, params.Nz)

  T0 = params.theta1*params.dz1 / (params.chi_rho**(1.0/params.m1) - 1.0) - params.theta0*params.dz0
  T1 = T0 + params.theta0*params.dz0
  T2 = T1 + params.theta1*params.dz1
  
  rho1 = params.rho0 * (T1/T0)**params.m0
  rho2 = rho1 * (T2/T1)**params.m1
  p0   = params.rho0*T0
  p1   = rho1*T1
  p2   = rho2*T2

  rho = np.empty_like(z)
  p   = np.empty_like(z)
  T   = np.empty_like(z)
  dom1 = z < z1
  dom2 = (z >= z1) & (z <= z2)
  dom3 = z >= z2

  T[dom1] = T0 + z[dom1]      * params.theta0
  T[dom2] = T1 + (z[dom2]-z1) * params.theta1
  T[dom3] = T2 + (z[dom3]-z2) * params.theta2

  rho[dom1] = params.rho0 * (T[dom1]/T0)**params.m0
  rho[dom2] =        rho1 * (T[dom2]/T1)**params.m1
  rho[dom3] =        rho2 * (T[dom3]/T2)**params.m2

  p[dom1] = p0 * (T[dom1]/T0)**(params.m0+1)
  p[dom2] = p1 * (T[dom2]/T1)**(params.m1+1)
  p[dom3] = p2 * (T[dom3]/T2)**(params.m2+1)

  R = 1.0
  Sen = gamma0 / (gamma0-1.0) * np.log(T) - R*np.log(p)
  dSdr = np.gradient(Sen, z[0]-z[1])

  return z, T, rho, p, Sen, dSdr


fig, ax = plt.subplots(2, 2, figsize=(10, 10))
z, T, rho, p, Sen, dSdr = get_profile(params)
z1_line00 = ax[0,0].axvline(params.dz0, linestyle='--', color='black')
z2_line00 = ax[0,0].axvline(params.dz0+params.dz1, linestyle='--', color='black')
z1_line01 = ax[0,1].axvline(params.dz0, linestyle='--', color='black')
z2_line01 = ax[0,1].axvline(params.dz0+params.dz1, linestyle='--', color='black')
z1_line10 = ax[1,0].axvline(params.dz0, linestyle='--', color='black')
z2_line10 = ax[1,0].axvline(params.dz0+params.dz1, linestyle='--', color='black')
z1_line11 = ax[1,1].axvline(params.dz0, linestyle='--', color='black')
z2_line11 = ax[1,1].axvline(params.dz0+params.dz1, linestyle='--', color='black')
Tplt,    = ax[0,0].plot(z, T, color='blue')
rhoplt,  = ax[0,1].plot(z, rho, color='blue')
pplt,    = ax[1,0].plot(z, p, color='blue') 
dSdrplt, = ax[1,1].plot(z, dSdr, color='blue') 

ax[1,0].set_xlabel('z')
ax[1,1].set_xlabel('z')
ax[0,0].set_ylabel('Temperature')
ax[0,1].set_ylabel('Density')
ax[1,0].set_ylabel('Pressure')
ax[1,1].set_ylabel('Entropy gradient')

# Avoiding infinite recursion
mutex = False

fill_T   = None
fill_rho = None
fill_p   = None

def update_plots():
  global fill_T, fill_rho, fill_p
  z, T, rho, p, Sen, dSdr = get_profile(params)

  btn.set_active(not (np.any(T < 0.0) or np.any(rho < 0.0) or np.any(p < 0.0)))

  line_color='blue'
  if T.min() < 0.0:
    if fill_T:
      fill_T.remove()
    fill_T = ax[0,0].fill_between(z, 0.0, T.min(), hatch='///', fc='lightcoral', ec='white')
    line_color = 'red'
  if rho.min() < 0.0:
    if fill_rho:
      fill_rho.remove()
    fill_rho = ax[0,1].fill_between(z, 0.0, rho.min(), hatch='///', fc='lightcoral', ec='white')
    line_color = 'red'
  if p.min() < 0.0:
    if fill_p:
      fill_p.remove()
    fill_p = ax[1,0].fill_between(z, 0.0, p .min(), hatch='///', fc='lightcoral', ec='white')
    line_color = 'red'

  Tplt.set_ydata(T)
  rhoplt.set_ydata(rho)
  pplt.set_ydata(p)
  dSdrplt.set_ydata(dSdr)
  Tplt.set_xdata(z)
  rhoplt.set_xdata(z)
  pplt.set_xdata(z)
  dSdrplt.set_xdata(z)
  Tplt.set_color(line_color)
  rhoplt.set_color(line_color)
  pplt.set_color(line_color)
  dSdrplt.set_color(line_color)

  yscale = 'linear'
  if 'log' in params.yscale:
    yscale = 'log'
  yscaleT = 'linear'
  if params.yscale == 'log':
    yscaleT = 'log'
  
  ax[0,0].set_yscale(yscaleT)
  ax[0,1].set_yscale(yscale)
  ax[1,0].set_yscale(yscale)
  #ax[1,1].set_yscale(params.yscale)

  z1_line00.set_xdata(params.dz0)
  z1_line01.set_xdata(params.dz0)
  z1_line10.set_xdata(params.dz0)
  z1_line11.set_xdata(params.dz0)
  z2_line00.set_xdata(params.dz0+params.dz1)
  z2_line01.set_xdata(params.dz0+params.dz1)
  z2_line10.set_xdata(params.dz0+params.dz1)
  z2_line11.set_xdata(params.dz0+params.dz1)

  for i in range(2):
    for j in range(2):
      ax[i,j].relim()
      ax[i,j].autoscale_view()
  fig.canvas.draw_idle()

def update_m0(val):
  global mutex
  if mutex:
    return
  mutex = True
  params.m0 = val

  # Recalculating gravity
  g = (params.m1+1.0) * params.theta1
  params.theta0 = g / (params.m0+1.0)

  # Updating sliders
  t0_slider.set_val(params.theta0)
  
  update_plots()
  mutex = False

def update_m1(val):
  global mutex
  if mutex:
    return
  mutex = True
  params.m1 = val

  # Recalculating gravity
  g = (params.m0+1.0) * params.theta0
  params.theta1 = g / (params.m1+1.0)

  # Updating sliders
  t1_slider.set_val(params.theta1)

  update_plots()
  mutex = False

def update_m2(val):
  global mutex
  if mutex:
    return
  mutex = True
  params.m2 = val

  # Recalculating gravity
  g = (params.m0+1.0) * params.theta0
  params.theta2 = g / (params.m2+1.0)

  # Updating sliders
  t2_slider.set_val(params.theta2)

  update_plots()
  mutex = False

def update_t0(val):
  global mutex
  if mutex:
    return
  mutex = True
  params.theta0 = val

  # Recalculating gravity
  g = (params.m1+1.0) * params.theta1
  params.m0 = params.theta0 / g - 1.0

  # Updating sliders
  m0_slider.set_val(params.m0)

  update_plots()
  mutex = False

def update_t1(val):
  global mutex
  if mutex:
    return
  mutex = True
  params.theta1 = val

  # Recalculating gravity
  g = (params.m2+1.0) * params.theta2
  params.m1 = params.theta1 / g - 1.0

  # Updating sliders
  m1_slider.set_val(params.m1)

  update_plots()
  mutex = False

def update_t2(val):
  global mutex
  if mutex:
    return
  mutex = True
  params.theta2 = val

  # Recalculating gravity
  g = (params.m1+1.0) * params.theta1
  params.m2 = params.theta2 / g - 1.0

  # Updating sliders
  m2_slider.set_val(params.m2)

  update_plots()
  mutex = False


def update_g(g):
  global mutex
  if mutex:
    return
  mutex = True
  params.theta0 = g / (params.m0+1.0)
  params.theta1 = g / (params.m1+1.0)
  params.theta2 = g / (params.m2+1.0)

  t0_slider.set_val(params.theta0)
  t1_slider.set_val(params.theta1)
  t2_slider.set_val(params.theta2)

  update_plots()
  mutex = False

def update_dz0(dz):
  params.dz0 = dz
  update_plots()

def update_dz1(dz):
  params.dz1 = dz
  update_plots()

def update_dz2(dz2):
  params.dz2 = dz2
  update_plots()

def update_chi_rho(chi_rho):
  global mutex
  if mutex:
    return
  mutex = True

  params.chi_rho = chi_rho
  
  update_plots()
  mutex = False

def update_rho0(rho0):
  global mutex
  if mutex:
    return
  mutex = True

  params.rho0 = rho0
  
  update_plots()
  mutex = False

def update_scale(value):
  params.yscale = value
  update_plots()

generate_model = False
def validate(event):
  global generate_model
  generate_model = True

  plt.close('all')
  

fig.subplots_adjust(bottom=0.3, top=0.95, right=0.85)
axm0 = fig.add_axes([0.1, 0.25, 0.3, 0.03])
m0_slider = Slider(
    ax=axm0,
    label='m0',
    valmin=1.51,
    valmax=50.0,
    valinit=4.0,
)
axm1 = fig.add_axes([0.1, 0.21, 0.3, 0.03])
m1_slider = Slider(
    ax=axm1,
    label='m1',
    valmin=0.01,
    valmax=1.49,
    valinit=1.0,
)
axm2 = fig.add_axes([0.1, 0.17, 0.3, 0.03])
m2_slider = Slider(
    ax=axm2,
    label='m2',
    valmin=1.51,
    valmax=10.0,
    valinit=4.0,
)
axt0 = fig.add_axes([0.6, 0.25, 0.3, 0.03])
t0_slider = Slider(
    ax=axt0,
    label='theta0',
    valmin=1.0,
    valmax=40.0,
    valinit=4.0,
)
axt1 = fig.add_axes([0.6, 0.21, 0.3, 0.03])
t1_slider = Slider(
    ax=axt1,
    label='theta1',
    valmin=1.0,
    valmax=40.0,
    valinit=10.0,
)
axt2 = fig.add_axes([0.6, 0.17, 0.3, 0.03])
t2_slider = Slider(
    ax=axt2,
    label='theta2',
    valmin=1.0,
    valmax=40.0,
    valinit=4.0,
)
axg = fig.add_axes([0.1, 0.12, 0.3, 0.03])
g_slider = Slider(
    ax=axg,
    label='g',
    valmin=1.0,
    valmax=100.0,
    valinit=20.0,
)

axdz0 = fig.add_axes([0.6, 0.12, 0.3, 0.03])
dz0_slider = Slider(
    ax=axdz0,
    label='dz0',
    valmin=0.01,
    valmax=3.0,
    valinit=1.0,
)

axdz1 = fig.add_axes([0.1, 0.08, 0.3, 0.03])
dz1_slider = Slider(
    ax=axdz1,
    label='dz1',
    valmin=0.01,
    valmax=10.0,
    valinit=1.0,
)

axdz2 = fig.add_axes([0.6, 0.08, 0.3, 0.03])
dz2_slider = Slider(
    ax=axdz2,
    label='dz2',
    valmin=0.01,
    valmax=10.0,
    valinit=1.0,
)

axchirho = fig.add_axes([0.1, 0.04, 0.3, 0.03])
chirho_slider = Slider(
  ax=axchirho,
  label='chi_rho',
  valmin=1.1,
  valmax=100.0,
  valinit=3.0
)

axrho0 = fig.add_axes([0.6, 0.04, 0.3, 0.03])
rho0_slider = Slider(
  ax=axrho0,
  label='rho0',
  valmin=1.0,
  valmax=100.0,
  valinit=1.0
)

axscale = fig.add_axes([0.86, 0.5, 0.13, 0.1])
radio = RadioButtons(axscale, ('linear', 'log', 'log + linear T'), active=0)

axbtn = fig.add_axes([0.4, 0.0015, 0.2, 0.03])
btn = Button(axbtn, 'Generate model')

m0_slider.on_changed(update_m0)
m1_slider.on_changed(update_m1)
m2_slider.on_changed(update_m2)
t0_slider.on_changed(update_t0)
t1_slider.on_changed(update_t1)
t2_slider.on_changed(update_t2)
g_slider.on_changed(update_g)
dz0_slider.on_changed(update_dz0)
dz1_slider.on_changed(update_dz1)
dz2_slider.on_changed(update_dz2)
chirho_slider.on_changed(update_chi_rho)
rho0_slider.on_changed(update_rho0)
radio.on_clicked(update_scale)
btn.on_clicked(validate)

plt.show()

if not generate_model:
  exit(0)

stdout('2. Other parameters')
stdout('===================')

name       = inputs(' Name of the run', 'tri_layer')
tend       = inputd(' Time of end of the simulation', 500.0)
save_freq  = inputd(' Save frequency', 1.0)
check_freq = inputd(' Checkpoint frequency [only Dyablo]', 10.0)
CFL        = inputd(' CFL (Hydro and parabolic)', 0.1)
Ck         = inputd(' Ck (Thermal dissipation coefficient) : ', 0.07)
sigma      = inputd(' Sigma (Prandtl number) : ', 0.1)
A          = inputd(' Aspect ratio between horizontal axis and dz1 : ', 4.0)
Nx         = inputd(' Number of cells along the horizontal directions : ', 256)
Nz         = inputd(' Number of cells along the vertical direction : ', 256)
amr_levels = inputd(' AMR levels above level_min', 0)


# Calculating a bunch of stuff
gamma = 5.0/3.0
gval = params.theta1 * (params.m1+1.0)
z, T, rho, _, _, _ = get_profile(params)

Nz = z.shape[0]
zmax = params.dz0 + params.dz1 + params.dz2
di1 = params.dz0 * Nz / zmax
di2 = (params.dz0+params.dz1) * Nz / zmax
imid = int(0.5 * (di1+di2))
Tmid = T[imid]
rho_mid = rho[imid]
T0 = params.theta1*params.dz1 / (params.chi_rho**(1.0/params.m1) - 1.0) - params.theta0*params.dz0
T1 = T0 + params.theta0*params.dz0
T2 = T1 + params.theta1*params.dz1

rho1 = params.rho0 * (T1/T0)**params.m0

cp = gamma / (gamma-1.0)
nabla_minus_nabla_ad = params.theta1 * (1.0 - (params.m1+1)/cp)
one_over_Hp = rho1 * gval / (Tmid*T0)

Ra = rho_mid**2.0 * one_over_Hp * nabla_minus_nabla_ad * gval * params.dz1**4.0 / (sigma*Ck**2.0)
K    = Ck * cp
mu   = sigma*Ck

# Boundaries
bc_zmin = T0
bc_zmax = params.theta2

# Conductivities
K1 = 1
K0 = (params.m0+1.0)/(params.m1+1.0)
K2 = (params.m2+1.0)/(params.m1+1.0)

# Printing results
stdout(' . Layer #0: ')
stdout(f'   theta0 = {params.theta0}')
stdout(f'   m0     = {params.m0}')
stdout(f'   K0     = {K0}')
stdout(f'   dz0    = {params.dz0}')

stdout(' . Layer #1: ')
stdout(f'   theta1 = {params.theta1}')
stdout(f'   m1     = {params.m1}')
stdout(f'   K1     = {K1}')
stdout(f'   dz1    = {params.dz1}')

stdout('\n')
stdout(' . Layer #2:')
stdout(f'   theta2 = {params.theta2}')
stdout(f'   m2     = {params.m2}')
stdout(f'   K2     = {K2}')
stdout(f'   dz2    = {params.dz2}')


stdout(' . Diffusivities: ')
stdout(f'   Ck = {Ck}')
stdout(f'   K  = {K}')
stdout(f'   mu = {mu}')
stdout(f'   Ra = {Ra}')
stdout(f'   Pr = {sigma}')


stdout('\n')
stdout(' . Boundaries: ')
stdout(f'   Bottom gradient: {bc_zmax}')
stdout(f'   Top temperature: {bc_zmin}')

xmax = params.dz1 * A
z1 = params.dz0
z2 = params.dz0+params.dz1

stdout('\n')
stdout(' . Domain:')
stdout(f'   Horizontal extent         = [0.0, {xmax}]')
stdout(f'   Depth of top layer        = [0.0, {z1}]')
stdout(f'   Depth of middle layer     = [{z1}, {z2}]')
stdout(f'   Depth of bottom layer     = [{z2}, {zmax}]')
stdout(f'   Thickenn of top  layer    = {params.dz0}')
stdout(f'   Thickness of middle layer = {params.dz1}')
stdout(f'   Thickness of bottom layer = {params.dz2}')


bx = 4
cor_x = Nx // bx
cor_z = Nz // bx
level = int(max(np.log2(cor_x)+0.5, np.log2(cor_z)+0.5))
level_max = level  + amr_levels

dz = zmax / Nz
if params.dz0 < 5.0*dz:
  print(f'WARNING : Top layer thickness ({params.dz0}) smaller than 5 dz ({5.0*dz})')
if params.dz1 < 5.0*dz:
  print(f'WARNING : Middle layer thickness ({params.dz1}) smaller than 5 dz ({5.0*dz})')
if params.dz2 < 5.0*dz:
  print(f'WARNING : Bottom layer thickness ({params.dz2}) smaller than 5 dz ({5.0*dz})')

fname = name + '_dyablo.ini'
dyablo_template = ''.join(open('dyablo.template.tri_layer', 'r').readlines())
dyablo_template = dyablo_template.format(tend, save_freq, check_freq, 
                                         xmax, xmax, zmax, level, level_max, 
                                         cor_x, cor_x, cor_z,
                                         bc_zmin, bc_zmax, 
                                         params.m0, params.m1, params.m2, 
                                         params.theta0, params.theta1, params.theta2, 
                                         params.dz0, params.dz1, params.dz2, 
                                         T0, params.rho0, Ck, sigma,
                                         CFL, CFL, gval, K, mu)
f_out = open(fname, 'w')
f_out.write(dyablo_template)
f_out.close()
stdout(f' => Dyablo setup written to {fname}')


fname = name + '_raw.txt'
data_profile = np.stack((z, rho, p, T, Sen)).T
np.savetxt(fname, data_profile, header='z density pressure temperature entropy')
stdout(f' => Raw profile data written to {fname}')

stdout(f' => Log written to {log_name}')
log_out.close()


