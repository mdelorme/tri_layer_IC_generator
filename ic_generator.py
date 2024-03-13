import numpy as np
import matplotlib.pyplot as plt
log_name = 'log.txt'
log_out = open(log_name, 'w')
def stdout(*args):
  line = ' '.join(args)
  print(line)
  log_out.write(line + '\n')

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


# Constants
gamma = 5.0/3.0
mad   = 1.0/(gamma-1.0)

stdout('1. Input parameters')
stdout('===================')

stdout(' . Run info : ')
tend       = inputd(' Time of end of the simulation', 500.0)
save_freq  = inputd(' Save frequency', 1.0)
check_freq = inputd(' Checkpoint frequency [only Dyablo]', 10.0)
CFL        = inputd(' CFL (Hydro and parabolic)', 0.1)
name       = inputs(' Name of the run', 'tri_layer')

stdout('\n')
stdout(' . Physical parameters : ')
theta1  = inputd('Theta1 (temperature gradient in the central layer)', 10)
m1      = inputd('m1 (Polytropic index in the central layer)', 1.0)
K1      = inputd('K1 (Kappa relative in the central layer)', 1.0)
S       = inputd('S (Stiffness)', 5.0)
dz1     = inputd('dz1 (Central layer width)', 1.0)
dz2     = inputd('dz2 (Top/Bottom layer width)', 1.0)
sigma   = inputd('sigma (Prandtl number)', 0.1)
Ck      = inputd('Ck (Thermal diffusivity normalized [K cp]) : ', 0.07)
rho0    = inputd('rho0 (Density at the top of the box)', 1.0)
chi_rho = inputd('Chi_rho (density contrast between the top and the bottom of the central layer)', 3.0)
Nz      = inputi('Nz (number of points along vertical direction)', 256)
Nx      = inputi('Nx (number of points along horizontal direction)', 128)
xmax    = inputd('xmax (Horizontal size of the domain)', 4.0)

stdout('\n\n')
stdout('2. Derived values')
stdout('=================')

# Calculating the properties of the stable layers
m2      = S*(mad-m1) + mad
K2      = K1 * (m2+1.0)/(m1+1.0)
theta2  = theta1*K1/K2

# Polytropic model
T0 = (theta1*dz1 + theta2*dz2) / (chi_rho-1.0)
T1 = T0 + theta2*dz2
T2 = T1 + theta1*dz1
rho1 = rho0 * (T1/T0)**m2
rho2 = rho1 * (T2/T1)**m1
p0   = rho0*T0
p1   = rho1*T1
p2   = rho2*T2
gval = (m1+1.0) * theta1

# Domain
z1   = dz2
z2   = z1+dz1
zmax = z2+dz2

# Diffusivities
zmid    = 0.5 * (z1+z2)
Tmid    = T1 + (zmid-z1) * theta1
rho_mid = rho1 * (Tmid/T1)**m1

cp = gamma / (gamma-1.0)
nabla_minus_nabla_ad = theta1 * (1.0 - (m1+1)/cp)
one_over_Hp = rho1 * gval / (Tmid*T0)
Ra = rho_mid**2.0 * one_over_Hp * nabla_minus_nabla_ad * gval * dz1**4.0 / (sigma*Ck**2.0)
K    = Ck * cp
mu   = sigma*Ck


# Boundaries
bc_zmin = T0
bc_zmax = theta2

stdout(' . Convective layer: ')
stdout(f'   theta1 = {theta1}')
stdout(f'   m1     = {m1}')
stdout(f'   K1     = {K1}')
stdout(f'   dz1    = {dz1}')

stdout('\n')
stdout(' . Stable layers:')
stdout(f'   theta2 = {theta2}')
stdout(f'   m2     = {m2}')
stdout(f'   K2     = {K2}')
stdout(f'   dz2    = {dz2}')

stdout('\n')
stdout(' . Polytropic models:')
stdout(f'   rho0    = {rho0}')
stdout(f'   T0      = {T0}')
stdout(f'   p0      = {p0}')
stdout(f'   rho1    = {rho1}')
stdout(f'   T1      = {T1}')
stdout(f'   p1      = {p1}')
stdout(f'   rho2    = {rho2}')
stdout(f'   T2      = {T2}')
stdout(f'   p2      = {p2}')
stdout(f'   gz      = {gval}')
stdout(f'   chi_rho = {chi_rho}')
stdout(f'   S       = {S}')
stdout('\n')
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

stdout('\n')
stdout(' . Domain:')
stdout(f'   Horizontal extent         = [0.0, {xmax}]')
stdout(f'   Depth of top layer        = [0.0, {z1}]')
stdout(f'   Depth of middle layer     = [{z1}, {z2}]')
stdout(f'   Depth of bottom layer     = [{z2}, {zmax}]')
stdout(f'   Width of top/bottom layer = {dz2}')
stdout(f'   Width of middle layer     = {dz1}')
stdout(f'   Nx;Ny                     = {Nx}')
stdout(f'   Nz                        = {Nz}')

stdout('\n\n')
stdout('3. Plotting ICs')
stdout('===============')

z = np.linspace(0.0, zmax, Nz)
dom1 = z < z1
dom2 = (z >= z1) & (z < z2)
dom3 = z >= z2

rho = np.empty_like(z)
T   = np.empty_like(z)
p   = np.empty_like(z)

T[dom1] = T0 + z[dom1] * theta2
T[dom2] = T1 + (z[dom2]-z1) * theta1
T[dom3] = T2 + (z[dom3]-z2) * theta2

rho[dom1] = rho0 * (T[dom1]/T0)**m2
rho[dom2] = rho1 * (T[dom2]/T1)**m1
rho[dom3] = rho2 * (T[dom3]/T2)**m2

p[dom1] = p0 * (T[dom1]/T0)**(m2+1)
p[dom2] = p1 * (T[dom2]/T1)**(m1+1)
p[dom3] = p2 * (T[dom3]/T2)**(m2+1)

R = 1.0
Sen = gamma / (gamma-1.0) * np.log(T) - R*np.log(p)
dSdr = np.gradient(Sen, z[0]-z[1])

fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ax[0,0].plot(z, T, '-r')
ax[0,1].plot(z, rho, '-r')
ax[1,0].plot(z, p, '-r')
ax[1,1].plot(z, dSdr, '-r')

for i in range(2):
  for j in range(2):
    ax[i,j].axvline(z1, linestyle=':', color='k')
    ax[i,j].axvline(z2, linestyle=':', color='k')

ax[1,0].set_xlabel('z (depth)')
ax[1,1].set_xlabel('z (depth)')
ax[0,0].set_ylabel('Temperature')
ax[0,1].set_ylabel('Density')
ax[1,0].set_ylabel('Pressure')
ax[1,1].set_ylabel('Entropy gradient')
title = r'Tri-Layer initial conditions for $\theta_1={:.1f}$, $Ra={:.2e}$, $Pr={:.3f}$, $C_k={:.3f}$'.format(theta1, Ra, sigma, Ck)
plt.suptitle(title)
plt.tight_layout()
fname = 'initial_profile.png'
plt.savefig(fname)
stdout(f' . Initial profile saved to {fname}')

cs = np.sqrt(gamma * p / rho)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(z, cs)
ax.set_xlabel('z (depth)')
ax.set_ylabel('Speed of sound')
plt.tight_layout()
fname = 'speed_of_sound.png'
plt.savefig(fname)
stdout(f' . Speed of sound profile saved to {fname}')

stdout('\n\n\n')
stdout('4. Generating IC files')
stdout('======================')

# fv2d
fname = name + '_fv2d.ini'
fv2d_template = ''.join(open('fv2d.template', 'r').readlines())
fv2d_template = fv2d_template.format(Nx, Nz, xmax, zmax, tend, save_freq, 
                                     CFL, gval, z1, z2, K1, K2, T0, rho0,
                                     m1, m2, theta1, theta2, K, bc_zmin,
                                     bc_zmax, mu)
f_out = open(fname, 'w')
f_out.write(fv2d_template)
f_out.close()
stdout(f' => FV2D setup written to {fname}')

# dyablo
bx = 4
cor_x = Nx // bx
cor_z = Nz // bx
level = int(max(np.log2(cor_x)+0.5, np.log2(cor_z)+0.5))

fname = name + '_dyablo.ini'
dyablo_template = ''.join(open('dyablo.template', 'r').readlines())
dyablo_template = dyablo_template.format(tend, save_freq, check_freq, 
                                         xmax, xmax, zmax, level, level, 
                                         cor_x, cor_x, cor_z,
                                         bc_zmin, bc_zmax, K1, K2, m1, m2, theta1,
                                         theta2, z1, z2, T0, rho0,
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
