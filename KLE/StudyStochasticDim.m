clear all; close all;
addpath ./tools/
addpath ~/Dropbox/mrst-2023a/
startup

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GRID %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Lx  = 1.0;
Ly  = 1.0;
Lz  = 0.01;
nx  = 100;
ny  = 100;
nz  = 1;
depth = 1e3;
eta1  = 0.2;       % correlation length in the x direction
eta2  = 0.2;       % correlation length in the y direction
eta3  = 0.001;       % correlation length in the z direction
home_fig = './figuras/';
ntipo = 1;
nu = 0.5;
beta = 1;
num_elem = nx * ny;
M = 300 * 100;
if ntipo == 1, tipo = 'exp_'; end
if ntipo == 2, tipo = 'frac_'; end
if ntipo == 3, tipo = 'sexp_'; end
if ntipo == 4, tipo = 'matern_'; end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nD = '3D';
color = 'none';
% color = 'k';
vw  = [-35 20];
vw  = [0 90];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GRID %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dx  = Lx/double(nx);
dy  = Ly/double(ny);
dz  = Lz/double(nz);
G   = cartGrid([nx ny nz],[Lx Ly Lz]*meter^3);
G.nodes.coords(:, 3) = depth + G.nodes.coords(:, 3)*meter;
G.nodes.coords(:, 2) = G.nodes.coords(:, 2)*meter;
G.nodes.coords(:, 1) = G.nodes.coords(:, 1)*meter;
G   = computeGeometry(G);
[dim, nD, fine_grid, coarse_grid, dims, meshInfo] = preproc(Lx,Ly,Lz,...
    nx,ny,nz,nx,ny,nz);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% stoch= load('out/energy_sexp_autoval_100x100x1_0-1x0-1_10000.dat');
% file = 'out/avet_sexp_3_1x1x0.01_100x100x1_0.05x0.1x0.001_M10000.bin';
% stoch= load('out/energy_exp_autoval_100x100x1_0-2x0-2_10000.dat');
home = '/prj/prjmurad/mrborges/Dropbox/fieldsCNN/';
file = [ home 'avet_exp_1_1x3x0.01_100x300x1_0.2x0.2x0.001_M30000.bin'];
stoch= load([home 'energy_exp_autoval_100x300x1_0-2x0-2_30000.dat']);
fid  = fopen(file,"r");
T    = fread(fid, "single");
fclose(fid);
T    = reshape(T,[M,M]);
T    = T(1:num_elem,:);
lambda  = load([home 'aval_exp_1_1x1x0.01_100x100x1_0.2x0.2x0.001_M10000.dat']);
lambdaB = load([home 'aval_exp_1_1x3x0.01_100x300x1_0.2x0.2x0.001_M30000.dat']);
mm = 30000;
[nom,lb] = loglambdafig([1:mm],lambdaB,mm,home_fig,nx,ny,nz,eta1,...
    eta2,eta3,beta,nu,ntipo,tipo);
name = [nom '_' num2str(nx,5) 'x' num2str(ny,5) 'x'...
    num2str(nz,5) '_' lb '_' num2str(M,5)];
% print('-depsc','-r300',name);
print('-dpng','-r300',name);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu = 0.0;
sig= 1.0;
K = 10;
m = [30000 : -K : 1].';
Tenergy = sum(lambdaB);
err     = zeros(size(m,1),1);
energy  = zeros(size(m,1),1);
for i = 1 : size(m,1)
    energy(i) = sum(lambdaB(1:m(i))) / Tenergy;
    [m(i) energy(i)]
end
theta = single(lhsnorm(mu,sig,M));
Y   = T(:,1:M) * theta(1:M);
for i = 1 : length(m)
    close all
    Xi    = T(:,1:m(i)) * theta(1:m(i));
    err(i)= norm(Y - Xi)/norm(Y);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot2Y(m,energy,err, '$\mathsf{M}$', '\textbf{Energy (\%)}',...
    '\textbf{Relative error (\%)}')
name = [home_fig 'Energy_' tipo num2str(nx,5) 'x' num2str(ny,5) 'x'...
    num2str(nz,5) '_' lb '_' num2str(M,5)];
% print('-depsc','-r300',name);
print('-dpng','-r300',name);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m  = [stoch(:,2)' M 128 256];
m  = sort(m);
str= [stoch(:,1)' 100]';
theta = single(lhsnorm(mu,sig,M));
Y   = T(:,1:M) * theta(1:M);
err = zeros(length(m),1);
for i = 1 : length(m)
    close all
    Xi    = T(:,1:m(i)) * theta(1:m(i));
    err(i)= norm(Y - Xi)/norm(Y);
    lim = [0 0];
    plot_rock(Xi,G,'Yn','$Y$',color,lim,vw,1);
    base=['figuras/Y_' tipo 'M' num2str(m(i))];
    set(gcf,'PaperPositionMode','manual',...
        'PaperPosition',[0.01 0.01 4 4.5]);
    print('-dpng','-r300', base);
    pause(0.5)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
two2Dplot(str, [err err], 'Power', 'Relative error', 'dat1', 'dat2', 56)
close all
plot2Y([stoch(:,2);M],str,err, '$\mathsf{M}$', '\textbf{Energy}',...
    '\textbf{Relative error}')
name = [home_fig 'Energy2_' tipo num2str(nx,5) 'x' num2str(ny,5) 'x'...
    num2str(nz,5) '_' lb '_' num2str(M,5)];
% print('-depsc','-r300',name);
print('-dpng','-r300',name);
clear