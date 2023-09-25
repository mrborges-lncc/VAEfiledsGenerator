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
clen = '02';
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
home = '/home/mrborges/Dropbox/fieldsCNN/';
file = [ home 'avet_exp_1_1x3x0.01_100x300x1_0.2x0.2x0.001_M30000.bin'];
stoch= load([home 'energy_exp_autoval_100x300x1_0-2x0-2_30000.dat']);
fid  = fopen(file,"r");
T    = fread(fid, "single");
fclose(fid);
T    = reshape(T,[M,M]);
T    = T(1:num_elem,:);
lambda  = load([home 'aval_exp_1_1x3x0.01_100x300x1_0.2x0.2x0.001_M30000.dat']);
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m  = [stoch(1,2); stoch(4,2); M];
m  = sort(m);
str= [stoch(1,1); stoch(4,1); 100];
err = zeros(length(m),1);
energy  = zeros(length(m),1);
theta = single(lhsnorm(mu,sig,M));
Y   = T(:,1:M) * theta(1:M);
normY = norm(Y);
for i = 1 : length(m)
    close all
    Xi    = T(:,1:m(i)) * theta(1:m(i));
    lim = [0 0];
    plot_rock(Xi,G,'Yn','$Y$',color,lim,vw,1);
    base=['figuras/Y_' tipo clen '_E' num2str(str(i))];
    set(gcf,'PaperPositionMode','manual',...
        'PaperPosition',[0.01 0.01 4 4.5]);
    print('-dpng','-r300', base);
    pause(0.5)
    clear Xi
end
for i = 1 : length(m)
    fprintf('M = %d \t Energy = %4.2f \t Error = %4.2f\n', m(i), energy(i), err(i))
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
