clear all; close all;
addpath ./tools/
addpath ~/Dropbox/mrst-2023b/
startup

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GRID %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Lx  = 100.0;
Ly  = 100.0;
Lz  = 0.01;
nx  = 100;
ny  = 100;
nz  = 1;
NX = nx; NY = ny; NZ = nz;
depth = 1e3;
eta1  = 10.0;       % correlation length in the x direction
eta2  = 10.0;       % correlation length in the y direction
eta3  = 0.001;       % correlation length in the z direction
home_fig = './figuras/';
num_elem = nx * ny;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
home = '/media/mrborges/borges/fieldsCNN/';
file = {[home 'aval_exp_1_100x250x1_100x250x1_5x5x0.01_M25000.dat'],...
        [home 'aval_exp_1_100x250x1_100x250x1_10x10x0.01_M25000.dat'],...
        [home 'aval_exp_1_100x250x1_100x250x1_15x15x0.01_M25000.dat'],...
        [home 'aval_exp_1_100x250x1_100x250x1_20x20x0.01_M25000.dat'],...
        [home 'aval_exp_1_100x250x1_100x250x1_25x25x0.01_M25000.dat'],...
        [home 'aval_exp_1_100x250x1_100x250x1_30x30x0.01_M25000.dat'],...
        [home 'aval_exp_1_100x250x1_100x250x1_35x35x0.01_M25000.dat'],...
        [home 'aval_sexp_3_100x100x1_100x100x1_5x5x0.01_M10000.dat'],...
        [home 'aval_sexp_3_100x100x1_100x100x1_10x10x0.01_M10000.dat'],...
        [home 'aval_sexp_3_100x100x1_100x100x1_15x15x0.01_M10000.dat'],...
        [home 'aval_sexp_3_100x100x1_100x100x1_20x20x0.01_M10000.dat'],...
        [home 'aval_sexp_3_100x100x1_100x100x1_25x25x0.01_M10000.dat'],...
        [home 'aval_sexp_3_100x100x1_100x100x1_30x30x0.01_M10000.dat'],...
        [home 'aval_sexp_3_100x100x1_100x100x1_35x35x0.01_M10000.dat']};
MM   = {25000, 25000, 25000, 25000, 25000, 25000, 25000, 5000, 5000,...
    5000, 5000, 5000, 5000, 5000};
eta = {[5.0, 5.0, 0.01], [10.0, 10.0, 0.01], [15.0, 15.0, 0.01],...
       [20.0, 20.0, 0.01], [25.0, 25.0, 0.01], [30.0, 30.0, 0.01],...
       [35.0, 35.0, 0.01], [5.0, 5.0, 0.01], [10.0, 10.0, 0.01],...
       [15.0, 15.0, 0.01], [20.0, 20.0, 0.01], [25.0, 25.0, 0.01],...
       [30.0, 30.0, 0.01], [35.0, 35.0, 0.01]};
ntipo = [1,1,1,1,1,1,1,3,3,3,3,3,3,3];
mu   = 0.0;
sig  = 1.0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nrand = 1;
cont = 0;
beta = 0.0; nu = 0.5;
lim  = [-3.5  3.5];
theta = single(lhsnorm(mu,sig,MM{1}));
for i = 1 : length(file)/2
    eta1 = eta{i}(1); eta2 = eta{i}(2); eta3 = eta{i}(3);
    if ntipo(i) == 1, tipo = 'exp_'; end
    if ntipo(i) == 3, tipo = 'sexp_'; end
    M = MM{i};
    L = load(file{i}); L = L([1:M]);
    [nom,lb] = loglambdafig([1:M],L,M,home_fig,nx,ny,nz,eta1,...
    eta2,eta3,beta,nu,ntipo(i),tipo);
    name = [nom '_' num2str(NX,5) 'x' num2str(NY,5) 'x'...
    num2str(NZ,5) '_' lb '_' num2str(M,5)]
    set(gcf,'PaperPositionMode','manual','PaperPosition',[0.1 0.1 8.5 4.5]);
    print('-dpng','-r300',name);
    close all
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
