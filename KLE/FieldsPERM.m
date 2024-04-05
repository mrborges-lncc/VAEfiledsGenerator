clear all; close all;
addpath ./tools/
addpath ~/Dropbox/mrst-2023b/
startup

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GRID %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Lx  = 101.0;
Ly  = 101.0;
Lz  = 0.01;
nx  = 101;
ny  = 101;
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
file = {[home 'avet_exp_1_101x250x1_101x250x1_10x10x0.01_M25250.mat'],...
        [home 'avet_exp_1_101x250x1_101x250x1_15x15x0.01_M25250.mat'],...
        [home 'avet_exp_1_101x250x1_101x250x1_20x20x0.01_M25250.mat'],...
        [home 'avet_exp_1_101x250x1_101x250x1_25x25x0.01_M25250.mat'],...
        [home 'avet_exp_1_101x250x1_101x250x1_30x30x0.01_M25250.mat'],...
        [home 'avet_exp_1_101x250x1_101x250x1_35x35x0.01_M25250.mat']};
% file = {[home 'avet_sexp_3_1x1x0.01_100x100x1_0.2x0.2x0.001_M10000.bin'],...
%     [home 'avet_sexp_3_1x1x0.01_100x100x1_0.1x0.1x0.001_M10000.bin'],...
%     [home 'avet_sexp_3_1x1x0.01_100x100x1_0.3x0.3x0.001_M10000.bin'],...
%     [home 'avet_sexp_3_1x1x0.01_100x100x1_0.05x0.05x0.001_M10000.bin']};
MM   = {25250, 25250, 25250, 25250, 25250, 25250};
mu   = 0.0;
sig  = 1.0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nrand = 1;
cont = 0;
lim  = [-4 4];
theta = single(lhsnorm(mu,sig,MM{1}));
for i = 1 : length(file)
    fid = fopen(file{i},"r");
    T   = fread(fid, "single");
    fclose(fid);
    M   = MM{i};
    T   = reshape(T,[M,M]);
    T   = T(1:num_elem, 1:M);
    for nr = 1 : Nrand
        cont = cont + 1;
        Y     = T * theta(1:M);
        fprintf('Real.: %d \t Mean: %4.2f \t Std: %4.2f\n',cont,...
            mean(Y),std(Y));
        plot_rock(Y,G,'Yn','$Y$','none',lim,[0 90],1);
        base=['figuras/Y_' num2str(i,'%d') '_' num2str(nr,'%d')];
        set(gcf,'PaperPositionMode','auto');
        print('-dpng','-r600', base);
        clear Y theta
    end
    clear T
end
fclose(fileIDin);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
