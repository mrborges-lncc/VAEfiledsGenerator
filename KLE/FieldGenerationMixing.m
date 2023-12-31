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
NX = nx; NY = ny; NZ = nz;
depth = 1e3;
eta1  = 0.1;       % correlation length in the x direction
eta2  = 0.1;       % correlation length in the y direction
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
home = '~/Dropbox/fieldsCNN/';
file = {[home 'avet_exp_1_1x3x0.01_100x300x1_0.1x0.1x0.001_M30000.bin'],...
    [home 'avet_exp_1_1x3x0.01_100x300x1_0.2x0.2x0.001_M30000.bin'],...
    [home 'avet_exp_1_1x3x0.01_100x300x1_0.3x0.3x0.001_M30000.bin'],...
    [home 'avet_exp_1_1x3x0.01_100x300x1_0.4x0.4x0.001_M30000.bin']};
% file = {[home 'avet_sexp_3_1x1x0.01_100x100x1_0.2x0.2x0.001_M10000.bin'],...
%     [home 'avet_sexp_3_1x1x0.01_100x100x1_0.1x0.1x0.001_M10000.bin'],...
%     [home 'avet_sexp_3_1x1x0.01_100x100x1_0.3x0.3x0.001_M10000.bin'],...
%     [home 'avet_sexp_3_1x1x0.01_100x100x1_0.05x0.05x0.001_M10000.bin']};
MM   = {30000, 30000, 30000, 30000, 10000, 30000, 30000, 30000, 30000};
mu   = 0.0;
sig  = 1.0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nrand = 25000;
name2 = ['mix_' num2str(Lx,'%3.2f') 'x' num2str(Ly,'%3.2f') 'x' ...
    num2str(Lz,'%3.2f') '_' num2str(NX,'%d') 'x' ...
    num2str(NY,'%d') 'x' num2str(NZ,'%d')];
namein= [home name2 '_' num2str(Nrand*length(file),'%d') '.mat']
fileIDin  = fopen(namein,'w');
cont = 0;
for i = 1 : length(file)
    fid = fopen(file{i},"r");
    T   = fread(fid, "single");
    fclose(fid);
    M   = MM{i};
    T   = reshape(T,[M,M]);
    T   = T(1:num_elem, 1:M);
    for nr = 1 : Nrand
        cont = cont + 1;
        theta = single(lhsnorm(mu,sig,M));
        Y     = T * theta(1:M);
        fprintf('Real.: %d \t Mean: %4.2f \t Std: %4.2f\n',cont,mean(Y),std(Y));
        fwrite(fileIDin ,Y ,'single');
        clear Y theta
    end
    % clear T
end
fclose(fileIDin);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
