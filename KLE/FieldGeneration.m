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
ntipo = 3;
nu = 0.5;
beta = 1;
num_elem = nx * ny;
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
M    = 300 * 100;
n    = 100 * 100;
mu   = 0.0;
sig  = 1.0;
home = '~/Dropbox/fieldsCNN/';
file = '/home/mrborges/Dropbox/fieldsCNN/avet_exp_3_1x3x0.01_100x300x1_0.2x0.2x0.001_M30000.bin';
file = '/prj/prjmurad/mrborges/Dropbox/fieldsCNN/avet_exp_1_1x3x0.01_100x300x1_0.1x0.1x0.001_M30000.bin';
file = '/prj/prjmurad/mrborges/Dropbox/fieldsCNN/avet_sexp_3_1x3x0.01_100x300x1_0.1x0.1x0.001_M30000.bin';
fid  = fopen(file,"r");
T    = fread(fid, "single");
T    = reshape(T,[M,M]);
fclose(fid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lb = '0-1x0-1';
M  = 172;
Nrand = 5000;
name2 = [tipo num2str(Lx,'%3.2f') 'x' num2str(Ly,'%3.2f') 'x' ...
    num2str(Lz,'%3.2f') '_' num2str(NX,'%d') 'x' ...
    num2str(NY,'%d') 'x' num2str(NZ,'%d') '_l' num2str(eta1,'%3.2f')...
    'x' num2str(eta2,'%3.2f') 'x' num2str(eta3,'%3.2f')];
namein= [home name2 '_' num2str(Nrand,'%d') '.mat'];
if(nz==1)
    name = ['campos/' tipo num2str(Lx,5) 'x' num2str(Ly,5) '_'...
        num2str(NX,5) 'x' num2str(NY,5) '_' lb '_M' num2str(M,'%d') '_'];
else
    name = ['campos/' tipo num2str(Lx,5) 'x' num2str(Ly,5) 'x' ...
        num2str(Lz,5) '_' num2str(NX,5) 'x' num2str(NY,5) 'x' ...
        num2str(NZ,5) '_' lb '_M' num2str(M,'%d') '_'];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fileIDin  = fopen(namein,'w');
T = T(1:n, 1:M);
for nr = 1 : Nrand
    theta = single(lhsnorm(mu,sig,M));
    Y     = T * theta(1:M);
    fprintf('Real.: %d \t Mean: %4.2f \t Std: %4.2f\n',nr,mean(Y),std(Y));
    fwrite(fileIDin ,Y ,'single');
    imprime3D(Lx,Ly,Lz,NX,NY,NZ,ntipo,beta,Y,nr,home,name,0);
    clear Y
end
fclose(fileIDin)
