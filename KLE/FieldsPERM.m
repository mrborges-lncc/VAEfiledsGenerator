clear all; close all;
addpath ./tools/
addpath ~/Dropbox/mrst-2023b/
startup

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GRID %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Lx  = 100.0;
Ly  = 100.0;
Lz  = 0.01;
nx  = 50;
ny  = 50;
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
home = '~/Dropbox/matricesKLE/';
file = {[home 'avet_exp_1_100x100x1_50x50x1_5x5x0.01_M2500.bin'],...
        [home 'avet_exp_1_100x100x1_50x50x1_10x10x0.01_M2500.bin'],...
        [home 'avet_exp_1_100x100x1_50x50x1_15x15x0.01_M2500.bin'],...
        [home 'avet_exp_1_100x100x1_50x50x1_20x20x0.01_M2500.bin'],...
        [home 'avet_exp_1_100x100x1_50x50x1_25x25x0.01_M2500.bin'],...
        [home 'avet_exp_1_100x100x1_50x50x1_30x30x0.01_M2500.bin'],...
        [home 'avet_exp_1_100x100x1_50x50x1_35x35x0.01_M2500.bin'],...
        [home 'avet_sexp_3_100x100x1_50x50x1_5x5x0.01_M2500.bin'],...
        [home 'avet_sexp_3_100x100x1_50x50x1_10x10x0.01_M2500.bin'],...
        [home 'avet_sexp_3_100x100x1_50x50x1_15x15x0.01_M2500.bin'],...
        [home 'avet_sexp_3_100x100x1_50x50x1_20x20x0.01_M2500.bin'],...
        [home 'avet_sexp_3_100x100x1_50x50x1_25x25x0.01_M2500.bin'],...
        [home 'avet_sexp_3_100x100x1_50x50x1_30x30x0.01_M2500.bin'],...
        [home 'avet_sexp_3_100x100x1_50x50x1_35x35x0.01_M2500.bin']};
MM   = {2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500,...
    2500, 2500, 2500, 2500, 2500};
nm   = {'exp5','exp10','exp15','exp20','exp25','exp30','exp35',...
    'sexp5','sexp10','sexp15','sexp20','sexp25','sexp30','sexp35'};
mu   = 0.0;
sig  = 1.0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nrand = 4;
cont = 0;
lim  = [-3.5  3.5];
for i = 1 : length(file)
    fid = fopen(file{i},"r");
    T   = fread(fid, "single");
    fclose(fid);
    M   = MM{i};
    T   = reshape(T,[M,M]);
    T   = T(1:num_elem, 1:M);
    for nr = 1 : Nrand
        cont = cont + 1;
        theta = single(lhsnorm(mu,sig,MM{1}));
        Y     = T * theta(1:M);
        fprintf('Real.: %d \t Mean: %4.2f \t Std: %4.2f\n',cont,...
            mean(Y),std(Y));
        plot_rock(Y,G,'Yn','','none',lim,[0 90],1);
        base=['figuras/Y_' nm{i} '_' num2str(nr,'%d')];
        set(gcf,'PaperPositionMode','manual','PaperPosition',...
            [0.1 0.1 5 5]);
        print('-dpng','-r600', base);
        clear Y
    end
    clear T
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
