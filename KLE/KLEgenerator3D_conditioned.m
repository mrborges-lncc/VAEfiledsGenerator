%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        RANDOM FIELDS GENERATOR
%                    BASED ON KARHUNEN-LOEVE EXPANSION
% CONDITIONING BASED ON OSSIANDER et al. (2014) - Conditional Stochastic 
% Simulations of Flow and Transport with Karhunen-Loève Expansions, 
% Stochastic Collocation, and Sequential Gaussian Simulation, Journal of 
% Applied Mathematics Volume 2014, Article ID 652594, 21 pages
% http://dx.doi.org/10.1155/2014/652594
% AUTHOR: MARCIO RENTES BORGES
% DATE: 29/04/2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
addpath ./tools/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inputbox = 10; % if == 1 display a dialog box to input data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% INPUT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tStart = tic;
home     = '/media/mrborges/m4borges/fields/';
home     = './fields/';
homeT    = './out/';
%homeT    = '/media/mrborges/m4borges/kle_matrix/';
home_fig = './figuras/';
homep    = './paraview/';
%
%** Determina o tipo de campo que sera gerado (ntipo==1 => exponencial)****
%** (ntipo==2 => fractal) *************************************************
%** (ntipo==3 => exponencial 2) *******************************************
% ntipo = input('Digite 1 para campos exponenciais ou 2 para fractais :');
graf=1;
fig=1;
%** Determina se a covariancia dos campos sera verificada (band_cov==1)****
band_cov = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%*************************************************************************
%%* ENTRADA DE DADOS ******************************************************
if(inputbox==1)
    [Lx,Ly,Lz,nx,ny,nz,NX,NY,NZ,eta1,eta2,eta3,beta,nu,...
        varY,Nrand,interpolacao,M,ntipo,TIPOINPUT,...
        file_input_cond]=finputbox();
else
    ntipo = 3; % 1 == exponential, 3 == square exponential %%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%% physical dimensions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Lx = 1.0;
    Ly = 1.0;
    Lz = 0.06;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%% mesh for covariance matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nx = 50;
    ny = 50;
    nz = 3;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Mesh for interpolation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    NX = 34;
    NY = 34;
    NZ = 50;
    interpolacao = 10; % if == 1 the eigenvector are interpolated to this mesh
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    eta1  = 0.10;       % correlation length in the x direction
    eta2  = 0.10;       % correlation length in the y direction
    eta3  = 0.05;       % correlation length in the z direction
    Nrand = 20000;      % total number of realizations
    M     = 0;      % number of terms used in the KL expansion. OBS: if == 0 it 
                       % uses the maximum number of terms (nx^2 x ny^2 x nz^2)
    TIPOINPUT = 10;     % if == 1 reads the conditioned points from the file
                       % indicated in "file_input_cond"
%     file_input_cond = './in/inputdata.dat';
    file_input_cond = './in/inputdata.dat';
%     file_input_cond = './in/inputdataSYNTY.dat';
    varY  = 1.0;           % field variance
    beta  = 0.25;          % if ntipo == 2 this is the Hurst coefficient
    nu    = 0.5;           % Matern coeff
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Variables Adjustment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
cutoff   = Lx/double(nx); % cutoff used when ntipo == 2
alpha    = 1.0;           % KEEP == 1
tipo_prt = 4;             % if == 1 print the fields in the LNCC format,
                          % if == 0 print in the UTDallas simulator format
                          % if == 3 print binary
                          % if == 4 Neural Network
                          % otherwise print both formats
paraview_print = 10;      % if == 1 print paraview visualization
printa         = 1;       % if == 1 save the T matrix = sqrt(lambda)*phi
printabin      = 1;       % if == 1 save the T in a binary file
print2python   = false;
estatistica    = 10;
if ntipo == 1, tipo = 'exp_'; end
if ntipo == 2, tipo = 'frac_'; end
if ntipo == 3, tipo = 'sexp_'; end
if ntipo == 4, tipo = 'matern_'; end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Interpolation parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[nx,ny,nz,NX,NY,NZ,M,tEINT] = preInterpolation(interpolacao,...
    nx,ny,nz,NX,NY,NZ,M);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%** CONDITIONING INPUT ****************************************************
% vet(n,i) coordenada i da posicao do dado condicionado n
[vet, dados, n_dados] = inputConditioning(file_input_cond, TIPOINPUT);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Print informations ******************************************************
informacoes3D(Lx,Ly,Lz,NX,NY,NZ,vet,dados,beta,[eta1 eta2 eta3],ntipo);
%**************************************************************************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%** COVARIANCE MATRIX *****************************************************
%**************************************************************************
disp('------------------------------');
disp('------------------------------');
disp('BILDING THE COVARIANCE MATRIX');
tSMatrix=tic;
% constroi a matriz do problema de autovalores 
% Coordinates matrix:
% coord(1:3,j): (x,y,z)- coordinate of the element center
[C, sigma, num_elem] = bildMatrix(Lx,Ly,Lz,nx,ny,nz,eta1,eta2,eta3,...
    beta,nu,ntipo,varY,interpolacao);
tEMatrix=toc(tSMatrix);
disp(['C matrix done: ' num2str(tEMatrix) ' seg.']);
disp('------------------------------');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EIGEN-PAIRS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%**************************************************************************
disp('------------------------------');
disp('COMPUTING THE EIGENVALUES AND EIGENVECTORS');
tSauto=tic;
%**************************************************************************
[lambda, phi] = EigenPairs(C);
tEauto=toc(tSauto);
disp(['Eigenpairs computation done: ' num2str(tEauto) ' seg.']);
disp('------------------------------');
clear C
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TOL = 1.0E-10;
lambda(find(lambda < TOL)) = 0.0;
[nom,lb] = loglambdafig([1:num_elem],lambda,M,home_fig,nx,ny,nz,eta1,...
    eta2,eta3,beta,nu,ntipo,tipo);
name = [nom '_' num2str(NX,5) 'x' num2str(NY,5) 'x'...
    num2str(NZ,5) '_' lb '_' num2str(M,5)];
% print('-depsc','-r300',name);
print('-dpng','-r300',name);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Define the names of the eigenpairs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[name_autovet, name_autoval, name_mutilde, name_MMat] = ...
    fnames(homeT,ntipo,tipo,beta,eta1,eta2,eta3,Lx,Ly,Lz,NX,NY,NZ,M);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% INTERPOLATION OF PHI MATRIX TO THE MESH NX x NY x NZ %%%%%%%%%%%%%%%%%%
if interpolacao == 1
    disp('------------------------------');
    disp('INTERPOLATION OF PHI MATRIX');
    tINTERP=tic;
    [phi,num_elem] = interpMatrixT(phi,M,interpolacao,Lx,Ly,Lz,...
        nx,ny,nz,NX,NY,NZ);
    tINTERPT=toc(tINTERP);
    disp(['Interpolation done: ' num2str(tINTERPT) ' seg.']);
    disp('------------------------------');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CONDITIONING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%**************************************************************************
[pnode] = InfoConditioning(TIPOINPUT, vet, n_dados, Lx, Ly, Lz, NX, NY, NZ);
%**************************************************************************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Matrizes para o novo condicionamento %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%**************************************************************************
if TIPOINPUT == 1
    disp('------------------------------');
    disp('COMPUTING THE CONDITIONING MATRIX');
    tSC=tic;
    [MMat, mutilde] = CONDmatrix(phi,lambda,M,n_dados,pnode,dados);
    tEauto=toc(tSC);
    disp(['MMat and mutilde computed: ' num2str(tEauto) ' seg.']);
    disp('------------------------------');
end
%**************************************************************************
if(printa == 1 && TIPOINPUT == 1)
    disp('SAVING THE PROJECTION');
    tSave=tic;
    mu = double(mutilde);
    save(name_mutilde,'mu','-ascii');
    clear mu
    if printabin == 1
        name_MMat = [name_MMat(1:end-3) 'bin'];
        fileID=fopen(name_MMat,'w+','l');
        fwrite(fileID,reshape(MMat,1,M*M),'single');
        fclose(fileID);
    else
        save(name_MMat,'MMat','-ascii');
    end
    tEsave=toc(tSave);
    disp(['Saved: ' num2str(tEsave) ' seg.']);
    disp('------------------------------');
else
    tEsave=0.0;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%******** MATRIZ Theta*sqrt(lambda) ***************************************
disp('------------------------------');
disp('BILDING T MATRIX');
tMatT=tic;
nome= [homeT 'energy_' name(11:end) '.dat'];
phi = TlambdaMatrix(phi,lambda,M,num_elem,nome);
tEMatT=toc(tMatT);
disp(['Matrix T done: ' num2str(tEMatT) ' seg.']);
disp('------------------------------');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%**** IMPRESSAO DOS AUTOPARES *********************************************
%**************************************************************************
if(printa==1)
    disp('SAVING THE EIGENPAIRS');
    tSave=tic;
    lamb=double(lambda(1:M));
    save(name_autoval,'lamb','-ascii');
    clear lamb
    if printabin == 1
        name_autovet= [name_autovet(1:end-3) 'bin']
        if print2python
            save(name_autovet, 'phi', '-v7');
        else
            fileID=fopen(name_autovet,'w+','l');
            fwrite(fileID,reshape(phi(:,1:M),[1,num_elem*M]),'single');
            fclose(fileID);
        end
    else
        save(name_autovet,'phi','-ascii');
    end
    tEsave=toc(tSave);
    disp(['Eigenpairs saved: ' num2str(tEsave) ' seg.']);
    disp('------------------------------');
else
    tEsave=0.0;
end
%**************************************************************************
if tipo_prt == 4
    name2 = [tipo num2str(Lx,'%3.2f') 'x' num2str(Ly,'%3.2f') 'x' ...
        num2str(Lz,'%3.2f') '_' num2str(NX,'%d') 'x' ...
        num2str(NY,'%d') 'x' num2str(NZ,'%d') '_l' num2str(eta1,'%3.2f')...
        'x' num2str(eta2,'%3.2f') 'x' num2str(eta3,'%3.2f')];
    namein    = [home name2 '_' num2str(Nrand,'%d') '.mat'];
    fileIDin  = fopen(namein,'w');
    fprintf('\n ===============================================================')
    fprintf('\n Output File: %s\n',namein);
    fprintf('\n ===============================================================\n')
end

%**************************************************************************
%**************************************************************************
%**************************************************************************
%**** LOOP TO REALIZATIONS ************************************************
%**************************************************************************
disp('------------------------------');
disp('FILED GENERATION');
fcond = '';
if n_dados > 0, fcond = 'cond_'; end
if(nz==1)
    name = [tipo fcond num2str(Lx,5) 'x' num2str(Ly,5) '_'...
        num2str(NX,5) 'x' num2str(NY,5) '_' ...
        lb '_'];
else
    name = [tipo fcond num2str(Lx,5) 'x' num2str(Ly,5) 'x' num2str(Lz,5) '_'...
        num2str(NX,5) 'x' num2str(NY,5) 'x' num2str(NZ,5) '_'...
        lb '_'];
end
%
%**************************************************************************
%******** LOOP sobre as realizacoes ***************************************
%**************************************************************************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MAIN LOOP OVER REALIZATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%**************************************************************************
mu  = 0.0;
sig = 1.0;
Xi  = single(zeros(num_elem,1));
mY  = single(zeros(num_elem,1));
X   = [];
corretor = 0.0;
tCOND=0.0;
THETA = [];
tSgera = tic;
for nr=1:Nrand
    fprintf('---------------------------------------------------\n');
    fprintf('Realization: %d\n',nr);
%******** CONDICIONAMENTO DO CAMPO ****************************************
    if(n_dados>0)
        theta = single(mutilde+MMat*lhsnorm(mu,sig,M));
    else
        theta = single(lhsnorm(mu,sig,M));
    end
%     Xi = mY + sum(phi.*theta',2);
    Xi = mY + phi * theta;
    if(estatistica==1)
        X = [X; Xi];
        THETA=[THETA; theta];
        fprintf('Mean = %4.2g \t Variance = %4.2g\n',mean(Xi),var(Xi));
    end
%******* impressao dos campos *********************************************
    if tipo_prt == 4
        fwrite(fileIDin ,Xi ,'single');
    else
        if(nz==1000)
            imprime(Lx,Ly,NX,NY,ntipo,beta,Xi,nr,home,name,tipo_prt);
        else
    %         beta = 5.13678e-13;
    %         rho  = 0.396281;
            imprime3D(Lx,Ly,Lz,NX,NY,NZ,ntipo,beta,Xi,nr,home,name,tipo_prt);
    %         imprime3D(Lx,Ly,Lz,NX,NY,NZ,ntipo,beta,[beta.*exp(rho*Xi)],...
    %             nr,home,['k_' name],tipo_prt);
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PARAVIEW PRINTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(paraview_print==1)
        filename = [homep 'field_' name 'M' num2str(M,5)...
            '-' num2str(nr-1,5)];
        paraviewprinter(Lx,Ly,Lz,nx,ny,nz,Xi,filename)
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

if tipo_prt == 4, fclose(fileIDin); end

tEgera=toc(tSgera);
tElapsed=toc(tStart);
disp(['Tempo total gasto na geraccao dos campos: ' num2str(tEgera) ' seg.'])
disp('------------------------------');

if(estatistica==1)
    fprintf('Mean of Y.......: %f\n',mean(X));
    fprintf('Variance of Y...: %f\n',var(X));
    fprintf('Minimum of Y....: %f\n',min(X));
    fprintf('Maximum of Y....: %f\n',max(X));
end

fig3D(Xi,Lx,Ly,Lz,NX,NY,NZ,fig)
name = [home_fig tipo 'field_' num2str(NX,5),...
    'x' num2str(NY,5) 'x' num2str(NZ,5) '_' lb '_' num2str(M,5)];
print('-depsc','-r300',name);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('############################################');
disp(['TOTAL TIME: ' num2str(tElapsed) ' seg.']);
disp('############################################');
disp(['PERCENTAGE OF TOTAL TIME SPENT TO CONSTRUCTION OF THE C MATRIX: '...
    num2str(100*tEMatrix/tElapsed) ' %']);
disp('############################################');
disp(['PERCENTAGE OF TOTAL TIME SPENT TO COMPUTE THE EIGENPAIRS: '...
    num2str(100*tEauto/tElapsed) ' %']);
disp('############################################');
disp(['PERCENTAGE OF TOTAL TIME SPENT TO CONSTRUCTION OF THE T MATRIZ (phi*sqrt(lambda)): '...
    num2str(100*tEMatT/tElapsed) ' %']);
disp('############################################');
disp(['PERCENTAGE OF TOTAL TIME SPENT TO INTERPOLATION: '...
    num2str(100*tEINT/tElapsed) ' %']);
disp('############################################');
disp(['PERCENTAGE OF TOTAL TIME SPENT TO SAVE THE T MATRIX: '...
    num2str(100*tEsave/tElapsed) ' %']);
disp('############################################');
disp(['PERCENTAGE OF THE TOTAL TIME SPENT TO GENERATE THE FIELDS: '...
    num2str(100*tEgera/tElapsed) ' %']);
disp('############################################');
%**************************************************************************
if(estatistica==1)
    NORMAL(THETA,mean(THETA),sqrt(var(THETA)),'$\theta$');
    name = [home_fig 'distr_' tipo '_field_' num2str(NX,5),...
        'x' num2str(NY,5) 'x' num2str(NZ,5) '_' num2str(M,5)];
    print('-dpng','-r300',name);
%     print('-depsc','-r300',name);
end
clear;