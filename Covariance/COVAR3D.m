%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% COVARIANCIA DOS CAMPOS 3D %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
clear
close all
addpath ./tools/

Lx  = 1;
Ly  = 1;
Lz  = 0.01;
nx  = 28;
ny  = 28;
nz  = 1;
tipo_prt = 0;
ntipo = 3;
beta  = 0.5;
Nrand = 1000;
home ='../KLE/fields/';
% home ='../fields/';
name = 'field_PERM_'
nameout='./out/';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X    = zeros(nx*ny*nz,1);
covx = zeros(floor(nx/2)+1,1);
covy = zeros(floor(ny/2)+1,1);
covz = zeros(floor(nz/2)+1,1);
contx= zeros(floor(nx/2)+1,1);
conty= zeros(floor(ny/2)+1,1);
contz= zeros(floor(nz/2)+1,1);
med=0.0;
Y=[];
for i=1:Nrand
    nr=i-1;
    X=leitura3D(Lx,Ly,Lz,nx,ny,nz,ntipo,beta,nr,home,name,tipo_prt);
    Y=[Y;X];
    med=med+mean(X);
end
med =mean(Y);
vari=var(Y);
clear Y X
%
for i=1:Nrand
    nr=i-1;
    X=leitura3D(Lx,Ly,Lz,nx,ny,nz,ntipo,beta,nr,home,name,tipo_prt);
    m=0;
    dx=Lx/nx;
    dy=Ly/ny;
    dz=Lz/nz;
    vx=[0.*dx:dx:Lx/2]';
    vy=[0.*dy:dy:Ly/2]';
    vz=[0.*dz:dz:Lz/2]';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% covariancia na direcao x %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i=0:floor(nx/2)
        for z=1:nz
            for k=1:ny
                for j=1:nx-i
                    loc=j+(k-1)*nx+(z-1)*nx*ny;
                    covx(i+1)=covx(i+1)+(X(loc)-med)*(X(loc+i)-med);
                    contx(i+1)=contx(i+1)+1;
                end
            end
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% covariancia na direcao y %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i=0:floor(ny/2)
        for z=1:nz
            for k=1:nx
                for j=1:ny-i
                    loc=(j-1)*nx+1+(k-1)+(z-1)*nx*ny;
                    covy(i+1)=covy(i+1)+(X(loc)-med)*(X(loc+i*nx)-med);
                    conty(i+1)=conty(i+1)+1;
                end
            end
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% covariancia na direcao z %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i=0:floor(nz/2)
        for z=1:ny
            for k=1:nx
                for j=1:nz-i
                    loc=k+(j-1)*nx*ny+(z-1)*nx;
                    loc2=loc+(nx*ny)*i;
                    covz(i+1)=covz(i+1)+(X(loc)-med)*(X(loc2)-med);
                    contz(i+1)=contz(i+1)+1;
                end
            end
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
covx=covx./contx;
covy=covy./conty;
covz=covz./contz;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(length(covx)>length(vx))
    covx=covx(1:length(vx));
end
if(length(covy)>length(vy))
    covy=covy(1:length(vy));
end
if(length(covz)>length(vz))
    covz=covz(1:length(vz));
end
if(length(vx)>length(covx))
    vx=vx(1:length(covx));
end
if(length(vy)>length(covy))
    vy=vy(1:length(covy));
end
if(length(vz)>length(covz))
    vz=vz(1:length(covz));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cx=[vx covx];
cy=[vy covy];
cz=[vz covz];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1)
plot(vx,covx,'+',vy,covy,'o',vz,covz,'s')
if(ntipo==1)
    pref='g';
end
if(ntipo==2)
    pref='g';
end
if(ntipo==3)
    pref='gs';
end

save([nameout pref name num2str(Nrand) 'x.dat'],'cx','-ascii')
save([nameout pref name num2str(Nrand) 'y.dat'],'cy','-ascii')
save([nameout pref name num2str(Nrand) 'z.dat'],'cz','-ascii')

fprintf('Mean.......: %f\n',med)
fprintf('Variance...: %f\n',vari)

clear all