function [y,Lx,Ly,Lz,nx,ny,nz] = perm_reader(name,ni,dim)
 fname = [name num2str(ni,'%d') '.dat'];
 if(dim == '3D')
     [y,Lx,Ly,Lz,nx,ny,nz]= leitura3D(fname);
 end
 if(dim == '2D')
     [y,Lx,Ly,nx,ny]= leitura2D(fname);
     Lz = 0.1; nz = 1;
 end
end

