function fig3D(Xi,Lx,Ly,Lz,NX,NY,NZ,fig)
    if(fig==1&&NZ~=1)
        dx = Lx/double(NX);
        dy = Ly/double(NY);
        dz = Lz/double(NZ);
        x1 = dx/2:dx:Lx;
        y1 = dy/2:dy:Ly;
        z1 = dz/2:dz:Lz;
        [X2,Y2,Z2] = meshgrid(x1,y1,z1);
        X=X2*0;
        k=0;
        for l=1:NZ
            for j = 1:NY
                for i = 1:NX
                    k=k+1;
                    X(j,i,l)=Xi(k);
                end
            end
        end
        yslice = [Ly/2];
        zslice = [Lz/2];
        xslice = [dx, Lx/2, Lx-dx];
        figure(2)
        slice(X2,Y2,Z2,X,xslice,yslice,zslice), shading flat;
        daspect([1 1 1]);
        xlim([0 Lx]);
        ylim([0 Ly]);
        zlim([0 Lz]);
        view(-35,20);
    end
end