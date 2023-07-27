function imprime3D(Lx,Ly,Lz,nx,ny,nz,ntipo,beta,Xi,nr,home,name,iprt)
   nfile= num2str(nr-1,5);
   bb2 = [home name nfile '.dat'];
   fprintf('File name: %s\n',bb2);
   if(iprt==1)
       outfile2 = fopen(bb2, 'wt');    
        fprintf(outfile2,'%f\n',Lx);
        fprintf(outfile2,'%f\n',Ly);
        fprintf(outfile2,'%f\n',Lz);
        fprintf(outfile2,'%d\n',nx);
        fprintf(outfile2,'%d\n',ny);
        fprintf(outfile2,'%d\n',nz);
        fprintf(outfile2,'%d\n',ntipo);
        fprintf(outfile2,'%f\n',beta);
        fprintf(outfile2,'%d\n',2);
        fprintf(outfile2,'%d\n',2);

        m=0;
        for z=1:nz
            fprintf(outfile2,'%d\n',z-1);
            for i=1:ny
                fprintf(outfile2,'%d\n',i-1);
                for j=1:nx
                    m=m+1;
                    fprintf(outfile2,'%g ',Xi(m));
                end
                fprintf(outfile2,'\n192837465\n');
            end
        end
        fclose(outfile2);
   end
   if(iprt==0)
       Xi = double(Xi);
       save(bb2,'Xi','-ascii');
   end       
   return
end
%
