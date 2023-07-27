function informacoes3D(Lx,Ly,Lz,nx,ny,nz,v,d,BETA,ETA,nt)
fprintf('\n------------------------------')
fprintf('\n----- Covariance function ----\n')
if nt==1
    fprintf('\nExponential fields')
    fprintf('\nCorrelation lengths: %4.2f',ETA)
end
if nt==3
    fprintf('\nSquare exponential fields')
    fprintf('\nCorrelation lengths: %4.2f',ETA)
end
if nt==2
    fprintf('\nFractal fields')
  fprintf('\nHurst coefficient: %4.2f\n',BETA)
end
fprintf('\n------------------------------')
fprintf('\n-------- DOMAIN SIZE ---------')
fprintf('\nLx = %4.2f\nLy = %4.2f\nLz = %4.2f\n',Lx,Ly,Lz)
fprintf('\n------------------------------')
fprintf('\n------------ MESH ------------')
fprintf('\nnx = %d\nny = %d\nnz = %d\n',nx,ny,nz)
%
n=length(d);
if(n==0)
fprintf('\n------------------------------')
fprintf('\n------- UNCONDITIIONED -------') 
else
fprintf('\n------------------------------')
fprintf('\n-------- CONDITIONED ---------')
fprintf('\n---- Number of points (nc): %d',n)
fprintf('\n----- Positions (x,y,z) ------\n')
  v
fprintf('---------- Values ------------\n')
  d
fprintf('------------------------------')
end
fprintf('\n##############################\n')
return
