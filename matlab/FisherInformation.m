clear all; close all;

N = 10;
syms f x m s B t

f(x,m,s) = (1. / (s * sqrt(2. * pi))) * exp(-(1/2) *((x - m) / (s))^2);

B(x,t) = (t^x) * (1. - t)^(1.-x);
Bl(x,t)= log(B(x,t));
diff(diff(Bl(x,t),t),t);

fl(x,m,s) = simplify(log(f(x,m,s)));
flm(x,m,s) = simplify(diff(fl,m));
fls(x,m,s) = simplify(diff(fl,s));
flmm(x,m,s) = simplify(diff(flm,m));
flss(x,m,s) = simplify(diff(fls,s));
flms(x,m,s) = simplify(diff(flm,s));
flsm(x,m,s) = simplify(diff(fls,m));
% mu = 0.0;
% st = 1.0;
% 
% X  = mu + st * randn(N,1);
% muX= mean(X)
% stX= std(X)
% 
% t = [-5:0.01:5].';
% y = f(t,mu,st);
% 
% plot(t,y,'-');
% 
% score_m(x,m,s) = diff(log(f(x,m,s)),m);
% score_s(x,m,s) = diff(log(f(x,m,s)),s);
% 
% dfm(x) = diff(log(f(x,m,st)),m)
% 
% sm = double(score_m(X, muX, stX))
% ss = double(score_s(X, muX, stX))
% 
% I(m,s) = [int(diff(diff(log(f(x,m,s)),m),m) * f(x,m,s),x) int(diff(diff(log(f(x,m,s)),s),m) * f(x,m,s),x);
%           int(diff(diff(log(f(x,m,s)),m),s) * f(x,m,s),x) int(diff(diff(log(f(x,m,s)),s),s) * f(x,m,s),x)]

mu = [-6:0.1:14];
st = sqrt(25)
md = 5.0;
hold on
for i = 1 : 20
    xi = md + st * randn(1,1);
    y  = double(fl(xi,mu,st));
    plot(mu,y,'-');
end
st = sqrt(1.)
for i = 1 : 20
    xi = md + st * randn(1,1);
    y  = double(fl(xi,mu,st));
    plot(mu,y,'--');
end

f(x,m,s) = (1. / (sqrt(2. * pi * s))) * exp(-(1/2) *((x - m)^2) / (s));
logf(x,m,s) = log(f(x,m,s));
dlogfmu(x,m,s) = diff(diff(logf(x,m,s),m),m)
dlogfsg(x,m,s) = simplify(diff(diff(logf(x,m,s),s),s))

