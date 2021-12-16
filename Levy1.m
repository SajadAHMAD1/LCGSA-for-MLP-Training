%  Traning Feed-forward Neural Networks using LCGSA   %
%                                                                   %
%  Developed in MATLAB R2013b                                       %
%                                                                   %
%  programmer: Sajad Ahmad Rather                        %
%                                                                   %
%         e-Mail: sajad.win8@gmail.com                              %
%                                                                   %
% Homepage: https://www.linkedin.com/in/sajad-ahmad-rather-97a398110/   %
%  
function o=Levy1(n,dim)

beta=2; %3/2, 1
%Eq. (3.10)
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u=randn(1,dim)*sigma;
v=randn(1,dim);
step=u./abs(v).^(1/beta);
%     stepsize=0.01*step.*(X-best);
%     s=s+stepsize.*randn(size(s));
% Eq. (3.9)
o=0.01*step;%0.01