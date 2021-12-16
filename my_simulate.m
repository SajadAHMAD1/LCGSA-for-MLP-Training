%  Traning Feed-forward Neural Networks using Grey Wolf Optimizer   %
%                                                                   %
%  Developed in MATLAB R2011b(7.13)                                 %
%                                                                   %
%  Author and programmer: Seyedali Mirjalili                        %
%                                                                   %
%         e-Mail: ali.mirjalili@gmail.com                           %
%                 seyedali.mirjalili@griffithuni.edu.au             %
%                                                                   %
%       Homepage: http://www.alimirjalili.com                       %
%                                                                   %
%   Main paper: S. Mirjalili,How effective is the Grey Wolf         %
%               optimizer in training multi-layer perceptrons       %
%              Applied Intelligece, in press, 2015,                 %
%               http://dx.doi.org/10.1007/s10489-014-0645-7         %
%                                                                   %

% This function simulates the MLP

function o=my_simulate(Ino,Hno,Ono,W,B,x)
h=zeros(1,Hno);
o=zeros(1,Ono);
index=-1;

for i=1:Hno

    index=index+1;    
    ssum=0;
    for j=1:size(x,2)
        ssum= ssum+x(1,j)*W(index*Ino+j);
    end
    h(i)=My_sigmoid(ssum+B(i));
end

k=size(x,2);

for j=1:Hno
    o=o+(h(j)*W(k*Hno+j));
end

for k=1:Ono
    o(k)=My_sigmoid(o(k)+B(Ino+1));
end


