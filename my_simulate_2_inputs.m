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

function o=my_simulate_2_inputs(W,B,x1,x2,Hno)
%Hno=50;
h=zeros(1,Hno);

for i=1:Hno
    h(i)=My_sigmoid(x1*W(i)+x2*W(Hno+i)+B(i));
end

sum=0;
for j=1:Hno
    sum=sum+(h(j)*W(2*Hno+j));
end
sum=sum+B(Hno+1);
o=My_sigmoid(sum);