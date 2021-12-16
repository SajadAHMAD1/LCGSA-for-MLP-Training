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

function output=My_sigmoid(x)
output=1/(1+exp(-x));
