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

function [InitFunction, CostFunction, FeasibleFunction] = MLP_XOR

InitFunction = @MLP_XORInit;
CostFunction = @MLP_XORCost;
FeasibleFunction = @MLP_XORFeasible;
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [MaxParValue, MinParValue, Population, OPTIONS] = MLP_XORInit(OPTIONS)

global MinParValue MaxParValue
Granularity = 0.1;
MinParValue = -10;
MaxParValue = 10;
%MaxParValue = floor(1 + 2 * 2.048 / Granularity);
% Initialize population
for popindex = 1 : OPTIONS.popsize
    chrom = (MinParValue + (MaxParValue - MinParValue) * rand(1,OPTIONS.numVar));
    Population(popindex).chrom = chrom;
end
OPTIONS.OrderDependent = true;
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Population] = MLP_XORCost(OPTIONS, Population)
input=  [0 0 0 0 1 1 1 1;0 0 1 1 0 0 1 1;0 1 0 1 0 1 0 1];
%        0 0 1 1 0 0 1 1
%        0 1 0 1 0 1 0 1
target3=[0 1 1 0 1 0 0 1];

 Hno=7;
dim = 5*7+1;                      % Dimension of the problem
global MinParValue MaxParValue
popsize = OPTIONS.popsize;
for popindex = 1 : popsize
    Population(popindex).cost = 0;
        fitness=0;
        for indexi=1:4*Hno
        W(indexi)=Population(popindex).chrom(1,indexi);
    end
    
        for indexi=4*Hno+1:5*Hno+1
        B(indexi-4*Hno)=Population(popindex).chrom(1,indexi);
        end
        
        for pp=1:8
            actualvalue=my_simulate(W,B,input(1,pp),input(2,pp), input(3,pp));
            fitness=fitness+(target3(pp)-actualvalue)^2;
        end
        

        fitness=fitness/8;         
        Population(popindex).cost=fitness;
    %     for i = 1 : OPTIONS.numVar
%         gene = Population(popindex).chrom(i);
%         
%         Population(popindex).cost = Population(popindex).cost + x^2;
%     end
end
return
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Population] = MLP_XORFeasible(OPTIONS, Population)

global MinParValue MaxParValue
for i = 1 : OPTIONS.popsize
    for k = 1 : OPTIONS.numVar
        Population(i).chrom(k) = max(Population(i).chrom(k), MinParValue);
        Population(i).chrom(k) = min(Population(i).chrom(k), MaxParValue);
    end
end
return;
        
        
