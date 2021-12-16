%  Traning Feed-forward Neural Networks using LCGSA   %
%                                                                   %
%  Developed in MATLAB R2013b                                       %
%                                                                   %
%  programmer: Sajad Ahmad Rather                        %
%                                                                   %
%         e-Mail: sajad.win8@gmail.com                              %
%                                                                   %
% Homepage: https://www.linkedin.com/in/sajad-ahmad-rather-97a398110/   %
%                                                                   %


clear all 
close all
clc
Q=1;            % ACO Parameter
tau0=10;        % Initial Phromone             (ACO)
alpha=0.3;      % Phromone Exponential Weight  (ACO)
rho=0.1;        % Evaporation Rate             (ACO)
beta_min=0.2;   % Lower Bound of Scaling Factor (DE)
beta_max=0.8;   % Upper Bound of Scaling Factor (DE)
pCR=0.2;        % Crossover Probability         (DE)

Runno=10; % Number of Runs

SearchAgents_no=50; % Number of search agents
ElitistCheck=1;
min_flag=1;
Rpower=1;

Max_iteration=100;  % Maximum numbef of iterations
Algorithm_num=8;
% chValueInitial=20; % CGSA
 tic
% classification datasets

%   Function_name='F1'; %MLP_XOR dataset
% Function_name='F2'; %MLP_Balloon dataset
% Function_name='F3'; %MLP_Iris dataset
% Function_name='F4'; %MLP_Cancer dataset
% Function_name='F5'; %MLP_Heart dataset

% Function approximation datasets

 Function_name='F6'; %MLP_Sigmoid dataset
% Function_name='F7'; %MLP_Cosine dataset
% Function_name='F8'; %MLP_Sine dataset
% Function_name='F9'; %MLP_Sphere dataset

% Load details of the selected data set
[lb,ub,dim,fobj]=Get_Functions_details(Function_name);

% 
% if Function_name=='F1' 
% input=  [0 0 0 0 1 1 1 1;0 0 1 1 0 0 1 1;0 1 0 1 0 1 0 1];
% target3=[0 1 1 0 1 0 0 1];
%  Hno=7;
% dim = 5*7+1;                      % Dimension of the problem
%  
%     for i=1:1:Runno
%       
%         [Fbest,Lbest,BestChart]=GSA(SearchAgents_no,Max_iteration,ElitistCheck,min_flag,Rpower,lb,ub,dim,fobj);
%         BestSolutions1(i) = Fbest;
%         [PcgCurve,GBEST]=pso(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions2(i) = GBEST.O;
%          [gBestScore1,gBest1,GlobalBestCost1]=PSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%          BestSolutions3(i) = gBestScore1;
%           [gBestScore,gBest,GlobalBestCost]= CPSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%           BestSolutions4(i) = gBestScore;
%         [BestSolACO,BestAnt,BestCostACO] = ACO(SearchAgents_no, Max_iteration,Q,tau0,alpha,rho,lb,ub,dim,fobj);
%         BestSolutions5(i) = BestSolACO.Cost;
%         [BestCost,Best_Hab,BestSol] = bbo( SearchAgents_no, Max_iteration,lb,ub,dim,fobj);
%         BestSolutions6(i) = BestSol.Cost;
%         [BestSolDE,DBestSol,BestCostDE] = DE(SearchAgents_no, Max_iteration,beta_min,beta_max,pCR,lb,ub,dim,fobj);
%         BestSolutions7(i) = BestSolDE.Cost ;
%         
% [Best_scoreDA,Best_posDA,DA_cg_curve]=DA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%  BestSolutions8(i) = Best_scoreDA ;
%  [Best_scoreSCA,Best_posSCA,SCA_cg_curve]=SCA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%  BestSolutions9(i) = Best_scoreSCA ;
%  [Best_score,Best_posSSA,SSA_cg_curve]=SSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%  BestSolutions10(i) = Best_score ;
%  
%   [CFbest,CLbest,CBestChart]= LCGSA(SearchAgents_no,Max_iteration, min_flag,lb,ub,dim,fobj, Algorithm_num); 
%   BestSolutions11(i) = CFbest;
%   
% %  disp(['GSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Fbest),')'])  
% %  disp(['PSO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(GBEST.O),')'])  
% %  disp(['PSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore1),')'])  
% %  disp(['CPSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore),')'])
% %  disp(['ACO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolACO.Cost),')'])
% % disp(['BBO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSol.Cost),')'])
% % disp(['DE is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolDE.Cost),')']) 
% %  disp(['DA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_scoreDA),')']) 
% % disp(['SCA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_scoreSCA),')'])
% % disp(['SSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_score),')'])
% disp(['LCGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(CFbest),')'])
%      end
% %   for tt=1:11
%        Rrate=0;
%                 W=Lbest(1:4*Hno);
%                 B=Lbest(4*Hno+1:dim);
%                 W=PcgCurve(1:4*Hno);
%                 B=PcgCurve(4*Hno+1:dim);
%                  W=gBest1(1:4*Hno);
%                  B=gBest1(4*Hno+1:dim);
%                   W=gBest(1:4*Hno);
%                   B=gBest(4*Hno+1:dim);
% W=BestAnt(1:4*Hno);
% B=BestAnt(4*Hno+1:dim);
% W=Best_Hab(1:4*Hno);
% B=Best_Hab(4*Hno+1:dim);
% W=DBestSol(1:4*Hno);
% B=DBestSol(4*Hno+1:dim);
%  W=DA_cg_curve(1:4*Hno);
%  B=DA_cg_curve(4*Hno+1:dim);
%  W=SCA_cg_curve(1:4*Hno);
%  B=SCA_cg_curve(4*Hno+1:dim);
%  W=SSA_cg_curve(1:4*Hno);
%  B=SSA_cg_curve(4*Hno+1:dim);
%  
%  W=CLbest(1:4*Hno);
%  B=CLbest(4*Hno+1:dim);
%   
%        
% %         for pp=1:8            
% %             actualvalue=my_simulate(3,Hno,1,W,B,input(:,pp)');
% %             if(target3(pp)==1)
% %                 if (actualvalue>=0.95)
% %                     Rrate=Rrate+1;
% %                 end
% %             end
% %             if(target3(pp)==0)
% %                 if (actualvalue(1)<0.05)
% %                     Rrate=Rrate+1;
% %                 end  
% %             end
% %         end
% %          
% %         Classification_rate=(Rrate/8)*100;
%         
% % disp(num2str(Classification_rate(i))) 
% 
% end
%     end
%  display('--------------------------------------------------------------------------------------------')
%  display('Classification rate')
% %  display('GSA   PSO')
%   display('   GSA    PSO       PSOGSA    CPSOGSA     ACO       BBO   DE       DA       SCA       SSA     LCGSA')
%  display(Classification_rate(1:11))
%  display('--------------------------------------------------------------------------------------------')
%  disp(num2str(Classification_rate)) 
% %     A_Classification_rate=Classification_rate;
%     Average= mean(BestSolutions1);
%     StandDP=std(BestSolutions1);
%     Med = median(BestSolutions1); 
%     [BestValueP I] = min(BestSolutions1);
%     [WorstValueP IM]=max(BestSolutions1);
%   end
    
%     
% if Function_name=='F2'
% 
% load baloon.txt
%  x=sortrows(baloon,2);
% %  I2=x(1:150,1:4);
%  I2(:,1)=x(1:20,1);
%  I2(:,2)=x(1:20,2);
%  I2(:,3)=x(1:20,3);
%  I2(:,4)=x(1:20,4);
%  T=x(1:20,5);
%  
% 
% Hno=9;
% dim = 6*9+1;
% %  
%  for i=1:1:Runno
%       
%         [Fbest,Lbest,BestChart]=GSA(SearchAgents_no,Max_iteration,ElitistCheck,min_flag,Rpower,lb,ub,dim,fobj);
%         BestSolutions1(i) = Fbest;
%         [PcgCurve,GBEST]=pso(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions2(i) = GBEST.O;
%          [gBestScore1,gBest1,GlobalBestCost1]=PSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%          BestSolutions3(i) = gBestScore1;
%           [gBestScore,gBest,GlobalBestCost]= CPSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%           BestSolutions4(i) = gBestScore;
%         [BestSolACO,BestAnt,BestCostACO] = ACO(SearchAgents_no, Max_iteration,Q,tau0,alpha,rho,lb,ub,dim,fobj);
%         BestSolutions5(i) = BestSolACO.Cost;
%         [BestCost,Best_Hab,BestSol] = bbo( SearchAgents_no, Max_iteration,lb,ub,dim,fobj);
%         BestSolutions6(i) = BestSol.Cost;
%         [BestSolDE,DBestSol,BestCostDE] = DE(SearchAgents_no, Max_iteration,beta_min,beta_max,pCR,lb,ub,dim,fobj);
%         BestSolutions7(i) = BestSolDE.Cost ;
%         
% [Best_scoreDA,Best_posDA,DA_cg_curve]=DA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%  BestSolutions8(i) = Best_scoreDA ;
%  [Best_scoreSCA,Best_posSCA,SCA_cg_curve]=SCA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%  BestSolutions9(i) = Best_scoreSCA ;
%  [Best_score,Best_posSSA,SSA_cg_curve]=SSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%  BestSolutions10(i) = Best_score ;
% %  
%   [CFbest,CLbest,CBestChart]= LCGSA(SearchAgents_no,Max_iteration, min_flag,lb,ub,dim,fobj, Algorithm_num); 
%   BestSolutions11(i) = CFbest;
%   
%  
% %   disp(['GSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Fbest),')'])  
% %   disp(['PSO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(GBEST.O),')'])  
% %   disp(['PSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore1),')'])  
% %   disp(['CPSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore),')'])
% %  disp(['ACO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolACO.Cost),')'])
% % disp(['BBO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSol.Cost),')'])
% % disp(['DE is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolDE.Cost),')']) 
% %   disp(['DA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_scoreDA),')']) 
% %  disp(['SCA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_scoreSCA),')'])
% % disp(['SSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_score),')'])
% disp(['LCGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(CFbest),' ,Algorithm = ', num2str(Algorithm_num), ')'])
%  end
%  
%                 W=Lbest(1:45);
%                 B=Lbest(46:55);
%                  W=PcgCurve(1:45);
%                  B=PcgCurve(46:55);
%                    W=gBest1(1:45);
%                    B=gBest1(46:55);
%                 W=gBest(1:45);
%                 B=gBest(46:55);
%         W=BestAnt(1:45);
%         B=BestAnt(46:55);
% W=Best_Hab(1:45);
% B=Best_Hab(46:55);
% W=DBestSol(1:45);
% B=DBestSol(46:55);
% 
%  W=DA_cg_curve(1:45);
%  B=DA_cg_curve(46:55);
%  W=SCA_cg_curve(1:45);
%  B=SCA_cg_curve(46:55);
%  W=SSA_cg_curve(1:45);
%  B=SSA_cg_curve(46:55);
% 
%  W=CLbest(1:45);
%  B=CLbest(46:55);
%  
%  Rrate =0;
%         for pp=1:20
%             actualvalue=my_simulate(4,9,1,W,B,I2(pp,:));
%             if(T(pp)==1)
%                 if (actualvalue>=0.95)
%                     Rrate=Rrate+1;
%                 end
%             end
%             if(T(pp)==0)
%                 if (actualvalue(1)<0.05)
%                     Rrate=Rrate+1;
%                 end  
%             end
% 
%         end
%                
%       Classification_rate=(Rrate/20)*100;
% end
%     disp(num2str(Classification_rate))   
% % disp(num2str(Classification_rate)) 
%     Average= mean(BestSolutions11);
%     StandDP=std(BestSolutions11);
%     Med = median(BestSolutions11); 
%     [BestValueP I] = min(BestSolutions11);
%     [WorstValueP IM]=max(BestSolutions11);
    
% end
% % % 
 if Function_name=='F3' 
    
    load iris.txt;
 x=sortrows(iris,2);
 I2=x(1:150,1:4);
 H2=x(1:150,1);
 H3=x(1:150,2);
 H4=x(1:150,3);
 H5=x(1:150,4);
 T=x(1:150,5);
 I=(I2-0.1)./(7.9-0.1);
 H2=H2';
 [xf,PS] = mapminmax(H2);
 I2(:,1)=xf;
 
 H3=H3';
 [xf,PS2] = mapminmax(H3);
 I2(:,2)=xf;
 
 H4=H4';
 [xf,PS3] = mapminmax(H4);
 I2(:,3)=xf;
 
 H5=H5';
 [xf,PS4] = mapminmax(H5);
 I2(:,4)=xf;
 Thelp=T;
 T=T';
 [yf,PS5]= mapminmax(T);
 T=yf;
 T=T';
   for i=1:1:Runno
%    [Fbest,Lbest,BestChart]=GSA(SearchAgents_no,Max_iteration,ElitistCheck,min_flag,Rpower,lb,ub,dim,fobj);
%         BestSolutions1(i) = Fbest;
%         [PcgCurve,GBEST]=pso(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions2(i) = GBEST.O;
%          [gBestScore1,gBest1,GlobalBestCost1]=PSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% %          BestSolutions3(i) = gBestScore1;
% %           [gBestScore,gBest,GlobalBestCost]= CPSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% %           BestSolutions4(i) = gBestScore;
% %         [BestSolACO,BestAnt,BestCostACO] = ACO(SearchAgents_no, Max_iteration,Q,tau0,alpha,rho,lb,ub,dim,fobj);
% %         BestSolutions5(i) = BestSolACO.Cost;
% %         [BestCost,Best_Hab,BestSol] = bbo( SearchAgents_no, Max_iteration,lb,ub,dim,fobj);
% %         BestSolutions6(i) = BestSol.Cost;
% %         [BestSolDE,DBestSol,BestCostDE] = DE(SearchAgents_no, Max_iteration,beta_min,beta_max,pCR,lb,ub,dim,fobj);
% %         BestSolutions7(i) = BestSolDE.Cost ;
% %         
% % [Best_scoreDA,Best_posDA,DA_cg_curve]=DA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% %  BestSolutions8(i) = Best_scoreDA ;
% %  [Best_scoreSCA,Best_posSCA,SCA_cg_curve]=SCA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% %  BestSolutions9(i) = Best_scoreSCA ;
% %  [Best_score,Best_posSSA,SSA_cg_curve]=SSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% %  BestSolutions10(i) = Best_score ;
% % %  
%   [CFbest,CLbest,CBestChart]= LCGSA(SearchAgents_no,Max_iteration, min_flag,lb,ub,dim,fobj, Algorithm_num); 
%   BestSolutions11(i) = CFbest;
% %          W=Lbest(1:63);
% %          B=Lbest(64:75);
% %         W=PcgCurve(1:63);
% %         B=PcgCurve(64:75);
% %         W=gBest1(1:63);
% %         B=gBest1(64:75);
% %           W=gBest(1:63);
% %           B=gBest(64:75);
% %             W=BestAnt(1:63);
% %             B=BestAnt(64:75);
% %             W=Best_Hab(1:63);
% %             B=Best_Hab(64:75);
% %             W=DBestSol(1:63);
% %             B=DBestSol(64:75);
% % % 
% %  W=DA_cg_curve(1:63);
% %  B=DA_cg_curve(64:75);
% %  W=SCA_cg_curve(1:63);
% %  B=SCA_cg_curve(64:75);
% %  W=SSA_cg_curve(1:63);
% %  B=SSA_cg_curve(64:75);
% %  
%  W=CLbest(1:63);
%  B=CLbest(64:75);
%  
% %    disp(['GSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Fbest),')'])  
% %    disp(['PSO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(GBEST.O),')'])  
% %    disp(['PSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore1),')'])  
% %   disp(['CPSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore),')'])
% %   disp(['ACO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolACO.Cost),')'])
% %  disp(['BBO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSol.Cost),')'])
% % disp(['DE is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolDE.Cost),')']) 
% %   disp(['DA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_scoreDA),')']) 
% %  disp(['SCA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_scoreSCA),')'])
% % disp(['SSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_score),')'])
% disp(['LCGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(CFbest),' ,Algorithm = ', num2str(Algorithm_num), ')'])
%    end
% 
%  Rrate=0;
%         for pp=1:150
%             actualvalue=my_simulate(4,9,3,W,B,I2(pp,:));
%             if(T(pp)==-1)
%                 if (actualvalue(1)>=0.95 && actualvalue(2)<0.05 && actualvalue(3)<0.05)
%                     Rrate=Rrate+1;
%                 end
%             end
%             if(T(pp)==0)
%                 if (actualvalue(1)<0.05 && actualvalue(2)>=0.95 && actualvalue(3)<0.05)
%                     Rrate=Rrate+1;
%                 end  
%             end
%             if(T(pp)==1)
%                 if (actualvalue(1)<0.05 && actualvalue(2)<0.05 && actualvalue(3)>=0.95)
%                     Rrate=Rrate+1;
%                 end              
%             end
%         end
%         
%         Classification_rate=(Rrate/150)*100;
%  end
% %     
%    disp(num2str(Classification_rate))  
%     Average= mean(BestSolutions11);
%     StandDP=std(BestSolutions11);
%     Med = median(BestSolutions11); 
%     [BestValueP I] = min(BestSolutions11);
%     [WorstValueP IM]=max(BestSolutions11);
% %     
% 
% % % % %    
% 
% if Function_name=='F4'
%     
%     load Cancer.txt
%  x=Cancer;
% %   I2=x(1:150,1:4);
%  H2=x(1:699,2:11);
%  for iii=1:699
%      for jjj=1:10
%          H2(iii,jjj)=((H2(iii,jjj)-1)/9);
%      end
%  end
%   I2=H2(1:699,1:9);
%  
%  T=H2(1:699,10);
%  Hno=19;
%  dim=11*19;
%  
%     for i=1:1:Runno
%         
%         [Fbest,Lbest,BestChart]=GSA(SearchAgents_no,Max_iteration,ElitistCheck,min_flag,Rpower,lb,ub,dim,fobj);
%         BestSolutions1(i) = Fbest;
%         [PcgCurve,GBEST]=pso(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions2(i) = GBEST.O;
% % % % %                               [gBestScore1,gBest1,GlobalBestCost1]=PSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% % % % %                                    BestSolutions3(i) = gBestScore1;
%          [gBestScore,gBest,GlobalBestCost]= CPSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%          BestSolutions4(i) = gBestScore;
%         [BestSolACO,BestAnt,BestCostACO] = ACO(SearchAgents_no, Max_iteration,Q,tau0,alpha,rho,lb,ub,dim,fobj);
%         BestSolutions5(i) = BestSolACO.Cost;
%         [BestCost,Best_Hab,BestSol] = bbo( SearchAgents_no, Max_iteration,lb,ub,dim,fobj);
%         BestSolutions6(i) = BestSol.Cost;
%         [BestSolDE,DBestSol,BestCostDE] = DE(SearchAgents_no, Max_iteration,beta_min,beta_max,pCR,lb,ub,dim,fobj);
%         BestSolutions7(i) = BestSolDE.Cost ;
%         
%          [Best_scoreDA,Best_posDA,DA_cg_curve]=DA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%          BestSolutions8(i) = Best_scoreDA ;
%     [Best_scoreSCA,Best_posSCA,SCA_cg_curve]=SCA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%      BestSolutions9(i) = Best_scoreSCA ;
%     [Best_score,Best_posSSA,SSA_cg_curve]=SSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%      BestSolutions10(i) = Best_score ;
% 
%         
%         
% Rrate=0;
%      
% W=Lbest(1:10*Hno);
% B=Lbest(10*Hno+1:dim);
% % W=PcgCurve(1:10*Hno);
% % B=PcgCurve(10*Hno+1:dim); 
% % % % % %                                 % W=gBest1(1:10*Hno);
% % % % %                                 % B=gBest1(10*Hno+1:dim);
% % W=gBest(1:10*Hno);
% % B=gBest(10*Hno+1:dim);
% % W=BestAnt(1:10*Hno);
% % B=BestAnt(10*Hno+1:dim);
% % W=Best_Hab(1:10*Hno);
% % B=Best_Hab(10*Hno+1:dim);
% % W=DBestSol(1:10*Hno);
% % B=DBestSol(10*Hno+1:dim);
% %   
% %  W=1:10*Hno;
% %  B=10*Hno+1:dim;
% %  W=SCA_cg_curve(1:10*Hno);
% %  B=SCA_cg_curve(10*Hno+1:dim);
% %  W=SSA_cg_curve(1:10*Hno);
% %  B=SSA_cg_curve(10*Hno+1:dim);
% 
% 
% 
%         for pp=600:699
%             actualvalue=my_simulate(9,Hno,1,W,B,I2(pp,:) );
%             if(T(pp)>=0.3 && T(pp)<0.4)
%                 if (abs(actualvalue-0.333333333333333)<0.1)
%                     Rrate=Rrate+1;
%                 end
%             end
%             if(T(pp)>=0.1 && T(pp)<0.2)
%                 if (abs(actualvalue-0.111111111111111)<0.1)
%                     Rrate=Rrate+1;
%                 end  
%             end
% 
%         end
%         
%         Classification_rate(1,i)=(Rrate/100)*100;
%   disp(['GSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Fbest),' Classification rate =' , num2str(Classification_rate(i)),')'])  
% % disp(['PSO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(GBEST.O),' Classification rate = ' , num2str(Classification_rate(i)),')'])  
% % disp(['PSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore1),' Classification rate = ' , num2str(Classification_rate(i)),')'])  
% % disp(['CPSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore),' Classification rate =' , num2str(Classification_rate(i)),')'])
% % disp(['ACO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolACO.Cost),' Classification rate = ' , num2str(Classification_rate(i)),')'])
% % disp(['BBO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSol.Cost),' Classification rate = ' , num2str(Classification_rate(i)),')'])
% % disp(['DE is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolDE.Cost),' Classification rate = ' , num2str(Classification_rate(i)),')'])
% % disp(['DA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_scoreDA),' Classification rate = ' , num2str(Classification_rate(i)),')']) 
% % disp(['SCA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_scoreSCA),' Classification rate = ' , num2str(Classification_rate(i)),')'])
% % disp(['SSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_score),' Classification rate = ' , num2str(Classification_rate(i)),')'])
% 
%     end
% %     A_Classification_rate=mean(Classification_rate);
% %     Average= mean(BestSolutions10);
% %     StandDP=std(BestSolutions10);
% %     Med = median(BestSolutions10); 
% %     [BestValueP I] = min(BestSolutions10);
% %     [WorstValueP IM]=max(BestSolutions10);
% end
% %

% if Function_name=='F5'
% 
% load Heart.txt
%  x=Heart;
% 
%  I2(:,1)=x(1:80,2);
%  I2(:,2)=x(1:80,3);
%  I2(:,3)=x(1:80,4);
%  I2(:,4)=x(1:80,5);
%  I2(:,5)=x(1:80,6);
%  I2(:,6)=x(1:80,7);
%  I2(:,7)=x(1:80,8);
%  I2(:,8)=x(1:80,9);
%  I2(:,9)=x(1:80,10);
%  I2(:,10)=x(1:80,11);
%  I2(:,11)=x(1:80,12);
%  I2(:,12)=x(1:80,13);
%  I2(:,13)=x(1:80,14);
%  I2(:,14)=x(1:80,15);
%  I2(:,15)=x(1:80,16);
%  I2(:,16)=x(1:80,17);
%  I2(:,17)=x(1:80,18);
%  I2(:,18)=x(1:80,19);
%  I2(:,19)=x(1:80,20);
%  I2(:,20)=x(1:80,21);
%  I2(:,21)=x(1:80,22); 
%  I2(:,22)=x(1:80,23);  
%  T=x(1:80,1);
% 
%  Hno=45;
% dim = 24*45+1;    
%  
% for i=1:1:Runno
%     
%    [Fbest,Lbest,BestChart]=GSA(SearchAgents_no,Max_iteration,ElitistCheck,min_flag,Rpower,lb,ub,dim,fobj);
%         BestSolutions1(i) = Fbest;
%         [PcgCurve,GBEST]=pso(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions2(i) = GBEST.O;
%          [gBestScore1,gBest1,GlobalBestCost1]=PSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%          BestSolutions3(i) = gBestScore1;
%           [gBestScore,gBest,GlobalBestCost]= CPSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%           BestSolutions4(i) = gBestScore;
%          [BestSolACO,BestAnt,BestCostACO] = ACO(SearchAgents_no, Max_iteration,Q,tau0,alpha,rho,lb,ub,dim,fobj);
%         BestSolutions5(i) = BestSolACO.Cost;
%         [BestCost,Best_Hab,BestSol] = bbo( SearchAgents_no, Max_iteration,lb,ub,dim,fobj);
%         BestSolutions6(i) = BestSol.Cost;
%         [BestSolDE,DBestSol,BestCostDE] = DE(SearchAgents_no, Max_iteration,beta_min,beta_max,pCR,lb,ub,dim,fobj);
%         BestSolutions7(i) = BestSolDE.Cost ;
%         
% [Best_scoreDA,Best_posDA,DA_cg_curve]=DA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%  BestSolutions8(i) = Best_scoreDA ;
%  [Best_scoreSCA,Best_posSCA,SCA_cg_curve]=SCA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%  BestSolutions9(i) = Best_scoreSCA ;
%  [Best_score,Best_posSSA,SSA_cg_curve]=SSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%  BestSolutions10(i) = Best_score ;
% % %  
%   [CFbest,CLbest,CBestChart]= LCGSA(SearchAgents_no,Max_iteration, min_flag,lb,ub,dim,fobj, Algorithm_num); 
%   BestSolutions11(i) = CFbest;
% 
% 
%     disp(['GSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Fbest),')'])  
% %    disp(['PSO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(GBEST.O),')'])  
% %    disp(['PSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore1),')'])  
% %   disp(['CPSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore),')'])
% %   disp(['ACO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolACO.Cost),')'])
% %  disp(['BBO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSol.Cost),')'])
% % disp(['DE is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolDE.Cost),')']) 
% %   disp(['DA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_scoreDA),')']) 
% %  disp(['SCA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_scoreSCA),')'])
% % disp(['SSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_score),')'])
% % disp(['LCGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(CFbest),' ,Algorithm = ', num2str(Algorithm_num), ')'])
% 
% end
%         W=Lbest(1:23*Hno);
%         B=Lbest(23*Hno+1:dim);
% W=PcgCurve(1:23*Hno);
% B=PcgCurve(23*Hno+1:dim); 
% W=gBest1(1:23*Hno);
% B=gBest1(23*Hno+1:dim);
% W=gBest(1:23*Hno);
% B=gBest(23*Hno+1:dim);
% W=BestAnt(1:23*Hno);
% B=BestAnt(23*Hno+1:dim);
% W=Best_Hab(1:23*Hno);
% B=Best_Hab(23*Hno+1:dim);
% W=DBestSol(1:23*Hno);
% B=DBestSol(23*Hno+1:dim);
% % % % 
% % %  W=DA_cg_curve(1:23*Hno);
% % %  B=DA_cg_curve(23*Hno+1:dim);
%   W=1:23*Hno;
%   B=23*Hno+1:dim;
% % %  W=SCA_cg_curve(1:23*Hno);
% % %  B=SCA_cg_curve(23*Hno+1:dim);
% % % %  W=SSA_cg_curve(1:23*Hno);
% % % % B=SSA_cg_curve(23*Hno+1:dim);
% W=CLbest(1:23*Hno);
% B=CLbest(23*Hno+1:dim);
% 
%   Rrate =0;
%         for pp=1:80
%             actualvalue=my_simulate(22,Hno,1,W,B,I2(pp,:) );
%             if(T(pp)==1)
%                 if (actualvalue>=0.95)
%                     Rrate=Rrate+1;
%                 end
%             end
%             if(T(pp)==0)
%                 if (actualvalue(1)<0.05)
%                     Rrate=Rrate+1;
%                 end  
%             end
% 
%         end
%         
%        Classification_rate =(Rrate/80)*100;
%    end
% %     
% %  

%     disp(num2str(Classification_rate)) 
%     Average= mean(BestSolutions11);
%     StandDP=std(BestSolutions11);
%     Med = median(BestSolutions11); 
%     [BestValueP I] = min(BestSolutions11);
%     [WorstValueP IM]=max(BestSolutions11);
%     
% end

if Function_name=='F6'  %% Sigmoid
    
    Hnode=15;
dim = 3*Hnode+1;
 
%for test 3 times more than the training samples
%   xf1=[0:0.01:pi];
%   yf1=sin(2.*xf1);
%   yf1=yf1.*exp(-xf1);
 xf1=[-3:0.05:3];
%  yf1=sin(2.*xf1);
%  yf1=yf1.*exp(-xf1);
 %yf1=xf1.^2;
%yf1=xf1.^4-6.*xf1.^2+3;
yf1=sigmf(xf1,[1 0]);

%   xf1=[-2*pi:0.05:2*pi];
%   yf1=sin(2.*xf1);
  %yf1=yf1.*exp(-xf1);
 yNN=zeros(1,10);
 [xf,PS] = mapminmax(xf1);
 [yf,PS2]= mapminmax(yf1);
  [M N]=size(xf);
  
  test_error=zeros(1,Runno);
  
   for i=1:1:Runno
%     
   [Fbest,Lbest,BestChart]=GSA(SearchAgents_no,Max_iteration,ElitistCheck,min_flag,Rpower,lb,ub,dim,fobj);
        BestSolutions1(i) = Fbest;
        [PcgCurve,GBEST]=pso(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
        BestSolutions2(i) = GBEST.O;
         [gBestScore1,gBest1,GlobalBestCost1]=PSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
         BestSolutions3(i) = gBestScore1;
          [gBestScore,gBest,GlobalBestCost]= CPSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
          BestSolutions4(i) = gBestScore;
         [BestSolACO,BestAnt,BestCostACO] = ACO(SearchAgents_no, Max_iteration,Q,tau0,alpha,rho,lb,ub,dim,fobj);
        BestSolutions5(i) = BestSolACO.Cost;
        [BestCost,Best_Hab,BestSol] = bbo( SearchAgents_no, Max_iteration,lb,ub,dim,fobj);
        BestSolutions6(i) = BestSol.Cost;
        [BestSolDE,DBestSol,BestCostDE] = DE(SearchAgents_no, Max_iteration,beta_min,beta_max,pCR,lb,ub,dim,fobj);
        BestSolutions7(i) = BestSolDE.Cost ;
%         
[Best_scoreDA,Best_posDA,DA_cg_curve]=DA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
 BestSolutions8(i) = Best_scoreDA ;
 [Best_scoreSCA,Best_posSCA,SCA_cg_curve]=SCA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
 BestSolutions9(i) = Best_scoreSCA ;
 [Best_score,Best_posSSA,SSA_cg_curve]=SSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
 BestSolutions10(i) = Best_score ;
% %  
  [CFbest,CLbest,CBestChart]= LCGSA(SearchAgents_no,Max_iteration, min_flag,lb,ub,dim,fobj, Algorithm_num); 
  BestSolutions11(i) = CFbest;
% 
% 
%      disp(['GSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Fbest),')'])  
%    disp(['PSO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(GBEST.O),')'])  
%    disp(['PSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore1),')'])  
%   disp(['CPSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore),')'])
%   disp(['ACO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolACO.Cost),')'])
%  disp(['BBO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSol.Cost),')'])
% disp(['DE is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolDE.Cost),')']) 
% disp(['DA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_scoreDA),')']) 
%  disp(['SCA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_scoreSCA),')'])
% disp(['SSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_score),')'])
disp(['LCGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(CFbest),' ,Algorithm = ', num2str(Algorithm_num), ')'])
% 
 end

        W=Lbest(1:2*Hnode);
        B=Lbest(2*Hnode+1:3*Hnode+1);
        W=PcgCurve(1:2*Hnode);
        B=PcgCurve(2*Hnode+1:3*Hnode+1); 
W=gBest1(1:2*Hnode);
B=gBest1(2*Hnode+1:3*Hnode+1);
W=gBest(1:2*Hnode);
B=gBest(2*Hnode+1:3*Hnode+1);
W=BestAnt(1:2*Hnode);
B=BestAnt(2*Hnode+1:3*Hnode+1);
W=Best_Hab(1:2*Hnode);
B=Best_Hab(2*Hnode+1:3*Hnode+1);
W=DBestSol(1:2*Hnode);
B=DBestSol(2*Hnode+1:3*Hnode+1);

W=DA_cg_curve(1:2*Hnode);
B=DA_cg_curve(2*Hnode+1:3*Hnode+1);
 W=SCA_cg_curve(1:2*Hnode);
 B=SCA_cg_curve(2*Hnode+1:3*Hnode+1);
 W=SSA_cg_curve(1:2*Hnode);
 B=SSA_cg_curve(2*Hnode+1:3*Hnode+1);
 W=CLbest(1:2*Hnode);
 B=CLbest(2*Hnode+1:3*Hnode+1);
 
        for pp=1:N
            yNN(pp)=my_simulate(1,15,1, W,B,xf(pp));
        end       

end
%      
%     Average= mean(BestSolutions11);
%     StandDP=std(BestSolutions11);
%     Med = median(BestSolutions11); 
%     [BestValueP I] = min(BestSolutions11);
%     [WorstValueP IM]=max(BestSolutions11);


   
% if Function_name=='F7' %% Cosine
%  
% Hnode=15;
% dim = 3*Hnode+1;
%  
% %for test 3 times more than the training samples
% %   xf1=[0:0.01:pi];
% %   yf1=sin(2.*xf1);
% %   yf1=yf1.*exp(-xf1);
%  %xf1=[-3:0.05:3];
% %  yf1=sin(2.*xf1);
% %  yf1=yf1.*exp(-xf1);
%  %yf1=xf1.^2;
% %yf1=xf1.^4-6.*xf1.^2+3;
% 
%   xf1=[1.25:0.04:2.75];
%   yf1=power(cos(xf1.*pi/2),7);
%   %yf1=yf1.*exp(-xf1);
%  yNN=zeros(1,10);
%  [xf,PS] = mapminmax(xf1);
%  [yf,PS2]= mapminmax(yf1);
%   [M N]=size(xf);
%   test_error=zeros(1,Runno);
%  for i=1:1:Runno
% %     
%    [Fbest,Lbest,BestChart]=GSA(SearchAgents_no,Max_iteration,ElitistCheck,min_flag,Rpower,lb,ub,dim,fobj);
%         BestSolutions1(i) = Fbest;
%         [PcgCurve,GBEST]=pso(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%         BestSolutions2(i) = GBEST.O;
%          [gBestScore1,gBest1,GlobalBestCost1]=PSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%          BestSolutions3(i) = gBestScore1;
%           [gBestScore,gBest,GlobalBestCost]= CPSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%           BestSolutions4(i) = gBestScore;
%          [BestSolACO,BestAnt,BestCostACO] = ACO(SearchAgents_no, Max_iteration,Q,tau0,alpha,rho,lb,ub,dim,fobj);
%         BestSolutions5(i) = BestSolACO.Cost;
%         [BestCost,Best_Hab,BestSol] = bbo( SearchAgents_no, Max_iteration,lb,ub,dim,fobj);
%         BestSolutions6(i) = BestSol.Cost;
%         [BestSolDE,DBestSol,BestCostDE] = DE(SearchAgents_no, Max_iteration,beta_min,beta_max,pCR,lb,ub,dim,fobj);
%         BestSolutions7(i) = BestSolDE.Cost ;
% %         
% [Best_scoreDA,Best_posDA,DA_cg_curve]=DA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%  BestSolutions8(i) = Best_scoreDA ;
%  [Best_scoreSCA,Best_posSCA,SCA_cg_curve]=SCA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%  BestSolutions9(i) = Best_scoreSCA ;
%  [Best_score,Best_posSSA,SSA_cg_curve]=SSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%  BestSolutions10(i) = Best_score ;
% % % %  
%   [CFbest,CLbest,CBestChart]= LCGSA(SearchAgents_no,Max_iteration, min_flag,lb,ub,dim,fobj, Algorithm_num); 
%   BestSolutions11(i) = CFbest;
% % 
% % 
% %       disp(['GSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Fbest),')'])  
% %    disp(['PSO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(GBEST.O),')'])  
% %    disp(['PSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore1),')'])  
% %   disp(['CPSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore),')'])
% %   disp(['ACO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolACO.Cost),')'])
% %  disp(['BBO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSol.Cost),')'])
% % disp(['DE is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolDE.Cost),')']) 
% % disp(['DA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_scoreDA),')']) 
% %  disp(['SCA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_scoreSCA),')'])
% % disp(['SSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_score),')'])
% disp(['LCGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(CFbest),' ,Algorithm = ', num2str(Algorithm_num), ')'])
% % 
%  end
% %         
%          W= Lbest(1:2*Hnode);
%          B=Lbest(2*Hnode+1:3*Hnode+1);
% W=PcgCurve(1:2*Hnode);
% B=PcgCurve(2*Hnode+1:3*Hnode+1); 
% W=gBest1(1:2*Hnode);
% B=gBest1(2*Hnode+1:3*Hnode+1);
% W=gBest(1:2*Hnode);
% B=gBest(2*Hnode+1:3*Hnode+1);
% W=BestAnt(1:2*Hnode);
% B=BestAnt(2*Hnode+1:3*Hnode+1);
% W=Best_Hab(1:2*Hnode);
% B=Best_Hab(2*Hnode+1:3*Hnode+1);
% W=DBestSol(1:2*Hnode);
% B=DBestSol(2*Hnode+1:3*Hnode+1);
% % 
%  W=DA_cg_curve(1:2*Hnode);
%  B=DA_cg_curve(2*Hnode+1:3*Hnode+1);
%  W=SCA_cg_curve(1:2*Hnode);
%  B=SCA_cg_curve(2*Hnode+1:3*Hnode+1);
%  W=SSA_cg_curve(1:2*Hnode);
%  B=SSA_cg_curve(2*Hnode+1:3*Hnode+1);
%  W=CLbest(1:2*Hnode);
%  B=CLbest(2*Hnode+1:3*Hnode+1);
% % 
%         for pp=1:N
%             yNN(pp)=my_simulate(1,15,1, W,B,xf(pp));
%         end
% %         
% 
% % 
% end
%     
%     Average= mean(BestSolutions11);
%     StandDP=std(BestSolutions11);
%     Med = median(BestSolutions11); 
%     [BestValueP I] = min(BestSolutions11);
%     [WorstValueP IM]=max(BestSolutions11);
% end

% if Function_name=='F8' %%Sine
% 
% Hnode=15;
% dim = 3*Hnode+1;
%  
% %for test 3 times more than the training samples
% %   xf1=[0:0.01:pi];
% %   yf1=sin(2.*xf1);
% %   yf1=yf1.*exp(-xf1);
%  %xf1=[-3:0.05:3];
% %  yf1=sin(2.*xf1);
% %  yf1=yf1.*exp(-xf1);
%  %yf1=xf1.^2;
% %yf1=xf1.^4-6.*xf1.^2+3;
% 
%   xf1=[-2*pi:0.05:2*pi];
%   yf1=sin(2.*xf1);
%   %yf1=yf1.*exp(-xf1);
%  yNN=zeros(1,10);
%  [xf,PS] = mapminmax(xf1);
%  [yf,PS2]= mapminmax(yf1);
%   [M N]=size(xf);
%   test_error=zeros(1,Runno);
%     for i=1:1:Runno
% %         [Fbest,Lbest,BestChart]=GSA(SearchAgents_no,Max_iteration,ElitistCheck,min_flag,Rpower,lb,ub,dim,fobj);
% %         BestSolutions1(i) = Fbest;
% %         [PcgCurve,GBEST]=pso(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% %         BestSolutions2(i) = GBEST.O;
%         
% %           [gBestScore1,gBest1,GlobalBestCost1]=PSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% %           BestSolutions3(i) = gBestScore1;
% 
% %         [gBestScore,gBest,GlobalBestCost]= CPSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% %         BestSolutions4(i) = gBestScore;
% %         [BestSolACO,BestAnt,BestCostACO] = ACO(SearchAgents_no, Max_iteration,Q,tau0,alpha,rho,lb,ub,dim,fobj);
% %         BestSolutions5(i) = BestSolACO.Cost;
% %         [BestCost,Best_Hab,BestSol] = bbo( SearchAgents_no, Max_iteration,lb,ub,dim,fobj);
% %         BestSolutions6(i) = BestSol.Cost;
%         [BestSolDE,DBestSol,BestCostDE] = DE(SearchAgents_no, Max_iteration,beta_min,beta_max,pCR,lb,ub,dim,fobj);
%         BestSolutions(i) = BestSolDE.Cost ;
% % 
% % [Best_scoreDA,Best_posDA,DA_cg_curve]=DA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% %          BestSolutions8(i) = Best_scoreDA ;
% %     [Best_scoreSCA,Best_posSCA,SCA_cg_curve]=SCA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% %      BestSolutions9(i) = Best_scoreSCA ;
% %     [Best_score,Best_posSSA,SSA_cg_curve]=SSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% %      BestSolutions10(i) = Best_score ;
% 
% 
% % 
% %         W=Lbest(1:2*Hnode);
% %         B= Lbest(2*Hnode+1:3*Hnode+1);
% % W=PcgCurve(1:2*Hnode);
% % B=PcgCurve(2*Hnode+1:3*Hnode+1); 
% % W=gBest1(1:2*Hnode);
%  B=gBest1(2*Hnode+1:3*Hnode+1);
% % W=gBest(1:2*Hnode);
% % B=gBest(2*Hnode+1:3*Hnode+1);
% % W=BestAnt(1:2*Hnode);
% % B=BestAnt(2*Hnode+1:3*Hnode+1);
% % W=Best_Hab(1:2*Hnode);
% % B=Best_Hab(2*Hnode+1:3*Hnode+1);
% W=DBestSol(1:2*Hnode);
% B=DBestSol(2*Hnode+1:3*Hnode+1);
% % 
% %  W=DA_cg_curve(1:2*Hnode);
% %  B=DA_cg_curve(2*Hnode+1:3*Hnode+1);
% %  W=SCA_cg_curve(1:2*Hnode);
% %  B=SCA_cg_curve(2*Hnode+1:3*Hnode+1);
% %  W=SSA_cg_curve(1:2*Hnode);
% %  B=SSA_cg_curve(2*Hnode+1:3*Hnode+1);
% 
% 
%         
%         for pp=1:N
%             yNN(pp)=my_simulate(1,15,1, W,B,xf(pp));
%         end
% %           
% %         disp(['GSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Fbest),')'])  
% % disp(['PSO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(GBEST.O), ')'])  
% % % disp(['PSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore1),')'])  
% % disp(['CPSOGSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(gBestScore), ')'])
% % disp(['ACO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolACO.Cost),')'])
% % disp(['BBO is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSol.Cost),')'])
% disp(['DE is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(BestSolDE.Cost),')'])
% % disp(['DA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_scoreDA),')']) 
% % disp(['SCA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_scoreSCA),')'])
% % disp(['SSA is training FNN (Run # = ', num2str(i),' ,MSE = ', num2str(Best_score),')'])
%      end
%     
% %     Average= mean(BestSolutions2);
% %     StandDP=std(BestSolutions2);
% %     Med = median(BestSolutions2); 
% %     [BestValueP I] = min(BestSolutions2);
% %     [WorstValueP IM]=max(BestSolutions2);
% end


% if Function_name=='F9'  %% Sphere
% 
% Hnode=15;
% dim = 4*Hnode+1
%  
% %for test 3 times more than the training samples
% %   xf1=[0:0.01:pi];
% %   yf1=sin(2.*xf1);
% % %   yf1=yf1.*exp(-xf1);
% %  xf1=[-2:0.05:2];
% % %  yf1=sin(2.*xf1);
% % %  yf1=yf1.*exp(-xf1);
% %  yf1=xf1.^2;
% % %yf1=xf1.^4-6.*xf1.^2+3;
% %yf1=sigmf(xf1,[1 0]);
% 
%  xf1=[-2:0.05:2];
%  yf1=xf1.^2; 
%   [xf1,yf1] = meshgrid(-2:.1:2);
% % yf1=xf1.^4-6.*xf1.^2+3;
% %yf1=sigmf(xf1,[1 0]);
%    [M N]=size(xf1);
% for i=1:M
%     for j=1:N
%         L=[xf1(i,j) yf1(i,j)];
%         zf1(i,j)=sum(L.^2);
%     end
% end
% 
%    
%  [xf,PS] = mapminmax(xf1);
%  [yf,PS2]= mapminmax(yf1);
%  [zf,PS3]= mapminmax(zf1);
%   
%   
%   zNN=zeros(1,10);
%   test_error=zeros(1,Runno);
%     for i=1:Runno
%         [Fbest,Lbest,BestChart]=GSA(SearchAgents_no,Max_iteration,ElitistCheck,min_flag,Rpower,lb,ub,dim,fobj);
% %         BestSolutions(i) = Fbest;
% %         [PcgCurve,GBEST]=pso(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% %         BestSolutions(i) = GBEST.O;
% %           [gBestScore1,gBest1,GlobalBestCost1]=PSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% %           BestSolutions(i) = gBestScore1;
% %         [gBestScore,gBest,GlobalBestCost]= CPSOGSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% %         BestSolutions1(i) = gBestScore;
%         for ww=1:3*Hnode
%             W(ww)=gBest1(i,ww);
%         end
%         for bb=3*Hnode+1:4*Hnode+1
%             B(bb-3*Hnode)=gBest1(i,bb);
%         end
%         
% %         for pp=1:N
% %             zNN(pp)=my_simulate(W,B,xf(pp),Hnode);
% %         end
% for ii=1:M
%     for jj=1:N
%         L=[xf1(ii,jj) yf1(ii,jj)];
%          zNN(ii,jj)=my_simulate_2_inputs(W,B,xf1(ii,jj),yf1(i,jj),Hnode);
%     end
% end


%%%Start Sphere%%%
%         figure
%         set(axes,'FontName','Times New Roman');
%         
%         hold on
%         grid on;
%         xfDenorm = mapminmax('reverse',xf,PS); 
%         yfDenorm = mapminmax('reverse',yf,PS2);
%         zfDenorm = mapminmax('reverse',zNN,PS3);
%         test_error(1,i)=test_error(1,i)+sum(sum(abs( zfDenorm- zf1 )));  
%         %surfc(xf1,yf1,zf1);
%         %colormap('Summer');
%         surfc(xfDenorm,yfDenorm,zfDenorm);
%         view([-38,30])
%         
%         colormap('Winter');
%         plot(xf1,yf1,'DisplayName','Real curve','Color','b');
%         plot(xfDenorm,yfDenorm,'DisplayName','Approximated curve','Marker','.','LineStyle','-','Color','r');
%         eqtext = '$$sin(2x)e^{-x}$$'; 
%        
%         name='PSOGSA';
%   
%         
%         title([['\fontsize{15}\it ', name]],'FontName','Times New Roman');
%         xlabel('\fontsize{15}\it X');
%         ylabel('\fontsize{15}\it Y');
%         zlabel('\fontsize{15}\it Z');
%         %legend('toggle');
% %         set(legend,'FontAngle','italic','FontName','Times New Roman') 

%%End Sphere%%%%



     %%%%Sigmoid Function%%%
% figure (1)
%         set(axes,'FontName','Times New Roman');
%         
%        hold on
%         grid on;
%         xfDenorm = mapminmax('reverse',xf,PS); 
%         yfDenorm = mapminmax('reverse',yNN,PS2);
%         test_error(1,i)=test_error(1,i)+sum(abs( yfDenorm- yf1 ));
%         A_Test_Error=mean(test_error);
%         plot(xf1,yf1,'DisplayName','Real curve\bf','Color','b');
%         plot(xfDenorm,yfDenorm,'DisplayName','Approximated curve','Marker','.','LineStyle','-','Color','r');
%         %eqtext = '$$sin(2x)e^{-x}$$'; 
%        name='LCGSA'
%         
%         
%         title([['\fontsize{15}\it\bf ', name]],'FontName','Times New Roman');
%         xlabel('\fontsize{15}\it\bf X');
%         ylabel('\fontsize{15}\it\bf Y');
%         legend('toggle');
%         set(legend,'FontAngle','italic','FontName','Times New Roman') 
% %%%%%end sigmoid%%%%%



%%%%% Start Cosine%%%%
% figure (1)
%         set(axes,'FontName','Times New Roman');
%         
%         hold on
%         grid on;
%         xfDenorm = mapminmax('reverse',xf,PS); 
%         yfDenorm = mapminmax('reverse',yNN,PS2);
%         test_error(1,i)=test_error(1,i)+sum(abs( yfDenorm- yf1 )); 
%         A_Test_Error=mean(test_error);
%         plot(xf1,yf1,'DisplayName','Real curve','Color','b');
%         plot(xfDenorm,yfDenorm,'DisplayName','Approximated curve','Marker','.','LineStyle','-','Color','r');
%         %eqtext = '$$sin(2x)e^{-x}$$'; 
%         
%         name='LCGSA'
% 
%         title([['\fontsize{15}\it ', name]],'FontName','Times New Roman');
%         xlabel('\fontsize{15}\it X');
%         ylabel('\fontsize{15}\it Y');
%         legend('toggle');
%         set(legend,'FontAngle','italic','FontName','Times New Roman') 

%%%%% End Cosine%%%%

      %%%%% Start Sine%%%%
% figure
%         set(axes,'FontName','Times New Roman');
%         
%         hold on
%         grid on;
%         xfDenorm = mapminmax('reverse',xf,PS); 
%         yfDenorm = mapminmax('reverse',yNN,PS2);
%         test_error(1,i)=test_error(1,i)+sum(abs( yfDenorm- yf1 )); 
%         A_Test_Error=mean(test_error);
%         plot(xf1,yf1,'DisplayName','Real curve','Color','b');
%         plot(xfDenorm,yfDenorm,'DisplayName','Approximated curve','Marker','.','LineStyle','-','Color','r');
%         %eqtext = '$$sin(2x)e^{-x}$$'; 
%         
%         name='DE'
% 
%         title([['\fontsize{15}\it\bf ', name]],'FontName','Times New Roman');
%         xlabel('\fontsize{15}\it\bf X');
%         ylabel('\fontsize{15}\it\bf Y');
%         legend('toggle');
%         set(legend,'FontAngle','italic','FontName','Times New Roman') 

%%%%% End Sine%%%%  


% disp(['Best=',num2str( BestValueP)])
% disp(['Worst=',num2str(WorstValueP)])
% disp(['Average=',num2str( Average)])
% disp(['Standard_Deviation=',num2str( StandDP)])
% disp(['Median=',num2str(Med)])
% % 
% % % disp(['Mean_Classification_Rate=',num2str(A_Classification_rate)]);
%   disp(['Mean_Test_Error = ' , num2str(A_Test_Error)])


%    figure (2)
%  
%      semilogy(1:Max_iteration,BestChart,'DisplayName','GSA','Color','g','Marker','o','LineStyle','-','LineWidth',2.5,'MarkerSize',5);
% %       disp( ['Time_GSA =', num2str(toc)]); 
%          hold on
% % % % %     
%       semilogy(1:Max_iteration,PcgCurve,'DisplayName','PSO','Color','c','Marker','p','LineStyle','-','LineWidth',2.5,'MarkerSize',5); %Done (Cyan)
% %      disp( ['Time_PSO =', num2str(toc)]); 
% % % 
%       semilogy(1:Max_iteration,GlobalBestCost1,'DisplayName','PSOGSA','Color','b','Marker','*','LineStyle','-','LineWidth',3);
% %       disp( ['Time_PSOGSA =', num2str(toc)]); 
% % % % % % % 
%        semilogy(1:Max_iteration,GlobalBestCost,'DisplayName','CPSOGSA', 'Color',[0.6350 0.0780 0.1840],'Marker','d','MarkerSize',5,'LineStyle','-','LineWidth',2.5);
% %       disp( ['Time_CPSOGSA =', num2str(toc)]); 
% % % % % % % % 
%         semilogy(1:Max_iteration,BestCostACO,'DisplayName','ACO','Color','k','Marker','.','MarkerSize',5,'LineStyle','-','LineWidth',2.5);%Done (Black)
% %       disp( ['Time_ACO =', num2str(toc)]); 
% % % % % 
%      semilogy(1:Max_iteration,BestCost,'DisplayName','BBO','Color',[0.75, 0.75, 0],'Marker','<','MarkerSize',5,'LineStyle','-','LineWidth',2.5);%Done (Blue)
% %    disp( ['Time_BBO =', num2str(toc)]); 
% % % % 
%      semilogy(1:Max_iteration,BestCostDE,'DisplayName','DE','Color',[0.8500 0.3250 0.0980],'Marker','^','MarkerSize',5,'LineStyle','-','LineWidth',2.5); % Done Triplet 2)
% %   disp( ['Time_DE =', num2str(toc)]); 
% % % % % % % 
%      semilogy(1:Max_iteration,DA_cg_curve,'DisplayName','DA','Color',[0.4 0.1 0.5],'Marker','v','MarkerSize',5,'LineStyle','-','LineWidth',2.5); %Done (Triplet 6)
% %   disp( ['Time_DA =', num2str(toc)]); 
% % % % 
%       semilogy(1:Max_iteration,SCA_cg_curve,'DisplayName','SCA','Color',[0.6 0.8 1],'Marker','>','MarkerSize',5,'LineStyle','-','LineWidth',2.5); %Done (Triplet 4)
% %     disp( ['Time_SCA =', num2str(toc)]); 
% % % % % % % 
%       semilogy(1:Max_iteration,SSA_cg_curve,'DisplayName','SSA','Color','y','Marker','+','MarkerSize',5,'LineStyle','-','LineWidth',2.5); % (Yellow)
% %   disp( ['Time_SSA =', num2str(toc)]); 
% % % 
%    semilogy(1:Max_iteration,CBestChart,'DisplayName','LCGSA', 'Color', 'r','Marker','*','MarkerSize',5,'LineStyle','-','LineWidth',3); %Done (Red)
% %    disp( ['Time_LCGSA =', num2str(toc)]);
% %    
% %    title ('\fontsize{15}\bf XOR Dataset');
% %     title ('\fontsize{15}\bf Balloon Dataset');
% %     title ('\fontsize{15}\bf Iris Dataset');
% %    title ('\fontsize{15}\bf Breast Cancer Dataset');
% %   title ('\fontsize{15}\bf Heart Dataset');
% 
% % % %  title ('\fontsize{15}\bf Sigmoid Dataset');
%    title ('\fontsize{15}\bf Cosine Dataset');
% % % %  title ('\fontsize{15}\bf Sine Dataset');
%    xlabel('\fontsize{15}\bf Generation');
%    ylabel('\fontsize{15}\bf MSE Values');
% %         legend('\fontsize{15}\bf GSA','\fontsize{15}\bf PSO','\fontsize{15}\bf PSOGSA','\fontsize{15}\bf CPSOGSA','\fontsize{15}\bf BBO','\fontsize{15}\bf DE','\fontsize{15}\bf ACO','\fontsize{15}\bf DA','\fontsize{15}\bf SCA','\fontsize{15}\bf SSA','\fontsize{15}\bf LCGSA',1);
% %         legend('\fontsize{15}\bf GSA','\fontsize{15}\bf PSO','\fontsize{15}\bf LCGSA',1);
%    axis tight
% % grid on
% box on

% 
      figure
% % % % % %  %%BoxPlot
% % % % % %  
      boxplot([BestSolutions1',BestSolutions2',BestSolutions3',BestSolutions4',BestSolutions5',BestSolutions6',BestSolutions7',BestSolutions8',BestSolutions9',BestSolutions10',BestSolutions11'],...
       {'GSA','PSO','PSOGSA','CPSOGSA','ACO','BBO','DE','DA','SCA','SSA','LCGSA'})
           color = [([0.55 0.71 0]); ([0.55 0.71 0]); ([1 0 0]); ([0.4 0.1 0.5]);([0.4 0 0.5]); ([1 0.2 1]);([1 0.8 1]); ([0.6 0.8 1]);([0.6 0.8 1]);( [0 1 0.46]);([0.9290 0.6940 0.1250])]; 
% % % % % % % % % % % % % color = [([1 0 0]); ([0 1 0]);([0 0 1]);([0 1 1]);([1 0 1]);([1 1 0]);([0 0.4470 0.7410]);([0.8500 0.3250 0.0980]);([0.9290 0.6940 0.1250]);([0.4940 0.1840 0.5560]);([0.4660 0.6740 0.1880]);([0.3010 0.7450 0.9330]);([0.6350 0.0780 0.1840])];
 h = findobj(gca,'Tag','Box'); 
          for j=1:length(h) 
          patch(get(h(j),'XData'),get(h(j),'YData'),color(j));
         end 
% %     title ('\fontsize{15}\bf XOR Dataset');
% %      title ('\fontsize{15}\bf Balloon Dataset');
% % % % %         title ('\fontsize{15}\bf Iris Dataset');
% % % % %    title ('\fontsize{15}\bf Breast Cancer Dataset');
% %    title ('\fontsize{15}\bf Heart Dataset');
% 
 title ('\fontsize{15}\bf Sigmoid Dataset');
%  title ('\fontsize{15}\bf Cosine Dataset');
% % % % %  title ('\fontsize{15}\bf Sine Dataset');
% % % % % 
          xlabel('\fontsize{15}\bf Algorithm(s)');
          ylabel('\fontsize{15}\bf Best Fitness Values');
% % %    
          box on
% % % % Wilcoxon rank sum test
% % 
% % % disp(' Wilcoxon RankSum Test ')
%           [p]= signrank(BestSolutions3,BestSolutions1)
% % % 
% save BestSolutions1
% save BestSolutions2
% save BestSolutions3
% save BestSolutions4
% save BestSolutions5
% save BestSolutions6
% save BestSolutions7
% save BestSolutions8
% save BestSolutions9
% save BestSolutions10
% save BestSolutions11

