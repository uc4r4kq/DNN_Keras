
% vmodel = load('vmodel909.mat');
% vmodel = vmodel.vmodel;
% predicao = load('vmodel909_preds.mat');
% predicao = predicao.vmodel;
% subplot(1,2,1);imagesc(vmodel);subplot(1,2,2);imagesc(predicao);

vmodel = load('vmodel901_preds.mat');
vmodel = vmodel.vmodel;
vmodel2 = load('vmodel900_preds.mat');
vmodel2 = vmodel2.vmodel;
predicao = load('vmodel909_preds.mat');
predicao = predicao.vmodel;
subplot(1,3,1);imagesc(vmodel);subplot(1,3,2);imagesc(vmodel2);subplot(1,3,3);imagesc(predicao);