%   Biometric Systems
%   CYS616
%   Wael Ghazi Ahmed Alnahari
%   440804845
%   Dr. Mostafa Abdel-Halim Mostafa Ahmad
%   A CONVOLUTIONAL NEURAL NETWORK FOR IRIS RECOGNITION
clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables. Or clearvars if you want.
workspace;  % Make sure the workspace panel is showing.
reset(gpuDevice(1));  % Reset GPU memory

%% Load train Data
    categories = 		{'001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','025','026','027','028','029','030','031','032','033','034','035','036','037','038','039','040','041','042','043','044','045','046','047','048','049','050','051','052','053','054','055','056','057','058','059','060','061','062','063','064','065','066','067','068','069','070','071','072','073','074','075','076','077','078','079','080','081','082','083','084','085','086','087','088','089','090','091','092','093','094','095','096','097','098','099','100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138','139','140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','166','167','168','169','170','171','172','173','174','175','176','177','178','179','180','181','182','183','184','185','186','187','188','189','190','191','192','193','194','195','196','197','198','199','200','201','202','203','204','205','206','207','208','209','210','211','212','213','214','215','216','217','218','219','220','221','222','223'};

    num_train = 7*223;
    num_test = 3;
    imdsTrain = imageDatastore(fullfile(pwd,"TrainData", categories),'IncludeSubfolders',true,'FileExtensions','.bmp','LabelSource','foldernames');
    img = readimage(imdsTrain,1);
    [x , y] = size(img);
    inputSize=[x y 1];
%      imdsTrain=augmentedImageDatastore(inputSize, imdsTrain,'ColorPreprocessing','rgb2gray');
%  I = imread(imdsTrain.Files{1});
%   [x , y] = size(I);
%         for i=1:size(imdsTrain.Files,1)
%             I = imread(imdsTrain.Files{i});
%             [a b c]=size(I);
%             if c>1
%             I = rgb2gray(I);
%             end
%             I = imresize(I, [x y]);
%             imwrite(I, imdsTrain.Files{i});
%         end
       
    
    
%% Load Test Data
    imdsValidation = imageDatastore(fullfile(pwd,"TestData", categories),'IncludeSubfolders',true,'FileExtensions','.bmp','LabelSource','foldernames');
%     imdsValidation = augmentedImageDatastore(inputSize, imdsValidation,'ColorPreprocessing','rgb2gray');
%        A = imread(imdsValidation.Files{1});
%       for i=1:size(imdsValidation.Files,1)
%             A = imread(imdsValidation.Files{i});
%             [a b c]=size(A);
%             if c>1
%             A = rgb2gray(A);
%             end
%             A = imresize(A, [x y]);
%             imwrite(A, imdsValidation.Files{i});
%           end
%       
    

%% Calculate the number of images in each category. 
    labelCount = countEachLabel(imdsTrain);

    %% Define Network Architecture
    layers = [
    imageInputLayer([x y 1]);  
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer(); 
    
    maxPooling2dLayer(5,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer();
    
        averagePooling2dLayer(5,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer();
    
    maxPooling2dLayer(5,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer();
    
         averagePooling2dLayer(5,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer();
    

    
    fullyConnectedLayer(223,'BiasLearnRateFactor',2);
    softmaxLayer
    classificationLayer];

    %% Specify Training Options
	options = trainingOptions('sgdm', ...
	'InitialLearnRate', 0.0001, ...
	'ValidationData',imdsValidation, ...
	'ValidationFrequency',30, ...
	'Shuffle','every-epoch', ...
	'MaxEpochs', 94, ...
	'MiniBatchSize', 8, ...
	'ValidationFrequency',50, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.05, ...
    'LearnRateDropPeriod',60, ...
    'ExecutionEnvironment','parallel',...
	'Verbose', true, 'Plots','training-progress');  

%% Train Network Using Training Data
    [net_Wael, info] = trainNetwork(imdsTrain,layers,options);
    save net_Wael
    
    %% Classify validation
    labels = classify(net_Wael,imdsValidation);

    %% *Test one at a time*
    for i=1:223
        ii = randi(num_test*2);
        im = imread(imdsValidation.Files{ii});
        figure, imshow(im);
        if labels(ii) == imdsValidation.Labels(ii)
           colorText = 'g'; 
        else
            colorText = 'r';
        end
        title(char(labels(ii)),'Color',colorText);
    end
    
    %% Compute Accuracy
    YValidation = imdsValidation.Labels;
    accuracy02 = sum(labels == YValidation)/numel(YValidation);
    display(accuracy02);
