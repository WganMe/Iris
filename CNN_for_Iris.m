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
reset(gpuDevice);  % Reset GPU memory

%% Load train Data
categories = {'001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020', ...
			  '021','022','023','024','025','026','027','028','029','030','031','032','033','034','035','036','037','038','039','040', ...
              '041','042','043','044','045','046','047','048','049','050','051','052','053','054','055','056','057','058','059','060', ...
              '061','062','063','064','065','066','067','068','069','070','071','072','073','074','075','076','077','078','079','080', ...
              '081','082','083','084','085','086','087','088','089','090','091','092','093','094','095','096','097','098','099','100', ...
              '101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120', ...
              '121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138','139','140', ...
              '141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160', ...
              '161','162','163','164','165','166','167','168','169','170','171','172','173','174','175','176','177','178','179','180', ...
              '181','182','183','184','185','186','187','188','189','190','191','192','193','194','195','196','197','198','199','200', ...
              '201','202','203','204','205','206','207','208','209','210','211','212','213','214','215','216','217','218','219','220', ...
              '221','222','223'};

    imdsTrain = imageDatastore(fullfile(pwd,"Dataset\TrainData", categories),'IncludeSubfolders',true,'FileExtensions','.bmp','LabelSource','foldernames');
num_train = size(imdsTrain.Files,1);

%% Size of first image in dataset
    img = readimage(imdsTrain,1);
    [x , y , z] = size(img);

%% Load Test Data
    imdsValidation = imageDatastore(fullfile(pwd,"Dataset\TestData", categories),'IncludeSubfolders',true,'FileExtensions','.bmp','LabelSource','foldernames');
num_test = size(imdsValidation.Files,1);
%% Calculate the number of images in each category. 
    labelCount = countEachLabel(imdsTrain);

    %% Define Network Architecture
    layers = [
 
    imageInputLayer([x y z]);  
 
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
 
 
    fullyConnectedLayer(size(categories,2),'BiasLearnRateFactor',2);
    softmaxLayer
    classificationLayer];
 
    %% Specify Training Options
    options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',50, ...
    'Shuffle','every-epoch', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 8, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',50, ...
    'ExecutionEnvironment','parallel', ...
    'Verbose', true, 'Plots','training-progress');


%% Train Network Using Training Data
    [net_Wael, info] = trainNetwork(imdsTrain,layers,options);
    save net_Wael

    %% Classify validation
    labels = classify(net_Wael,imdsValidation);

        %% *Test one at a time*
true=0;
false=0;
    for i = 1:size(imdsValidation.Files,1)
        im = imread(imdsValidation.Files{i});
            if labels(i) == imdsValidation.Labels(i)
               true=true+1;
            else
                false=false+1;  
            end
    end

    % Find the square root of the false matches to determine the subplots

    f0=floor(sqrt(false));
    if f0 == sqrt(false)
        f1=f0;
    else
        f1=f0+1;
    end

    % Recalbirating true and false
    true=0;
    false=0;

    %start plotting mismatches
    figure('Name','All Mismatch Pictures')
    for i = 1:size(imdsValidation.Files,1)
        im = imread(imdsValidation.Files{i});
        if labels(i) ~= imdsValidation.Labels(i)
            
            colorText = 'r';
            false=false+1;  
            if false > f0*f1
                    break     
            else 
            subplot(f1,f0,false);
            imshow(im);
            title(char(labels(i)),'Color',colorText);
            end
        end
    end
            xlabel('ALL mismatch incidents')

    %start plotting sample of correct match
    figure('Name','Sample of Correct Match')
    true=0;
    false=0;
        for i = 1:size(imdsValidation.Files,1)
            ii = randi(size(imdsValidation.Files,1));
            im = imread(imdsValidation.Files{ii});
            if labels(ii) == imdsValidation.Labels(ii)
               colorText = 'g'; 
               true=true+1;
               if true > f0*f1
                    break    
               else 
                       subplot(f1,f0,true);
                       imshow(im);
                       title(char(labels(ii)),'Color',colorText);
               end
            end
        end
            xlabel('Sample of Correct Incidents')
   
    %% Compute Accuracy
    YValidation = imdsValidation.Labels;
    accuracy02 = sum(labels == YValidation)/numel(YValidation);
    display(accuracy02);
