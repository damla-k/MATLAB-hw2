load('Multi_Level_Perc.mat','rand_ex','test_input')

function ima(A) 
% Translate vector to become nonnegative
% Scale to interval [0,20]
% Reshape the vector as a matrix and then show image

    az=squeeze(A);  
    az=reshape(az,16,16)';  

    az=(az-min(min(az))*ones(size(az)));
    az=(20/max(max(az)))*az;

    my_map =[1.0000    1.0000    1.0000
    0.8715    0.9028    0.9028
    0.7431    0.8056    0.8056
    0.6146    0.7083    0.7083
    0.4861    0.6111    0.6111
    0.3889    0.4722    0.5139
    0.2917    0.3333    0.4167
    0.1944    0.1944    0.3194
    0.0972    0.0972    0.1806
         0         0    0.0417];
    colormap(my_map)
    image(az) 
end 

ima(test_input(;,rand_ex)); 
title(['Digit image: ', num2str(test_target(rand_ex))]);