# SSRN
This is a keras reprodction of TRGS paper:*Spectral–Spatial Residual Network for Hyperspectral Image Classification: A 3-D Deep Learning Framework*
## Test result on Indian Pines Dataset using the author's best parameter configurations:
**Test loss**: 0.14532101154327393  
**Test acc**: 95.90243697166443%

**Classification result:**

                              precision    recall  f1-score   support

                     Alfalfa       1.00      0.50      0.67        32
                 Corn-notill       0.96      0.92      0.94      1000
                Corn-mintill       0.96      0.98      0.97       581
                        Corn       0.95      0.99      0.97       166
               Grass-pasture       0.96      0.96      0.96       338
                 Grass-trees       1.00      0.98      0.99       511
         Grass-pasture-mowed       1.00      0.47      0.64        19
               Hay-windrowed       0.92      1.00      0.96       334
                        Oats       1.00      0.93      0.96        14
              Soybean-notill       0.93      0.94      0.94       681
             Soybean-mintill       0.98      0.97      0.97      1719
               Soybean-clean       0.92      0.96      0.94       416
                       Wheat       1.00      0.99      0.99       143
                       Woods       0.97      0.98      0.97       886
    Building-Gras-Tree-Drive       0.91      0.91      0.91       270
          Stone-Steel-Towers       0.97      1.00      0.98        65
          
          
                    accuracy                           0.96      7175
                   macro avg       0.96      0.90      0.92      7175
                weighted avg       0.96      0.96      0.96      7175

**Confusion matrix:**
![CM](https://github.com/lzp-cumtb/SSRN/blob/main/pics/confusion_mat_without_norm.png)

**Predict map:**
![PM](https://github.com/lzp-cumtb/SSRN/blob/main/pics/pred_map.jpg)
