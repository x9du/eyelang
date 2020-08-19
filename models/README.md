# eyelang/models
Notes on the trained models

## eye-gaze_net_0.pth
- Accuracy 60%. Final loss 0.852, min loss 0.790. 2 epoch.
- Trainset MPIIGaze/Data/Normalized, all_norm_df, size (1, 36, 60), eye right.
- Testset MPIIGaze/Evaluation Subset/sample list for eye image, all_eval_df, size (1, 36, 60), eye 'right'.
- classes = ('left', 'right', 'up', 'down', 'center')
### NN
1. Convolution 2D 1: 32 filters, filter size 7 x 7, stride 1, padding 3. ReLU. Max-pool: 2 x 2, stride 2.
2. Convolution 2D 2: 32 filters. Filter shape 5 x 5, stride 1, padding 2. ReLU. Max-pooling layer: 2 x 2, stride 2.
3. Convolution 2D 3: 64 filters. Filter shape 3 x 3, stride 1, padding 1. ReLU. Max-pooling layer: 2 x 2, stride 2.
4. Linear 1: 64 * 4 * 7, 150. ReLU.
5. Linear 2: 150, 80. ReLU.
6. Linear 3: 80, 5.
### Labels
- left: x < -0.1
- right: x > 0.12
- up: y < -0.05
- down: y > 0.1
- center: else