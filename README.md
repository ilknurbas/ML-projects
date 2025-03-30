# Machine Learning Projects
A collection of machine learning projects exploring reinforcement learning, audio processing, and deep learning techniques with CNNs and RNNs. These projects tackle real-world challenges in decision-making, image denoising, and sound analysis. *Note that some sections of the code are based on templates provided.*

**`#1: Reinforcement learning - OpenAI Gym – Frozen Lake Environment`**
Implemented Q-learning to solve the stochastic navigation problem in OpenAI Gym’s Frozen Lake environment. The task involved learning an optimal policy to navigate through a 4x4 grid, starting from the top-left corner and reaching the bottom-right goal. Experiments were conducted using both deterministic (non-slippery) and non-deterministic (slippery) versions of the environment. Multiple runs were performed to assess performance and the impact of hyperparameters like the learning rate (α) and discount factor (γ), with graphs generated to visualize average reward improvement across training episodes.

**`#2:` Reinforcement learning (Q-learning, Q-network) - OpenAI Gym – Taxi`**
Developed Q-learning and Deep Q-Networks (DQN) for optimal decision-making in OpenAI Gym's Taxi environment. Q-learning was used to solve the Taxi problem by updating a Q-table to find the optimal policy, with performance monitored through reward plots. Then, a DQN replaced the Q-table with a neural network, trained either by sampling from the Q-table or using the Q-table to drive the Taxi and collect input-output pairs. The task aimed to compare both approaches and determine how many samples were needed for the DQN to match the Q-table’s performance.

**`#3:` CNN-based Autoencoder for Denoising Noisy Images using the Fashion MNIST Dataset`**
Trained a convolutional autoencoder to remove noise from images using the Fashion MNIST dataset, applying deep learning techniques for unsupervised feature extraction and denoising.

**`#4:` Audio classification using convolutional neural networks (CNNs)`**
The CNN-based model was designed to classify audio signals as either speech or music. The model processes the audio into log-mel spectrograms before feeding them into two CNN blocks. PyTorch was used to build the model, and the Adam optimizer was chosen for training. The Binary Cross Entropy loss function was used during training, and hyperparameters were adjusted to optimize the model’s performance.

**`#5:` Audio detection with convolutional and recurrent neural networks (RNNs)`**
Developed an audio event detection system using a CNN-RNN hybrid model, capable of identifying sound events over time with fine temporal resolution. The system analyzes both spatial and temporal audio features for accurate event detection and classification.

**`#6:` Singing voice separation with RNN based denoising auto-encoders`**
Developed an RNN-based denoising autoencoder (DAE) for singing voice separation, predicting the magnitude spectrum of the clean vocal signal from a mixture. The model was trained on the DSD100 dataset using deep learning techniques for source separation in audio processing.

