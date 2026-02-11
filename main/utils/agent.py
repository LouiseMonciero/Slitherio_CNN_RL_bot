import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import cv2


class CNN_QNetwork(nn.Module):
    """Réseau de neurones convolutif pour estimer les Q-values."""
    
    def __init__(self, input_channels=3, n_actions=8):
        super(CNN_QNetwork, self).__init__()
        
        # Couches convolutives pour traiter les 3 canaux d'image
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        # Calculer la taille après convolutions (pour une image 84x84)
        self.fc_input_dim = self._get_conv_output((input_channels, 84, 84))
        
        # Couches fully connected
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim + 1, 512),  # +1 pour la direction
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
    
    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv(x)
            return int(np.prod(x.shape[1:]))
    
    def forward(self, frames, direction):
        """
        frames: tensor (batch, 3, H, W) - les 3 canaux empilés
        direction: tensor (batch, 1) - direction normalisée
        """
        x = self.conv(frames)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, direction], dim=1)
        return self.fc(x)


class ReplayBuffer:
    """Buffer pour stocker les expériences."""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class Agent:
    """Agent DQN pour Slither.io."""
    
    # Actions possibles : 8 directions (angles en degrés)
    ACTIONS = [0, 45, 90, 135, 180, -135, -90, -45]
    
    def __init__(
        self,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=32,
        target_update=10,
        device=None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = len(self.ACTIONS)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Epsilon pour exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Réseaux
        self.policy_net = CNN_QNetwork(input_channels=3, n_actions=self.n_actions).to(self.device)
        self.target_net = CNN_QNetwork(input_channels=3, n_actions=self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        
        # Replay buffer
        self.memory = ReplayBuffer()
        
        # Compteurs
        self.steps_done = 0
        self.episodes_done = 0
        
        # Pour calculer la récompense
        self.previous_score = 0
    
    def preprocess_state(self, game):
        """
        Convertit l'état du jeu en tenseurs pour le réseau.
        Retourne (frames_tensor, direction_tensor).
        """
        # Redimensionner les frames à 84x84
        def resize_frame(frame):
            if frame is None:
                return np.zeros((84, 84), dtype=np.float32)
            resized = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
            return resized.astype(np.float32) / 255.0
        
        import cv2
        
        me_frame = resize_frame(game.preprocessed_me_frame)
        dots_frame = resize_frame(game.preprocessed_dots_frame)
        snake_frame = resize_frame(game.preprocessed_snake_frame)
        
        # Empiler les 3 canaux
        frames = np.stack([me_frame, dots_frame, snake_frame], axis=0)
        frames_tensor = torch.FloatTensor(frames).unsqueeze(0).to(self.device)
        
        # Normaliser la direction (-180 à 180 -> -1 à 1)
        direction_normalized = game.direction / 180.0
        direction_tensor = torch.FloatTensor([[direction_normalized]]).to(self.device)
        
        return frames_tensor, direction_tensor
    
    def calculate_reward(self, game, is_game_over):
        """
        Calcule la récompense basée sur l'évolution du score.
        """
        reward = 0.0
        
        if is_game_over:
            # Pénalité importante pour la mort
            reward = -100.0
        else:
            # Récompense basée sur le changement de score
            score_diff = game.score - self.previous_score
            
            if score_diff > 0:
                # Bonus pour avoir mangé (score augmente)
                reward = score_diff * 10.0
            else:
                # Petite récompense pour survivre
                reward = 0.1
        
        self.previous_score = game.score
        return reward
    
    def select_action(self, game):
        """
        Sélectionne une action avec epsilon-greedy.
        Retourne l'index de l'action et l'angle correspondant.
        """
        if random.random() < self.epsilon:
            # Exploration : action aléatoire
            action_idx = random.randrange(self.n_actions)
        else:
            # Exploitation : meilleure action selon le réseau
            with torch.no_grad():
                frames, direction = self.preprocess_state(game)
                q_values = self.policy_net(frames, direction)
                action_idx = q_values.argmax(dim=1).item()
        
        self.steps_done += 1
        return action_idx, self.ACTIONS[action_idx]
    
    def store_transition(self, state, action_idx, reward, next_state, done):
        """Stocke une transition dans le replay buffer."""
        self.memory.push(state, action_idx, reward, next_state, done)
    
    def train_step(self):
        """Effectue une étape d'entraînement."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Échantillonner un batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convertir en tenseurs
        frames_batch = torch.cat([s[0] for s in states]).to(self.device)
        directions_batch = torch.cat([s[1] for s in states]).to(self.device)
        
        next_frames_batch = torch.cat([s[0] for s in next_states]).to(self.device)
        next_directions_batch = torch.cat([s[1] for s in next_states]).to(self.device)
        
        actions_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_batch = torch.FloatTensor(rewards).to(self.device)
        dones_batch = torch.FloatTensor(dones).to(self.device)
        
        # Q-values actuelles
        current_q = self.policy_net(frames_batch, directions_batch).gather(1, actions_batch)
        
        # Q-values cibles (Double DQN)
        with torch.no_grad():
            next_q = self.target_net(next_frames_batch, next_directions_batch).max(1)[0]
            target_q = rewards_batch + (1 - dones_batch) * self.gamma * next_q
        
        # Calcul de la loss et backprop
        loss = self.loss_fn(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Met à jour le réseau cible."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Réduit epsilon après chaque épisode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes_done += 1
        
        # Mise à jour du réseau cible périodiquement
        if self.episodes_done % self.target_update == 0:
            self.update_target_network()
    
    def reset_episode(self):
        """À appeler au début d'un nouvel épisode."""
        self.previous_score = 0
    
    def save(self, filepath):
        """Sauvegarde le modèle."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episodes_done': self.episodes_done,
            'steps_done': self.steps_done,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Charge le modèle."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.episodes_done = checkpoint['episodes_done']
        self.steps_done = checkpoint['steps_done']
        print(f"Model loaded from {filepath}")