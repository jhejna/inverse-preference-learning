from torch import nn
import research

class ActorCriticPolicy(nn.Module):

    def __init__(self, observation_space, action_space, 
                       actor_class, critic_class, encoder_class=None, 
                       actor_kwargs={}, critic_kwargs={}, encoder_kwargs={}, **kwargs) -> None:
        super().__init__()
        # Update all dictionaries with the generic kwargs
        self.action_space = action_space
        self.observation_space = observation_space

        encoder_kwargs.update(kwargs)
        actor_kwargs.update(kwargs)
        critic_kwargs.update(kwargs)
        self.actor_class, self.actor_kwargs = actor_class, actor_kwargs
        self.critic_class, critic_kwargs = critic_class, critic_kwargs
        self.encoder_class, encoder_kwargs = encoder_class, encoder_kwargs

        self.reset_encoder()
        self.reset_actor()
        self.reset_critic()

    def reset_encoder(self):
        encoder_class = vars(research.networks)[self.encoder_class] if isinstance(self.encoder_class, str) else self.encoder_class
        if encoder_class is not None:
            self._encoder = encoder_class(self.observation_space, self.action_space, **self.encoder_kwargs)
            # Modify the observation space
            if hasattr(self._encoder, "output_space"):
                observation_space = self._encoder.output_space
        else:
            self._encoder = nn.Identity()

    def reset_actor(self):
        actor_class = vars(research.networks)[self.actor_class] if isinstance(self.actor_class, str) else self.actor_class
        self._actor = actor_class(self.observation_space, self.action_space, **self.actor_kwargs)

    def reset_critic(self):
        critic_class = vars(research.networks)[self.critic_class] if isinstance(self.critic_class, str) else self.critic_class
        self._critic = critic_class(self.observation_space, self.action_space, **self.critic_kwargs)

    @property
    def actor(self):
        return self._actor
    
    @property
    def critic(self):
        return self._critic

    @property
    def encoder(self):
        return self._encoder
        
    def predict(self, obs, **kwargs):
        obs = self._encoder(obs)
        if hasattr(self._actor, "predict"):
            return self._actor.predict(obs, **kwargs)
        else:
            return self._actor(obs)
