from torch import nn

import research


class ActorCriticPolicy(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        actor_class,
        critic_class,
        encoder_class=None,
        actor_kwargs={},
        critic_kwargs={},
        encoder_kwargs={},
        **kwargs,
    ) -> None:
        super().__init__()
        # Update all dictionaries with the generic kwargs
        self.action_space = action_space
        self.observation_space = observation_space

        encoder_kwargs.update(kwargs)
        actor_kwargs.update(kwargs)
        critic_kwargs.update(kwargs)

        self.encoder_class, self.encoder_kwargs = encoder_class, encoder_kwargs
        self.actor_class, self.actor_kwargs = actor_class, actor_kwargs
        self.critic_class, self.critic_kwargs = critic_class, critic_kwargs

        self.reset_encoder()
        self.reset_actor()
        self.reset_critic()

    def reset_encoder(self, device=None):
        encoder_class = (
            vars(research.networks)[self.encoder_class] if isinstance(self.encoder_class, str) else self.encoder_class
        )
        if encoder_class is not None:
            self._encoder = encoder_class(self.observation_space, self.action_space, **self.encoder_kwargs)
        else:
            self._encoder = nn.Identity()
        if device is not None:
            self._encoder = self._encoder.to(device)

    def reset_actor(self, device=None):
        observation_space = (
            self.encoder.output_space if hasattr(self.encoder, "output_space") else self.observation_space
        )
        actor_class = (
            vars(research.networks)[self.actor_class] if isinstance(self.actor_class, str) else self.actor_class
        )
        self._actor = actor_class(observation_space, self.action_space, **self.actor_kwargs)
        if device is not None:
            self._actor = self._actor.to(self.device)

    def reset_critic(self, device=None):
        observation_space = (
            self.encoder.output_space if hasattr(self.encoder, "output_space") else self.observation_space
        )
        critic_class = (
            vars(research.networks)[self.critic_class] if isinstance(self.critic_class, str) else self.critic_class
        )
        self._critic = critic_class(observation_space, self.action_space, **self.critic_kwargs)
        if device is not None:
            self._critic = self._critic.to(device)

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


class ActorCriticRewardPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        actor_class,
        critic_class,
        reward_class,
        encoder_class=None,
        actor_kwargs={},
        critic_kwargs={},
        encoder_kwargs={},
        reward_kwargs={},
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            actor_class,
            critic_class,
            encoder_class=encoder_class,
            actor_kwargs=actor_kwargs,
            critic_kwargs=critic_kwargs,
            encoder_kwargs=encoder_kwargs,
            **kwargs,
        )
        assert encoder_class is None, "Reward policies currently do not support encoders"

        reward_kwargs.update(kwargs)
        self.reward_class, self.reward_kwargs = reward_class, reward_kwargs
        self.reset_reward()

    def reset_reward(self, device=None):
        observation_space = (
            self.encoder.output_space if hasattr(self.encoder, "output_space") else self.observation_space
        )
        reward_class = (
            vars(research.networks)[self.reward_class] if isinstance(self.reward_class, str) else self.reward_class
        )
        self._reward = reward_class(observation_space, self.action_space, **self.reward_kwargs)
        if device is not None:
            self._reward = self._reward.to(device)

    @property
    def reward(self):
        return self._reward
