import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class PolicyStabilityCallback(BaseCallback):
    """
    A custom callback to log the mean and variance of the policy loss,
    providing a measure of learning stability.

    :param log_interval: The frequency in steps to log the statistics.
    :param verbose: Verbosity level.
    """
    def __init__(self, log_interval: int = 4096, verbose: int = 0):
        super(PolicyStabilityCallback, self).__init__(verbose)
        self.log_interval = log_interval
        # A list to store policy losses between logging intervals
        self.policy_losses_since_last_log = []

    def _on_rollout_end(self) -> None:
        """
        This method is called at the end of each rollout, which is where
        Stable-Baselines3 calculates and logs the training losses.
        We capture the policy loss here.
        """
        # The 'policy_loss' is a standard value logged by the PPO algorithm
        # We access it from the model's internal logger
        if 'policy_loss' in self.model.logger.name_to_value:
            loss_value = self.model.logger.name_to_value['train/policy_loss']
            self.policy_losses_since_last_log.append(loss_value)

    def _on_step(self) -> bool:
        """
        This method is called after each step. We use it to periodically
        compute and log our custom statistics.
        """
        # Check if it's time to log our custom stats
        if self.log_interval > 0 and self.n_calls % self.log_interval == 0:
            
            # Ensure we have data to process
            if len(self.policy_losses_since_last_log) > 1:
                # Calculate the mean and variance of the collected policy losses
                mean_loss = np.mean(self.policy_losses_since_last_log)
                var_loss = np.var(self.policy_losses_since_last_log)

                # Record these new metrics to the main logger (e.g., TensorBoard)
                # They will appear under a "custom" tab in TensorBoard
                self.logger.record('custom/policy_loss_mean', mean_loss)
                self.logger.record('custom/policy_loss_variance', var_loss)
                
                # Clear the list to start collecting for the next interval
                self.policy_losses_since_last_log = []
            
        return True