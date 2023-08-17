import torch
import pytorch_lightning as pl
import numpy as np

class LitDiffusionModel(pl.LightningModule):
    def __init__(self, n_dim=3, n_steps=200, lbeta=1e-5, ubeta=1e-2, noise_schedule=0):
        super().__init__()
        """
        If you include more hyperparams (e.g. `n_layers`), be sure to add that to `argparse` from `train.py`.
        Also, manually make sure that this new hyperparameter is being saved in `hparams.yaml`.
        """
        self.save_hyperparameters()

        """
        Your model implementation starts here. We have separate learnable modules for `time_embed` and `model`.
        You may choose a different architecture altogether. Feel free to explore what works best for you.
        If your architecture is just a sequence of `torch.nn.XXX` layers, using `torch.nn.Sequential` will be easier.
        
        `time_embed` can be learned or a fixed function based on the insights you get from visualizing the data.
        If your `model` is different for different datasets, you can use a hyperparameter to switch between them.
        Make sure that your hyperparameter behaves as expecte and is being saved correctly in `hparams.yaml`.
        """

        """
        Model 1
        """

        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, n_dim)
        )
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2*n_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, n_dim)
        )


        """
        Model 2
        """
        # self.time_embed = torch.nn.Sequential(
        #     torch.nn.Linear(1, 32),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(32, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, 128),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, 32),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(32, 8),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(8, n_dim)
        # )
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(2*n_dim, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 128),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, 32),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(32, 16),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(16, n_dim)
        # )

        """
        Model 3     
        """

        # self.time_embed = torch.nn.Sequential(
        #     torch.nn.Linear(1, 32),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(32, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, 128),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, 32),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(32, 8),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(8, n_dim)
        # )
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(2*n_dim, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 512),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(512, 128),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, 32),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(32, 16),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(16, n_dim)
        # )


        """
        Be sure to save at least these 2 parameters in the model instance.
        """
        self.n_steps = n_steps
        self.n_dim = n_dim

        self.alphas = None
        self.betas = None

        """
        Sets up variables for noise schedule
        """
        self.init_alpha_beta_schedule(lbeta, ubeta, noise_schedule)

    def forward(self, x, t):
        """
        Similar to `forward` function in `nn.Module`. 
        Notice here that `x` and `t` are passed separately. If you are using an architecture that combines
        `x` and `t` in a different way, modify this function appropriately.
        """
        if not isinstance(t, torch.Tensor):
            t = torch.reshape(torch.FloatTensor([t]).expand(x.size(0)), (x.size(0), 1))
        # print(t.shape)
        t_embed = self.time_embed(t.float())
        return self.model(torch.cat((x, t_embed), dim=1).float())

    def init_alpha_beta_schedule(self, lbeta, ubeta, noise_schedule):
        """
        Set up your noise schedule. You can perhaps have an additional hyperparameter that allows you to
        switch between various schedules for answering q4 in depth. Make sure that this hyperparameter 
        is included correctly while saving and loading your checkpoints.
        """
        if (noise_schedule == 0):                                                   #Linear
            step = (ubeta - lbeta)/(self.n_steps)
            self.betas = torch.arange(lbeta, ubeta, step)
        elif (noise_schedule == 1):                                                 #Quadratic
            self.betas = torch.linspace(lbeta**0.5, ubeta**0.5, self.n_steps) ** 2  
        elif (noise_schedule == 2):                                                 #Bi-quadratic                         
            self.betas = torch.linspace(lbeta**0.25, ubeta**0.25, self.n_steps) ** 4
        elif (noise_schedule == 3):                                                 #Sigmoid
            vals = torch.linspace(-6, 6, self.n_steps)
            self.betas = torch.sigmoid(vals) * (ubeta - lbeta) + lbeta
        elif (noise_schedule == 4):                                                 #Cosine
            s = 0.008                                            
            new_steps = self.n_steps + 1
            x = torch.linspace(0, self.n_steps, new_steps)
            alphas_cumprod = torch.cos(((x / self.n_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(self.betas, 0.0001, 0.9999)
        self.alphas = torch.cumprod(1 - self.betas, dim = 0)

    def q_sample(self, x, t, epsilon):
        """
        Sample from q given x_t.
        """
        """
        n_samples - Number of samples in one batch
        t - (n_samples, 1)
        x, epsilon - (n_samples, n_dim)
        """
        Alpha = torch.reshape(self.alphas[torch.flatten((t-1).type(torch.int64))], (x.shape[0], 1))
        x_t = torch.sqrt(Alpha)*x + torch.sqrt(1 - Alpha)*epsilon
        return x_t

    def p_sample(self, x, t):
        """
        Sample from p given x_t.
        """
        x_t = x
        n_samples = x.shape[0]
        intermediate = torch.zeros(self.n_steps, n_samples, self.n_dim)
        results = torch.zeros(n_samples, self.n_dim)
        intermediate[0] = x_t
        for j in range(t):
            coeff1 = (1/torch.sqrt(1-self.betas[t - j - 1]))
            coeff2 = (self.betas[t - j - 1])/(torch.sqrt(1 - self.alphas[t - j - 1]))

            x_t = coeff1 * (x_t - coeff2*self.forward(x_t, float(t - j - 1))) + torch.sqrt(self.betas[t-j-1])*torch.randn(n_samples, self.n_dim)
            if (j < t - 1):
                intermediate[j + 1] = x_t
            else :
                results = x_t
        return results, intermediate

    def training_step(self, batch, batch_idx):
        """
        Implements one training step.
        Given a batch of samples (n_samples, n_dim) from the distribution you must calculate the loss
        for this batch. Simply return this loss from this function so that PyTorch Lightning will 
        automatically do the backprop for you. 
        Refer to the DDPM paper [1] for more details about equations that you need to implement for
        calculating loss. Make sure that all the operations preserve gradients for proper backprop.
        Refer to PyTorch Lightning documentation [2,3] for more details about how the automatic backprop 
        will update the parameters based on the loss you return from this function.

        References:
        [1]: https://arxiv.org/abs/2006.11239
        [2]: https://pytorch-lightning.readthedocs.io/en/stable/
        [3]: https://www.pytorchlightning.ai/tutorials
        """
        """
        batch-shape, epsilon-shape, q_sample - (n_samples, n_dim)
        """
        epsilon = torch.randn(batch.shape)
        t = torch.randint(1, self.n_steps, (batch.shape[0], 1))
        q_sample = self.q_sample(batch, t, epsilon)
        loss = torch.norm(epsilon - self.forward(q_sample, t))**2
        return loss/batch.shape[0]

    def sample(self, n_samples, progress=False, return_intermediate=False):
        """
        Implements inference step for the DDPM.
        `progress` is an optional flag to implement -- it should just show the current step in diffusion
        reverse process.
        If `return_intermediate` is `False`,
            the function returns a `n_samples` sampled from the learned DDPM
            i.e. a Tensor of size (n_samples, n_dim).
            Return: (n_samples, n_dim)(final result from diffusion)
        Else
            the function returns all the intermediate steps in the diffusion process as well 
            i.e. a Tensor of size (n_samples, n_dim) and a list of `self.n_steps` Tensors of size (n_samples, n_dim) each.
            Return: (n_samples, n_dim)(final result), [(n_samples, n_dim)(intermediate) x n_steps]
        """
        results = torch.zeros((n_samples, self.n_steps))
        x_t = torch.randn(n_samples, self.n_dim)
        results, intermediate = self.p_sample(x_t, self.n_steps)
        if (return_intermediate):
            return results, intermediate
        else:
            return results

    def configure_optimizers(self):
        """
        Sets up the optimizer to be used for backprop.
        Must return a `torch.optim.XXX` instance.
        You may choose to add certain hyperparameters of the optimizers to the `train.py` as well.
        In our experiments, we chose one good value of optimizer hyperparameters for all experiments.
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)