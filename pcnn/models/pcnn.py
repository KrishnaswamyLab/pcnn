import torch
import torch.nn as nn
import torch_geometric
import pytorch_lightning as pl
from pcnn.models.layers import BaseLayer
from pcnn.utils import compute_sparse_diffusion_operator

class PCNN(pl.LightningModule):
    def __init__(self,num_layers, input_dim, hidden_dim, num_classes, lr, compute_P, K = None, **kwargs):
        
        super().__init__()

        self.save_hyperparameters()
        
        self.num_layers = num_layers

        self.lr = lr
        
        if kwargs['layer']['filter_method'] == "extract_scattering":
            J = kwargs['graph_construct']['J']
            n_norms = len(kwargs['graph_construct']['norm_list'])
            if kwargs["scattering_n_pca"] is not None:
                num_scattering_feats = kwargs["scattering_n_pca"]
            else: 
                num_scattering_feats = ((2 * n_norms) + int(0.5*(J*(1+J)) * n_norms )) * input_dim
            self.bypass_pooling = True
        else:
            num_scattering_feats = 0
            self.bypass_pooling = False
        
        self.layers = nn.ModuleList([BaseLayer(output_dim = hidden_dim, input_dim = input_dim, K = K, num_scattering_feats = num_scattering_feats, **kwargs["layer"])])
        if num_layers > 1:
            for _ in range(num_layers-1):
                self.layers = self.layers.append(BaseLayer(output_dim = hidden_dim, input_dim = hidden_dim, K= K, num_scattering_feats = num_scattering_feats,  **kwargs["layer"]))

        self.model = nn.Sequential(*self.layers)

        self.pooling = torch_geometric.nn.global_mean_pool

        output_dim = self.layers[-1].output_dim

        self.classifier = nn.Sequential(nn.Linear( output_dim, int(output_dim/2)), nn.ReLU(), nn.Linear(int(output_dim/2),num_classes))

        self.loss_fun = torch.nn.CrossEntropyLoss()

        self.num_classes = num_classes

        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.compute_P = compute_P # wether to compute the diffusion operator in the forward pass
        self.K = K #number of diffusion steps

    def forward(self, x):
        if self.compute_P: #Compute the powers of the diffusion operator
            P = compute_sparse_diffusion_operator(x)
            Pk_list = []
            Pk = P
            for k in range(self.K):
                Pk_list.append(Pk)
                Pk = torch.sparse.mm(Pk, P)
            x.Pk = Pk_list

        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        y = batch.y
        graph_out = self(batch)
        if not self.bypass_pooling:
            pooled = self.pooling(graph_out.x, graph_out.batch)
        else:
            pooled = graph_out.x.float()

        y_hat = self.classifier(pooled)

        loss = self.loss_fun(y_hat,y)
        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        y = batch.y
        graph_out = self(batch)
        
        if not self.bypass_pooling:
            pooled = self.pooling(graph_out.x, graph_out.batch)
        else:
            pooled = graph_out.x.float()
        y_hat = self.classifier(pooled)

        loss = self.loss_fun(y_hat,y)
        self.log('val_loss', loss)

        self.validation_step_outputs.append({'val_loss': loss,'y_hat': y_hat, 'y': y})
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs])
        acc = torch.sum(y_hat.argmax(dim=1) == y).item() / (len(y) * 1.0)
        self.log('val_acc', acc)
        self.validation_step_outputs.clear()

    
    def test_step(self, batch, batch_idx):
        y = batch.y
        graph_out = self(batch)
        if not self.bypass_pooling:
            pooled = self.pooling(graph_out.x, graph_out.batch)
        else:
            pooled = graph_out.x.float()
        y_hat = self.classifier(pooled)

        loss = self.loss_fun(y_hat,y)
        self.log('test_loss', loss)
        
        self.test_step_outputs.append({'test_loss': loss,'y_hat': y_hat, 'y': y})
        return loss
    
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs])
        acc = torch.sum(y_hat.argmax(dim=1) == y).item() / (len(y) * 1.0)
        self.log('test_acc', acc)
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)