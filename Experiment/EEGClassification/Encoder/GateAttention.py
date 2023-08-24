import torch
from torch import nn

import torch
# import torchtext
import torch.distributions
from torch import nn 
import torch.nn.functional as F

class AuxiliaryNet(torch.nn.Module):
    """
    Arguments
    ---------
    batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
    aux_hidden_size : Size of the hidden_state of the LSTM   (* Later BiLSTM, check dims for BiLSTM *)
    embedding_length : Embeddding dimension of GloVe word embeddings
    --------
    """
    
    def __init__(self, batch_size, auxiliary_hidden_size, embedding_length, biDirectional = False, num_layers = 1, tau=1):
        super(AuxiliaryNet, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = auxiliary_hidden_size
        self.embedding_length = embedding_length	
        self.biDirectional	= biDirectional
        self.num_layers = num_layers

        self.aux_lstm = nn.LSTM(self.embedding_length, self.hidden_size, bidirectional = self.biDirectional, num_layers = self.num_layers, batch_first = True)   # Dropout  
        if(self.biDirectional):
            self.aux_linear = nn.Linear(self.hidden_size * 2,1)
        else:
            self.aux_linear = nn.Linear(self.hidden_size,1)
            self.sigmoid = torch.nn.Sigmoid()
            self.tau = tau
            

    def forward(self, input_sequence, is_train = True, batch_size=None):
        
        # input : Dimensions (batch_size x seq_len x embedding_length)

        out_lstm, (final_hidden_state, final_cell_state) = self.aux_lstm(input_sequence)   # ouput dim: ( batch_size x seq_len x hidden_size ) 
        out_linear = self.aux_linear(out_lstm)                                               # p_t dim: ( batch_size x seq_len x 1)
        p_t = self.sigmoid(out_linear)

        if is_train:
            p_t = p_t.repeat(1,1,2)
            p_t[:,:,0] = 1 - p_t[:,:,0] 
            g_hat = F.gumbel_softmax(p_t, self.tau, hard=False)   
            g_t = g_hat[:,:,1]
        
        else:
            # size : same as p_t [ batch_size x seq_len x 1]
            m = torch.distributions.bernoulli.Bernoulli(p_t)   
            g_t = m.sample()

        return g_t


class BackboneNet(torch.nn.Module):
    """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        backbone_hidden_size : Size of the hidden_state of the LSTM   (* Later BiLSTM, check dims for BiLSTM *)
        embedding_length : Embeddding dimension of GloVe word embeddings
        --------
        """
    def __init__(self, batch_size, backbone_hidden_size, embedding_length, biDirectional = False, num_layers = 2):

        super(BackboneNet, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = backbone_hidden_size
        self.embedding_length = embedding_length
        self.biDirectional  = biDirectional
        self.num_layers = num_layers

        self.backbone_lstm = nn.LSTM(self.embedding_length, self.hidden_size, bidirectional = self.biDirectional, batch_first = True, num_layers = self.num_layers)   # Dropout  

    def forward(self, input_sequence, batch_size=None):
        out_lstm, (final_hidden_state, final_cell_state) = self.backbone_lstm(input_sequence)   # ouput dim: ( batch_size x seq_len x hidden_size )
        return out_lstm


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.ff_1 = nn.Linear(self.input_dim, self.output_dim)
        self.relu = nn.ReLU()
        self.ff_2 = nn.Linear(self.output_dim,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out_1 = self.ff_1(x)
        out_relu = self.relu(out_1)
        out_2 = self.ff_2(out_relu)
        out_sigmoid = self.sigmoid(out_2)

        return out_sigmoid 


class GANet(torch.nn.Module):
		def __init__(self, batch_size, num_classes, mlp_out_size, vocab_size, embedding_length, weights, aux_hidden_size = 100, backbone_hidden_size = 100, biDirectional_aux = False, biDirectional_backbone = False):
			super(GANet, self).__init__() 
			"""
			Arguments
			---------
			batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
			output_size : 6 = (For TREC dataset)
			hidden_sie : Size of the hidden_state of the LSTM   (// Later BiLSTM)
			vocab_size : Size of the vocabulary containing unique words
			embedding_length : Embeddding dimension of GloVe word embeddings
			weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

			--------

			"""

			self.batch_size = batch_size
			self.num_classes = num_classes
			self.vocab_size = vocab_size
			self.embedding_length = embedding_length
			self.aux_hidden_size = aux_hidden_size
			self.backbone_hidden_size = backbone_hidden_size 
			self.mlp_out_size = mlp_out_size
			self.biDirectional_aux = biDirectional_aux
			self.biDirectional_backbone = biDirectional_backbone

			self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
			self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)

			self.auxiliary = AuxiliaryNet(self.batch_size, self.aux_hidden_size, self.embedding_length, self.biDirectional_aux)
			self.backbone = BackboneNet(self.batch_size, self.backbone_hidden_size, self.embedding_length, self.biDirectional_backbone)
			if(self.biDirectional_backbone):
				self.mlp = MLP(self.backbone_hidden_size * 2, self.mlp_out_size)
				self.FF = nn.Linear(self.backbone_hidden_size * 2,num_classes)
			else:
				self.mlp = MLP(self.backbone_hidden_size, self.mlp_out_size)
				self.FF = nn.Linear(self.backbone_hidden_size,num_classes)
			self.softmax = nn.Softmax()
			

		def masked_Softmax(self, logits, mask):
			# print("type of mask", type(mask))
			# print("gt size", mask.shape)
			mask_bool = mask >0
			logits[mask_bool] = float('-inf')
			return torch.softmax(logits, dim=1)	

		def forward(self,input_sequence, is_train = True):
			input_ = self.word_embeddings(input_sequence)
			g_t = self.auxiliary(input_, is_train)
			out_lstm = self.backbone(input_)

			if is_train:
				e_t = self.mlp(out_lstm)
				alpha = self.softmax(e_t)
			else:
				e_t = self.mlp(out_lstm)               # change if possible!
				alpha = self.masked_Softmax(e_t, g_t)

			c_t = torch.bmm(alpha.transpose(1,2), out_lstm)
			logits = self.FF(c_t)
			final_output = self.softmax(logits)
			# final_output = final_output.max(2)[1]
			final_output = final_output.squeeze(1)

			return final_output, g_t