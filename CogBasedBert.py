import torch
from torch import nn
from torch import tensor
from torch.nn.functional import softmax
from torch.autograd import Function
from transformers import RobertaTokenizer, RobertaModel
from torch.autograd import Variable
import torch


class GRL(Function):
    """
    GRL：
        The gradient is reversed during the backpropagation
    """
    @staticmethod
    def forward(ctx,i):
        result = i
        ctx.save_for_backward(result)
        return result
    @staticmethod
    def backward(ctx,grad_output):
        result, = ctx.saved_tensors
        return -grad_output*result

class Settings():
    """
    Parameters:
    ------------
    batch_size:
        The number of samples passed to the program for training at one time
    word_size:
        The dimensions of word embeddings from RoBERTa and the dimensions set for transformer encoder
    cog_size:
        The dimensions of cognitive signals input
    task_class:
        The number of sentiment classifications
    num_epoches:
        The number of iterations
    word_head_num:
        The number of multihead layers in word embeddings transformer encoder
    word_layer_num:
        The number of layers in word embeddings transformer encoder
    cog_head_num:
        The number of multihead layers in cognitive signals transformer encoder
    cog_layer_num:
        The number of layers in cognitive signals transformer encoder
    adv_head_num:
        The number of multihead layers in shared transformer encoder
    adv_layer_num:
        The number of layers in shared transformer encoder
    """
    def __init__(self):
        self.batch_size = 16
        self.dropout_prob = 0.3
        self.word_size = 300
        self.cog_size = 288
        self.task_class = 3
        self.num_epoches = 70
        self.word_head_num = 6
        self.word_layer_num = 6
        self.cog_head_num = 6
        self.cog_layer_num = 6
        self.adv_head_num = 3
        self.adv_layer_num = 3

class CogtoText(nn.Module):#set word_size to 300 dimension
    """
    CogtoText:
        Transform the cognitive signal dimension to be consistent with the word embeddings
    """
    def __init__(self,setting):
        super(CogtoText, self).__init__()
        self.input_size = setting.cog_size
        self.linear = nn.Linear(self.input_size,setting.word_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(setting.dropout_prob)

    def forward(self,input):
        self.input = self.linear(input)
        self.input = self.relu(self.input)
        self.output = self.dropout(self.input)
        return self.output

class Bert_Word(nn.Module):
    """
    Bert_Word:
        Extract word embeddings from RoBERTa
    """
    def __init__(self,setting):
        super(Bert_Word, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base',output_hidden_states=True)
        self.wordto300 = nn.Linear(768,setting.word_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(setting.dropout_prob)

    def forward(self,text_id,attention_masks):
        self.attention_masks = attention_masks
        self.text_id = text_id
        self.last_hidden_state, self.pooler_output,self._ = self.bert(self.text_id, attention_mask=self.attention_masks)
        self.last_hidden_state = self.dropout(self.relu(self.wordto300(self.last_hidden_state)))

        return self.last_hidden_state, self.pooler_output

class Spec_Word(nn.Module):
    """
    Spec_Word:
        Through the specific layers of word embeddings, the model learns specific features from word embeddings and then purify specific features
    """
    def __init__(self,setting):
        super(Spec_Word, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=setting.word_size,nhead=setting.word_head_num)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,num_layers=setting.word_layer_num)
        self.linear = nn.Linear(setting.word_size, setting.task_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(setting.dropout_prob)
    def forward(self,word):
        self.input = word
        self.input = torch.transpose(self.input, 0, 1)  # transformer_encoder:[seqlength,batch_size,dimension]
        self.input = self.transformer_encoder(self.input)
        self.input = torch.transpose(self.input, 0, 1)
        self.output = torch.mean(self.input,-2)
        self.output = self.dropout(self.relu(self.linear(self.output)))
        return self.input, self.output
    def compute_loss(self,word,label):
        self.loss_fn = nn.CrossEntropyLoss()
        self.input, self.output = self.forward(word)
        self.loss = self.loss_fn(self.output,label)
        return self.input,self.loss

class Com_Word(nn.Module):
    """
    Com_Word:
        Through the shared layer of word embeddings, the com_word learns common features from word embeddings
    """
    def __init__(self, setting):
        super(Com_Word, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=setting.word_size, nhead=setting.word_head_num)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=setting.word_layer_num)
        self.linear = nn.Linear(setting.word_size, setting.word_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(setting.dropout_prob)

    def forward(self, word):
        self.input = word
        self.input = torch.transpose(self.input, 0, 1)
        self.input = self.transformer_encoder(self.input)
        self.input = torch.transpose(self.input, 0, 1)
        self.output = self.dropout(self.relu(self.linear(self.input)))
        return self.output

class Spec_Cog(nn.Module):
    """
    Spec_Cog:
        Through the specific layers of cognitive signals, the model learns specific features from cognitive signals and then purify specific features
    """
    def __init__(self, setting):
        super(Spec_Cog, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=setting.word_size, nhead=setting.cog_head_num)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=setting.cog_layer_num)
        self.linear = nn.Linear(setting.word_size, setting.task_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(setting.dropout_prob)

    def forward(self, cog):
        self.input = cog
        self.input = torch.transpose(self.input, 0, 1)  # transformer_encoder:[seqlength,batch_size,dimension]
        self.input = self.transformer_encoder(self.input)
        self.input = torch.transpose(self.input, 0, 1)
        self.output = torch.mean(self.input,-2)
        self.output = self.dropout(self.relu(self.linear(self.output)))
        return self.input,self.output
    def compute_loss(self,cog,label):
        self.loss_fn = nn.CrossEntropyLoss()
        self.input, self.output = self.forward(cog)
        self.loss = self.loss_fn(self.output,label)
        return self.input,self.loss

class Com_Cog(nn.Module):
    """
    Com_Cog:
        Through the shared layer of cognitive signals, the com_cog learns common features from cognitive signals
    """
    def __init__(self, setting):
        super(Com_Cog, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=setting.word_size, nhead=setting.cog_head_num)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=setting.cog_layer_num)
        self.linear = nn.Linear(setting.word_size, setting.word_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(setting.dropout_prob)

    def forward(self, cog):
        self.input = cog
        self.input = torch.transpose(self.input, 0, 1)  # transformer_encoder:[seqlength,batch_size,dimension]
        self.input = self.transformer_encoder(self.input)
        self.input = torch.transpose(self.input, 0, 1)
        self.output = self.dropout(self.relu(self.linear(self.input)))
        return self.output

class Adv_Layer(nn.Module):
    """
    Adv_Layer:
        Through GRL layer, conduct adversarial learning between discriminator and generator(shared layer)
    """
    def __init__(self,setting):
        super(Adv_Layer, self).__init__()
        self.enconder_layer = nn.TransformerEncoderLayer(d_model=setting.word_size,nhead=setting.adv_head_num)
        self.transformer_encoder = nn.TransformerEncoder(self.enconder_layer,num_layers=setting.adv_layer_num)
        self.linear = nn.Linear(setting.word_size,2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(setting.dropout_prob)
    def forward(self,input):
        self.input = input
        self.input = torch.transpose(self.input, 0, 1)  # transformer_encoder:[seqlength,batch_size,dimension]
        self.input = self.transformer_encoder(self.input)
        self.input = torch.transpose(self.input, 0, 1)
        self.output = torch.mean(self.input,-2)
        self.output = GRL.apply(self.output)
        self.output = self.dropout(self.relu(self.linear(self.output)))
        return self.input,self.output
    def compute_loss(self,input,adv_label):
        self.loss_fn = nn.CrossEntropyLoss()
        self.input,self.output = self.forward(input)
        self.loss = self.loss_fn(self.output,adv_label)
        return self.input,self.loss

class Model(nn.Module):
    def __init__(self,setting):
        super(Model, self).__init__()
        self.cogtotext = CogtoText(setting)
        self.bert_word = Bert_Word(setting)
        self.spec_word = Spec_Word(setting)
        self.com_word = Com_Word(setting)
        self.spec_cog = Spec_Cog(setting)
        self.com_cog = Com_Cog(setting)
        self.adv_layer = Adv_Layer(setting)

        self.linear_cog = nn.Linear(setting.word_size*2,setting.word_size)
        self.cog_relu = nn.ReLU()

        self.cog_dropout = nn.Dropout(setting.dropout_prob)
        self.linear_word = nn.Linear(setting.word_size * 2, setting.word_size)
        self.word_relu = nn.ReLU()

        self.word_dropout = nn.Dropout(setting.dropout_prob)
        self.linear_cog_task = nn.Linear(setting.word_size,setting.task_class)
        self.linear_word_task = nn.Linear(setting.word_size, setting.task_class)

        self.dropout = nn.Dropout(setting.dropout_prob)

        self.U = Variable(torch.nn.init.xavier_normal_(torch.Tensor(setting.word_size,setting.word_size))).cuda()
        self.U.requires_grad_(True)
        self.tanh = nn.Tanh()
    def text_aware_attention(self,text,cog):
        self.text = torch.transpose(text, 1, 2)  # swap the column and the row of text matrix
        self.cog = torch.transpose(cog, 1, 2)  # swap the column and the row of cog matrix
        self.text_T = torch.matmul(torch.transpose(self.text, 1, 2), self.U)
        self.W = torch.tanh(torch.matmul(self.text_T, self.cog))
        self.score = softmax(self.W, 2)
        self.cog_T = torch.transpose(self.cog, 1, 2)
        self.outputs_T = torch.matmul(self.score, self.cog_T)
        self.outputs = self.outputs_T
        return self.outputs

    def forward(self,text_id,attention_masks,cog_input,label):
        self.text_id = text_id
        self.attention_masks = attention_masks
        self.cog_input = cog_input

        self.cog_input = self.cogtotext(self.cog_input)
        self.last_hidden_state, self.pooler_output = self.bert_word(self.text_id,self.attention_masks)
        self.cog_input = self.text_aware_attention(self.last_hidden_state,self.cog_input)

        self.special_word,self.word_loss = self.spec_word.compute_loss(self.last_hidden_state,label)
        self.common_word = self.com_word(self.last_hidden_state)
        self.special_cog,self.cog_loss = self.spec_cog.compute_loss(self.cog_input,label)
        self.common_cog = self.com_cog(self.cog_input)

        return self.special_word,self.word_loss,self.common_word,self.special_cog,self.cog_loss,self.common_cog
    def compute_orthogonality_loss(self,special_feature,common_feature):
        feature_product = torch.sum(special_feature*common_feature,dim=-1)
        special_feature_L2 = torch.norm(special_feature,p=2,dim=-1)
        common_feature_L2 = torch.norm(common_feature,p=2,dim=-1)
        L2 = special_feature_L2*common_feature_L2
        self.orthogonality_loss = torch.mean(abs(torch.sum(feature_product/L2,-1)))
        return self.orthogonality_loss

    def compute_loss(self,adv_label,special_feature,special_loss,common_feature,label,word,a,b,c):
        self.special_loss = special_loss
        self.orthogonality_loss = self.compute_orthogonality_loss(special_feature,common_feature)
        self.adv_feature,self.adv_loss = self.adv_layer.compute_loss(common_feature,adv_label)
        self.adv_feature = torch.mean(self.adv_feature,-2)
        self.special_feature = torch.mean(special_feature,-2)
        self.feature = torch.cat((self.adv_feature,self.special_feature),-1)
        self.feature = self.dropout(self.feature)
        self.loss_fn = nn.CrossEntropyLoss()
        if word:
            self.output = self.linear_word_task(self.word_dropout(self.word_relu(self.linear_word(self.feature))))
            self.sentiment_loss = self.loss_fn(self.output,label)
        else:
            self.output = self.linear_cog_task(self.cog_dropout(self.cog_relu(self.linear_cog(self.feature))))
            self.sentiment_loss = self.loss_fn(self.output, label)

        self.loss = self.sentiment_loss+a*self.special_loss+b*self.orthogonality_loss+c*self.adv_loss
        return self.loss,self.adv_loss
    def pred_result(self,text_id,attention_masks,cog_input,label):
        self.special_word,_,self.common_word,self.special_cog,___,self.common_cog\
            = self.forward(text_id,attention_masks,cog_input,label)
        self.adv_feature,_=self.adv_layer(self.common_word)
        self.adv_feature = torch.mean(self.adv_feature, -2)
        self.special_word = torch.mean(self.special_word, -2)
        self.feature = torch.cat((self.adv_feature,self.special_word),-1)
        self.output = self.linear_word_task(self.linear_word(self.feature))
        return self.output




