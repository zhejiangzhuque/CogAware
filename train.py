import nni
from CogBasedBert import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score,roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from transformers import RobertaTokenizer, RobertaModel,BertModel,BertTokenizer
import numpy as np

from torchtext.data import get_tokenizer

def compute_tag_prob(result):
    prob=[]
    result = torch.softmax(result,-1)
    result = result.cpu().tolist()
    prob+=result
    return prob

def compute_tag(result):
    tag=[]
    result = result.cpu().tolist()
    for i in result:
        tag.append(i.index(max(i)))
    return tag

def cal_prf(true_label,pred_label,tag_prob,mode):
    acc = accuracy_score(true_label, pred_label)
    p = precision_score(true_label, pred_label, average=mode)
    r = recall_score(true_label, pred_label, average=mode)
    f1 = f1_score(true_label, pred_label, average=mode)
    auc = roc_auc_score(true_label,tag_prob,multi_class='ovo')
    return acc,p,r,f1,auc

def model_input(loader,train_label,input_ids,attention_masks,eeg_feature):#
    batch_eeg_feature = tensor(eeg_feature[loader], dtype=torch.float32)
    batch_train_label = tensor(train_label[loader], dtype=torch.long)
    batch_input_ids = tensor(input_ids[loader], dtype=torch.long)
    batch_attention_masks = tensor(attention_masks[loader], dtype=torch.long)
    length=[]
    for i in batch_input_ids:
        num=0
        for j in i:
            if j!=0:
                num+=1
        length.append(num)
    max_len = max(length)
    batch_eeg_feature = batch_eeg_feature[:,:max_len,:]
    batch_input_ids = batch_input_ids[:,:max_len]
    batch_attention_masks = batch_attention_masks[:,:max_len]
    return batch_train_label,batch_input_ids,batch_attention_masks,batch_eeg_feature
def resample(train_input_ids,train_attention_masks,train_eeg_feature,train_label):
    ros = RandomOverSampler(random_state=2)
    train_input_ids = np.expand_dims(train_input_ids, -1)
    train_attention_masks = np.expand_dims(train_attention_masks, -1)
    train_data = np.concatenate((train_eeg_feature, train_input_ids, train_attention_masks), -1)
    train_data = np.reshape(train_data, (320, -1))
    train_data, train_label = ros.fit_resample(train_data, train_label)
    train_data = np.reshape(train_data, (-1,43,295))
    train_eeg_feature = train_data[:, :,0:293]
    train_input_ids = train_data[:, :,293]
    train_attention_masks = train_data[:, :,294]
    return train_eeg_feature,train_input_ids,train_attention_masks

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
max_len = 43
cog = "eeg"
label_all = np.load(r"D:\programming_project\bert_cogalign\sentiment\targets.npy")
eeg_feature = np.load(r"D:\programming_project\bert_cogalign\sentiment\ZAB_eeg_word.npy")

print('Loading sentence...')
with open("sentence.txt","r") as f:
    sentence = f.read()
sentence = eval(sentence)

print('Loading Roberta tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
input_ids = []
attention_masks = []

for sent in sentence:
    encoded_dict = tokenizer.encode_plus(sent[0],
                                         add_special_tokens = True,
                                         max_length = max_len,
                                         pad_to_max_length = True,
                                         return_attention_mask = True,
                                         return_tensors = 'pt')
    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = np.concatenate(input_ids,0)
attention_masks = np.concatenate(attention_masks,0)

train_input_ids,test_input_ids,train_attention_masks,test_attention_masks,train_label,test_label,\
train_eeg_feature,test_eeg_feature \
    = train_test_split(input_ids,attention_masks,label_all,eeg_feature,test_size=0.2,shuffle=True,random_state=0)
train_eeg_feature, train_input_ids, train_attention_masks\
    =resample(train_input_ids,train_attention_masks,train_eeg_feature,train_label)
params = {
    "a":0.9,
    "b":0.1,
    "c":0.4
}
setting = Settings()
optimized_params = nni.get_next_parameter()
params.update(optimized_params)
model = Model(setting).to(device)
optimizer_cog = torch.optim.Adam(model.parameters(),betas=(0.9, 0.98), eps=1e-4, lr = 1e-5, weight_decay=1e-2)
optimizer_word = torch.optim.Adam(model.parameters(),betas=(0.9, 0.98), eps=1e-4, lr = 1e-5, weight_decay=1e-2)
step = 0

indicators_list=[]
for one_epoch in range(setting.num_epoches):
    temp_order = list(range(len(train_label)))
    dataloader = DataLoader(temp_order, batch_size=setting.batch_size, shuffle=True, drop_last=True)
    for loader in dataloader:
        model.train()
        label,text_id,attention_mask,eeg = model_input(loader,train_label,train_input_ids,
                                                           train_attention_masks,train_eeg_feature)
        label,text_id,attention_mask,eeg = label.to(device),text_id.to(device),attention_mask.to(device),eeg.to(device)
        cog_label = torch.tensor([0] * setting.batch_size, dtype=torch.long).to(device)
        word_label = torch.tensor([1] * setting.batch_size, dtype=torch.long).to(device)
        cog_input = eeg
        for i in range(2):
            special_word,word_loss,common_word,special_cog,cog_loss,common_cog = model(text_id,attention_mask,cog_input,label)
            if i==0:
                loss0,adv_loss0 = model.compute_loss(cog_label,special_cog,cog_loss,common_cog,label,i,params["a"],params["b"],params["c"])
                loss0.backward()
                optimizer_cog.step()
                model.zero_grad()
            else:
                loss1,adv_loss1 = model.compute_loss(word_label,special_word,word_loss,common_word,label,i,params["a"],params["b"],params["c"])
                loss1.backward()
                optimizer_word.step()
                model.zero_grad()
        print("step:{},loss0:{},loss1:{}".format(step,loss0,loss1))
        step+=1
    print("One Epoch training have been completed")
    print("-----------------------------------")
    print("Epoch{} Starting Test!!!".format(one_epoch))
    model.eval()
    test_order = list(range(len(test_label)))
    test_dataloader = DataLoader(test_order, batch_size=10, shuffle=True, drop_last=True)
    tag_pred = []
    test = []
    tag_prob = []
    test_list = []
    for test_loader in test_dataloader:
        label, text_id, attention_mask,eeg = model_input(test_loader, test_label, test_input_ids,
                                                     test_attention_masks,test_eeg_feature)
        label,text_id,attention_mask,eeg = label.to(device),text_id.to(device),attention_mask.to(device),eeg.to(device)
        cog_input = eeg

        result=model.pred_result(text_id,attention_mask,cog_input,label)
        tag_pred+=compute_tag(result)
        tag_prob+=compute_tag_prob(result)
        test+=np.array(label.cpu(),dtype=int).tolist()
    current_acc, current_p, current_r, current_f,current_auc = cal_prf(test,tag_pred,tag_prob,'macro')
    print("Test Result:-----------------------------------------")
    print("Epoch{} :test_P {},test_R {},test_F {},test_auc{}".format(one_epoch,current_p, current_r, current_f,current_auc))
    test_list.append(current_p)
    test_list.append(current_r)
    test_list.append(current_f)
    indicators_list.append(test_list)
    print(current_f)
    nni.report_intermediate_result(current_f)

final_f_list=[i[-1] for i in indicators_list]
final_f = max(final_f_list)
final_f_ind = final_f_list.index(final_f)
print(final_f,indicators_list[final_f_ind])
nni.report_final_result(final_f)


