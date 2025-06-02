import codesecurity.tasks.code_authorship_attribution.preprocessing as caa_data
import codesecurity.tasks.code_authorship_attribution.caches_manager as caa_caches
import codesecurity.tasks.code_authorship_attribution.preprocessing as cacc_preprocess
from codesecurity.utils.check_symbol_log import get_symbol_logger

import os
import torch
import pickle
import random
import sklearn.feature_selection as skfs
import sklearn.preprocessing as skp
import numpy as np

import torch.utils.data as torchdata
logger = get_symbol_logger("symbol_module",log_file="symbol_module.log")
class ForseeDataset(torchdata.Dataset):
    def __init__(self,features:caa_data.ForseeFeatures) -> None:
        super().__init__()
        self.features=features
        if features.id_mapping is None:
            self.class_number=None
        else:
            self.class_number=len(features.id_mapping)
        
        # layout_features=[e.layout for e in features.samples]
        # self.layout_scalar=skp.MinMaxScaler()
        # self.layout_scalar.fit(layout_features)

        # lexcial_features=[e.lexical for e in features.samples]
        # self.lexical_scalar=skp.MinMaxScaler()
        # self.lexical_scalar.fit(lexcial_features)

        # syntactic_features=[e.syntactic for e in features.samples]
        # self.syntactic_scalar=skp.MinMaxScaler()
        # self.syntactic_scalar.fit(syntactic_features)


    def __getitem__(self, index):
        sample=self.features.samples[index]
        #std_layout=self.layout_scalar.transform([sample.layout])[0]
        #std_lexical=self.lexical_scalar.transform([sample.lexical])[0]
        #std_syntactic=self.syntactic_scalar.transform([sample.syntactic])[0]
        return sample.layout,sample.lexical,sample.syntactic,sample.symbol,sample.label
        
    def __len__(self):
        return len(self.features.samples)

class DLCAISDataset(torchdata.Dataset):
    def __init__(self,features:caa_data.ForseeFeatures,sp:caa_data.DLCAISSuperParameter) -> None:
        super().__init__()
        self.features=features
        if features.id_mapping is None:
            self.class_number=None
        else:
            self.class_number=len(features.id_mapping)

        self.init(features,sp)
        
    def init(self,features:caa_data.ForseeFeatures,sp:caa_data.DLCAISSuperParameter):
        #self.scalar=skp.MinMaxScaler()
        lexical_features=[e.lexical for e in features.samples]
        #self.scalar.fit(lexical_features)
        labels=[e.label for e in features.samples]
        # lexical_features=lexical_features[:100]
        # labels=labels[:100]

        # print(lexical_features)
        # print(labels)
        if sp.input_dim<len(lexical_features[0]):
            mutux_info=skfs.mutual_info_classif(lexical_features,labels,random_state=17)
            select_locations=-mutux_info.argsort()[:sp.input_dim]
        else:
            select_locations=list(range(len(lexical_features[0])))

        self.select_locations=select_locations
        
        #return mutux_info
    
    def get_feature_matrix(self):
        matrix=np.zeros((len(self.features.samples),len(self.select_locations)))
        for i in range(len(self.features.samples)):
            matrix[i]=self.features.samples[i].lexical[self.select_locations]
        
        return matrix
    
    def get_labels(self):
        return np.array([e.label for e in self.features.samples])
    
    def __getitem__(self, index):
        #std_lexical=self.scalar.transform([self.features.samples[index].lexical])[0]
        return self.features.samples[index].lexical[self.select_locations],self.features.samples[index].label
        
    def __len__(self):
        return len(self.features.samples)

class ForseePartialDataset(torchdata.Dataset):
    def __init__(self,forsee_dataset:ForseeDataset,use_layout,use_lexical,use_syntactic) -> None:
        super().__init__()
        self.forsee_dataset=forsee_dataset
        self.feature_indexes=[]
        if use_layout:
            self.feature_indexes.append(0)
        if use_lexical:
            self.feature_indexes.append(1)
        if use_syntactic:
            self.feature_indexes.append(2)
        
    def __getitem__(self, index):
        v,label=self.forsee_dataset[index][:-1],self.forsee_dataset[index][-1]
        return *[v[i] for i in self.feature_indexes],label
        
    def __len__(self):
        return len(self.forsee_dataset)

class ForseeLayoutDataset(torchdata.Dataset):
    def __init__(self,forsee_dataset:ForseeDataset) -> None:
        super().__init__()
        self.forsee_dataset=forsee_dataset
        
    def __getitem__(self, index):
        v,label=self.forsee_dataset[index][:-1],self.forsee_dataset[index][-1]
        return v[0],label
        
    def __len__(self):
        return len(self.forsee_dataset)

class ForseeSymbolDataset(torchdata.Dataset):
    def __init__(self, forsee_dataset:ForseeDataset) -> None:
        super().__init__()
        self.forsee_dataset = forsee_dataset
        
    def __getitem__(self, index):
        # 修改确保返回的是张量而不是列表
        v, label = self.forsee_dataset[index][:-1], self.forsee_dataset[index][-1]
        # logger.var_info(v[3],"v[3]")
        # logger.var_info(label,"label")
        return v[3], label
        # if isinstance(v[3], list):
        #     symbol_tensor = torch.tensor(v[3], dtype=torch.float32)
        # elif isinstance(v[3], torch.Tensor):
        #     symbol_tensor = v[3].float()
        # return symbol_tensor, label
        
    def __len__(self):
        return len(self.forsee_dataset)

class ForseeLexicalDataset(torchdata.Dataset):
    def __init__(self,forsee_dataset:ForseeDataset) -> None:
        super().__init__()
        self.forsee_dataset=forsee_dataset
        
    def __getitem__(self, index):
        v,label=self.forsee_dataset[index][:-1],self.forsee_dataset[index][-1]
        return v[1],label
        
    def __len__(self):
        return len(self.forsee_dataset)
    
class ForseeSyntacticDataset(torchdata.Dataset):
    def __init__(self,forsee_dataset:ForseeDataset) -> None:
        super().__init__()
        self.forsee_dataset=forsee_dataset
        
    def __getitem__(self, index):
        v,label=self.forsee_dataset[index][:-1],self.forsee_dataset[index][-1]
        return v[2],label
        
    def __len__(self):
        return len(self.forsee_dataset)

def layout_enhance(layout_tensors:list,threshold=4):
    
    for i,layout_tensor in enumerate(layout_tensors):
        #if len(layout_tensor.shape)==0: continue
        origin_length=layout_tensor.shape[1]
        for j in range(layout_tensor.shape[0]):
            if threshold>=1:
                start=random.randint(0,int(threshold))
            else:
                start=random.randint(0,int(origin_length*threshold))
            temp=layout_tensor[j,start:].clone()
            layout_tensor[j,:origin_length-start]=temp
            layout_tensor[j,origin_length-start:]=0
    return layout_tensors
# 定义symbol_enhance
def symbol_enhance(symbol_tensors: list, threshold=0.1):
    """
    对符号特征进行数据增强
    
    Args:
        symbol_tensors: 符号特征张量列表
        threshold: 扰动阈值
    
    Returns:
        增强后的符号特征张量
    """
    for tensor in symbol_tensors:
        if len(tensor.shape) == 0:
            continue
            
        # 创建扰动掩码矩阵
        mask = torch.rand(tensor.shape) < threshold
        
        # 区分不同类型的特征
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                if not mask[i, j]:
                    continue
                    
                feature_value = tensor[i, j].item()
                
                # 比例特征 (索引1-6, 8-10, 12-14, 16, 18-21)
                ratio_indices = [1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 14, 16, 18, 19, 20, 21]
                if j in ratio_indices:
                    # 对比例特征进行小幅度随机扰动，保持在0-1范围内
                    noise = (torch.rand(1).item() - 0.5) * 0.2  # 生成-0.1到0.1之间的噪声
                    tensor[i, j] = torch.clamp(tensor[i, j] + noise, 0.0, 1.0)
                
                # 总数特征 (索引0, 7, 11, 15, 17, 22)
                else:
                    # 对计数特征进行随机上下波动
                    if feature_value > 0:
                        # 对非零计数值进行扰动
                        noise_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.4  # 0.8到1.2之间的乘性噪声
                        tensor[i, j] = tensor[i, j] * noise_factor
    
    return symbol_tensors


def lexical_enhance(lexical_tensors:list,threshold=0.1):
    for lexical_tensor in lexical_tensors:
        lexical_tensor=lexical_tensor[0]
        mask=torch.rand(lexical_tensor.shape)<threshold
        lexical_tensor[mask]=0

    return lexical_tensors

def syntactic_enhance(syntactic_tensors:torch.Tensor,threshold=0.1):
    for syntactic_tensor in syntactic_tensors:
        mask=torch.rand(syntactic_tensor.shape)<threshold
        syntactic_tensor[mask]=0

    return syntactic_tensors

def combine_enhance(inputs):
    if len(inputs)==3:
        layout_tensor,lexical_tensor,syntactic_tensor=inputs
        layout_tensor=layout_enhance([layout_tensor])
        lexical_tensor=lexical_enhance([lexical_tensor])
        syntactic_tensor=syntactic_enhance([syntactic_tensor])

        return [layout_tensor,lexical_tensor,syntactic_tensor]
    else:
         # 新增符号特征版本
        layout_tensor, lexical_tensor, syntactic_tensor, symbol_tensor = inputs
        layout_tensor = layout_enhance([layout_tensor])
        lexical_tensor = lexical_enhance([lexical_tensor])
        syntactic_tensor = syntactic_enhance([syntactic_tensor])
        symbol_tensor = symbol_enhance([symbol_tensor])
        return [layout_tensor, lexical_tensor, syntactic_tensor, symbol_tensor]

def prepare_main_dataset(dataset_dir,lang,list_author_handle,id_mapping,sp,forsee_dataset,raw_dataset,refine_dataset,lexical_file,syntactic_file):
    if os.path.exists(refine_dataset):
        with open(refine_dataset,'rb') as f:
            refine_dataset=pickle.load(f)
        
        return ForseeDataset(refine_dataset)
    
    
    if os.path.exists(raw_dataset):
        refine_dataset_obj=cacc_preprocess.ForseeFeatureBuilder.build_from_pipe(sp,lexical_file=lexical_file,syntactic_file=syntactic_file,raw_data=raw_dataset)
        refine_dataset_obj.save(refine_dataset)
        
        return ForseeDataset(refine_dataset_obj)
    
    refine_dataset_path=refine_dataset
    refine_dataset=cacc_preprocess.prepare_forsee_features(dataset_dir,None,sp,list_author_handle,id_mapping,lexical_file=lexical_file,syntactic_file=syntactic_file,raw_data=raw_dataset)
    random.shuffle(refine_dataset.samples)
    refine_dataset.save(refine_dataset_path)
    
    return ForseeDataset(refine_dataset)

def prepare_subdataset(dataset_dir,lang,list_author_handle,id_mapping,sp,forsee_dataset,raw_dataset,refine_dataset,lexical_file,syntactic_file):
    if os.path.exists(refine_dataset):
        with open(refine_dataset,'rb') as f:
            refine_dataset=pickle.load(f)
        
        return ForseeDataset(refine_dataset)
    
    if os.path.exists(raw_dataset):
        refine_dataset_obj=cacc_preprocess.ForseeFeatureBuilder.build_from_pipe(sp,lexical_file=lexical_file,syntactic_file=syntactic_file,raw_data=raw_dataset)
        refine_dataset_obj.save(refine_dataset)
        
        return ForseeDataset(refine_dataset_obj)
    
    refine_dataset_obj=cacc_preprocess.prepare_forsee_features_for_external_data(dataset_dir,sp,id_mapping,lexical_file,syntactic_file,raw_dataset,list_author_handle)
    refine_dataset_obj.save(refine_dataset)
        
    return ForseeDataset(refine_dataset_obj)

def prepare_external_data(dataset_dir,caches_file,list_author_handle,id_mapping,sp,lexical_file,syntactic_file,min_number=1,max_number=2**32-1):
    refine_dataset_obj=cacc_preprocess.prepare_forsee_features_for_external_data(dataset_dir,sp,id_mapping,lexical_file,syntactic_file,caches_file,list_author_handle,min_number,max_number)
    
    return ForseeDataset(refine_dataset_obj)

def prepare_training_set(dataset_dir,lang,list_author_handle,meta:caa_caches.ForseeCachesMetatadata,sp:cacc_preprocess.ForseeSuperParameter):
    # this object type is str
    forsee_dataset=meta.training_default_task_data_file
    raw_dataset=meta.training_raw_data_file
    refine_dataset=meta.training_refine_data_file
    lexical_file=meta.lexical_file
    syntactic_file=meta.syntactic_file
    
    dataset=prepare_main_dataset(dataset_dir,lang,list_author_handle,None,sp,forsee_dataset,raw_dataset,refine_dataset,lexical_file,syntactic_file)
    
    return dataset

def prepare_eval_set(dataset_dir,lang,list_author_handle,id_mapping,meta:caa_caches.ForseeCachesMetatadata,sp:cacc_preprocess.ForseeSuperParameter):
    # this object type is str
    forsee_dataset=meta.test_default_task_data_file
    raw_dataset=meta.test_raw_data_file
    refine_dataset=meta.test_refine_data_file
    lexical_file=meta.lexical_file
    syntactic_file=meta.syntactic_file
    return prepare_subdataset(dataset_dir,lang,list_author_handle,id_mapping,sp,forsee_dataset,raw_dataset,refine_dataset,lexical_file,syntactic_file) 

# def prepare_test_set(dataset_dir,caches_dir,list_author_handle,id_mapping,meta:caa_caches.ForseeCachesMetatadata,sp:cacc_preprocess.ForseeSuperParameter):
#     pass

def prepare_fold_set(dataset_dir,lang,list_author_handle,meta:caa_caches.ForseeCachesMetatadata,sp:cacc_preprocess.ForseeSuperParameter,k):
    full_dataset=prepare_training_set(dataset_dir,lang,list_author_handle,meta,sp)
    
    training_set,eval_set=fold_dataset(full_dataset,k,42)
    
    return training_set,eval_set,full_dataset.features.id_mapping

def prepare_single_data(meta:caa_caches.ForseeCachesMetatadata,repo_dirs,labels,sp:cacc_preprocess.ForseeSuperParameter,lang=None,min_number=1,max_number=2**32-1):
    return ForseeDataset(cacc_preprocess.build_forsee_features_for_single_repo(repo_dirs,labels,sp,None,meta.lexical_file,meta.syntactic_file,lang,min_number=min_number,max_number=max_number))

def fold_dataset(full_dataset:torchdata.Dataset,k,seed=42):
    split_number=[-1,int(len(full_dataset)/k)]
    split_number[0]=len(full_dataset)-split_number[1]
    #print(split_number)
    #print(full_dataset[0])
    
    indexs=list(range(len(full_dataset)))
    random.seed(seed)
    random.shuffle(indexs)
    
    training_set,eval_set=torchdata.Subset(full_dataset,indexs[0:split_number[0]]),torchdata.Subset(full_dataset,indexs[split_number[0]:split_number[0]+split_number[1]])
    
    return training_set,eval_set