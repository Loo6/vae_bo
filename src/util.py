import numpy as np
from collections import defaultdict, Counter
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import selfies as sf

def text2dict_zinc_txt(data_dir, data_list, predict_prop = False):
    # This function is used to process the input data
    # taken from ZINC dataset
    #### OUTPUT #####
    # Dictionary of training, testing and validation data
    ## Sadegh Mohammadi, BCS, Monheim, 07.Nov.2018
    data_train = defaultdict(list)
    data_test = defaultdict(list)
    #data_valid = defaultdict(list)
    
    for data in data_list:
        print(data_dir+ data+'.txt')       
        with open(data_dir+ data+'.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('\t')
                smi = line[0].strip()
        
                try:
                    prop = np.double(line[1].strip())
                except:
                    prop = np.nan
                #mw = np.double(line[2].strip())
                #tpsa = np.double(line[1].strip())

                if data == data_list[0]:
                    data_train['SMILES'].append(smi)
                    data_train['prop'].append(prop)
                    #data_train['MW'].append(mw)
                    #data_train['tpsa'].append(tpsa)
                if data == data_list[1]:
                    data_test['SMILES'].append(smi)
                    data_test['prop'].append(prop)
                    #data_test['MW'].append(mw)
                    #data_test['tpsa'].append(tpsa)
                    
                #if data == data_list[2]:
                #    data_valid['SMILES'].append(smi)
                #    data_valid['logP'].append(logp)
                #    data_valid['MW'].append(mw)
    return data_train, data_test


def text2dict_zinc(datafile, properties = None):
    # This function is used to process the input data
    # taken from ZINC dataset
    #### OUTPUT #####
    # Dictionary of training, testing and validation data
    ## Sadegh Mohammadi, BCS, Monheim, 07.Nov.2018
    data = defaultdict(list)

    #print(os.path.join(data_dir, data_list[0]))
    #print(os.path.join(data_dir, data_list[1]))
    #train_df = pd.read_csv(os.path.join(data_dir, data_list[0]))
    #test_df = pd.read_csv(os.path.join(data_dir, data_list[1]))
    print(datafile)
    df = pd.read_csv(datafile, low_memory=False)
    print('Data:', df.shape[0])
    
    #print(train_df.columns)
    
    data['SMILES'] = df['smiles'].values

    if properties:
        print('Properties:', properties)
        data['prop'] = np.array(list(df[properties].values))
    else:
        data['prop'] = np.empty((len(data['SMILES']),))
        data['prop'].fill(np.nan)

    return data

# def smi_postprocessing(data, max_length, char2ind = None):
#     #print(data['SMILES'])
#     from collections import defaultdict
#     saver = defaultdict(list)
#     for indx, smi, prop in zip(np.arange(len(data['SMILES'])), data['SMILES'], data['prop']):
#         if len(smi) < max_length:
#             selfies = sf.encoder(smi)
#             selfies_new = "[G]" + selfies + "[E]"
#             if char2ind:
#                 try:
#                     [char2ind[char] for char in selfies_new]
#                     saver['SELFIES'].append((selfies_new))
#                     saver['prop'].append(prop)
#                 except Exception as e:
#                     print(e)
#                     print('Skipping ', selfies_new)
#                     continue
#             else:
#                 saver['SELFIES'].append((selfies_new))
#                 saver['prop'].append(prop)
#     #data_postproces={'SMILES':smi_list,'Effect':Effect}
#     return saver

def smi_postprocessing(data, max_length, char2ind=None):
    """
    将 SMILES 转为 SELFIES，并根据最大长度和字典映射进行筛选。
    使用 sf.split_selfies 切分成 token，再逐 token 检查映射。
    返回包含 'SELFIES' 列表和 'prop' 列表的字典。
    """
    saver = defaultdict(list)
    for smi, prop in zip(data['SMILES'], data['prop']):
        # 仅保留长度小于阈值的 SMILES
        if len(smi) >= max_length:
            continue

        # 编码为 SELFIES 串，并添加起始/终止标识
        sf_str = sf.encoder(smi)
        selfies_new = "[G]" + sf_str + "[E]"

        if char2ind:
            # 按 token 而非字符检查映射
            tokens = sf.split_selfies(selfies_new)
            try:
                for tok in tokens:
                    _ = char2ind[tok]
            except KeyError:
                print(f"Skipping invalid SELFIES (token not found in dict): {selfies_new}")
                continue

        # 通过检查或未传入 char2ind 时均保留
        saver['SELFIES'].append(selfies_new)
        saver['prop'].append(prop)

    return saver

# INCORRECT
def decode(smi):
    
    smi = smi.replace("L","Cl")
    smi = smi.replace("R","Br")
    smi = smi.replace("A","[nH]")
    smi = smi.replace('r','[nH+]')
    smi = smi.replace('a','[NH+]')
    smi = smi.replace('b','[NH2+]')
    smi = smi.replace('e','[NH3+]')
    smi = smi.replace("f","[N+]")
    smi = smi.replace("g","[O-]")
    smi = smi.replace("h","[Si]")
    smi = smi.replace("x","[C@]")
    smi = smi.replace("X","[C@@]")
    smi = smi.replace("X","[C@@]")
    smi = smi.replace("y","[C@H]")
    smi = smi.replace("Y","[C@@H]")
    smi = smi.replace("Q","[S@@]")
    smi = smi.replace("z","[S@]")
    smi = smi.replace("m","[P@@]")
    smi = smi.replace("k","[P@]")
    smi = smi.replace("i","[P@@H]") 
    smi = smi.replace('v','[N-]')
    smi = smi.replace('V','[n+]')
    smi = smi.replace('w','[n-]')
    smi = smi.replace('s','[S-]')
    smi = smi.replace('l','[NH-]')
    smi = smi.replace('q','[o+]')
   
    return smi

# CORRECT
def decode_1(smi):
    
    smi = smi.replace('r','[nH+]')
    smi = smi.replace('l','[NH-]')
    smi = smi.replace("i","[P@@H]") 
    smi = smi.replace("L","Cl")
    smi = smi.replace("R","Br")
    smi = smi.replace("A","[nH]")
    smi = smi.replace('a','[NH+]')
    smi = smi.replace('b','[NH2+]')
    smi = smi.replace('e','[NH3+]')
    smi = smi.replace("f","[N+]")
    smi = smi.replace("g","[O-]")
    smi = smi.replace("h","[Si]")
    smi = smi.replace("x","[C@]")
    smi = smi.replace("X","[C@@]")
    smi = smi.replace("X","[C@@]")
    smi = smi.replace("y","[C@H]")
    smi = smi.replace("Y","[C@@H]")
    smi = smi.replace("Q","[S@@]")
    smi = smi.replace("z","[S@]")
    smi = smi.replace("m","[P@@]")
    smi = smi.replace("k","[P@]")
    smi = smi.replace('v','[N-]')
    smi = smi.replace('V','[n+]')
    smi = smi.replace('w','[n-]')
    smi = smi.replace('s','[S-]')
    smi = smi.replace('q','[o+]')
   
    return smi


def check_all_key_in_test(data, char2ind):
    data_ = {}
    acc = []
    okchars = list(char2ind.keys())
    for smi in data['SMILES']:
        ok = all(c in okchars for c in smi)
        if ok:
            acc.append(smi)
    data_['SMILES'] = acc
    return data_
    

##### Physchem properties

def physchem_extract(data,physname):
    # here we extract the physchem properties
    if physname == 'MWt':
        MWt = np.asarray([ Descriptors.MolWt(Chem.MolFromSmiles(decode(smi[1:-1]))) for smi in data['SMILES']])
        min_MWt = np.min(MWt)
        max_MWt = np.max(MWt)
        return min_MWt, max_MWt, MWt
    elif physname == 'LogP':
        MLogP = np.asarray([Descriptors.MolLogP(Chem.MolFromSmiles(decode(smi[1:-1]))) for smi in data['SMILES']])
        min_MLogP = np.min(MLogP)
        max_MLogP = np.max(MLogP)
        return min_MLogP, max_MLogP, MLogP
    elif physname == 'MWt_LogP':
        MWt = np.asarray([Descriptors.MolWt(Chem.MolFromSmiles(decode(smi[1:-1]))) for smi in data['SMILES']])
        MLogP = np.asarray([Descriptors.MolLogP(Chem.MolFromSmiles(decode(smi[1:-1]))) for smi in data['SMILES']])
        physchems = np.vstack((MWt, MLogP)).transpose(1,0)
        min_phys = np.min(physchems,axis = 0)
        max_phys = np.max(physchems, axis = 0)
        return min_phys, max_phys, physchems
    else:
        raise ValueError('Keyword ' % (physname),' does not exist')
def max_min_norm(Descriptors,min_tr, max_tr):
    
    Descriptors_norm = (Descriptors - min_tr)/(max_tr - min_tr)
    
    return Descriptors_norm
def max_min_rescale(Descriptors,min_tr,max_tr):
    rescaling = (Descriptors*(max_tr - min_tr)) + min_tr
    return rescaling
    
def max_norm(Descriptors, max_tr):
    
    Descriptors_norm = (Descriptors )/(max_tr)
    
    return Descriptors_norm
    

def physchem_normalized(physchems):
    # The input is physchem properties, Mwt, MolLogP, TPSA,
    # Mwt_tr,Mlp_tr,Tpsa_tr: Physchem properties of the training set
    ####### OUTPUT #####
    # Normalized physchems ( Mwt, MolLogP, TPSA)
    from sklearn import preprocessing
    scaler   = preprocessing.StandardScaler().fit(physchems)
    physchemsn = scaler.transform(physchems)  
    return scaler,physchemsn   

def dictionary_build(selfies):
    # here we shape the dictionary given the raw smiles txt
    char2ind,ind2char=symdict(selfies)
    sos_indx = char2ind['[G]']
    eos_indx = char2ind['[E]']
    pad_indx = char2ind['[nop]']
    return char2ind,ind2char,sos_indx,eos_indx,pad_indx

def symdict(selfies):
    # This function is used for converting chemical symbols
    # to index and reverse one.
    #chars = ['<pad>'] + sorted(list(set(txt)))
    alphabet = set()
    for s in selfies:
        alphabet.update(sf.split_selfies(s))
    alphabet = ['[nop]'] + list(sorted(alphabet))
    print('total chars:', len(alphabet))
    symbol_to_idx = {s : i for i, s in enumerate(alphabet)}
    idx_to_symbol = {i : s for i, s in enumerate(alphabet)}
    return symbol_to_idx,idx_to_symbol

def char_weight(txt,char2ind):
    count = Counter(txt).most_common(len(char2ind))

    coocurrance = list((dict(count).values()))
    symbols = list((dict(count).keys()))
    lamda_factor =  np.log(coocurrance)/np.sum(np.log(coocurrance))
    lamda_factor = (1/(lamda_factor+0.000001))*0.01
    weights = {}
    for i,element in enumerate(symbols):
        if lamda_factor[i] > 1.:
            lamda_factor[i] = 0.90
        weights[element] = lamda_factor[i]

    #print(weights)
    class_weight = []
    for element in char2ind:
        if element == ' ':
            weight = 0.003
        else: 
            weight = weights[element]
        class_weight.append(weight)
    return class_weight 


def kl_anneal_function(anneal_function, step, k0, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k0*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

    
def randint(data,n_samples,rseed):
    import random 
    random.seed(rseed)
    idx_list = []
    while len(idx_list) < n_samples:
        idx = random.randint(1,len(data)-1)
        if idx not in idx_list:
            idx_list.append(idx)
    return idx_list
